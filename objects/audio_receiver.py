#!/usr/bin/env python3
import time
import subprocess
import threading
import os
import sys
from collections import deque
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

class AudioReceiver:
    """
    Captures live audio from an ALSA microphone input in a non-blocking background thread.
    Maintains a thread-safe rolling buffer of audio samples.
    Uses 'arecord' or 'ffmpeg' as an external process to capture raw 16-bit PCM.
    """
    def __init__(self, alsa_device="hw:1,0", sample_rate=48000, channels=1, max_buffer_seconds=10):
        self.alsa_device = alsa_device
        self.sample_rate = sample_rate
        self.channels = channels
        self.max_buffer_seconds = max_buffer_seconds
        
        # Calculate max samples to hold
        self.buffer_size = int(sample_rate * channels * max_buffer_seconds)
        self.audio_buffer = deque(maxlen=self.buffer_size)
        self.lock = threading.Lock()
        
        self.process = None
        self.thread = None
        self.stopped = False
        
    def _get_sharing_capture_device(self):
        """Converts direct hardware index (hw:1,0) to sharing-safe plug:dsnoop:1."""
        if self.alsa_device.startswith("hw:"):
            parts = self.alsa_device[3:].split(",")
            card_idx = parts[0]
            return f"plug:dsnoop:{card_idx}"
        return self.alsa_device

    def start(self):
        """Starts the background thread to capture ALSA audio."""
        self.stopped = False
        capture_device = self._get_sharing_capture_device()
        
        arecord_cmd = [
            "arecord",
            "-D", capture_device,
            "-c", str(self.channels),
            "-r", str(self.sample_rate),
            "-f", "S16_LE",
            "-t", "raw",
            "-"
        ]
        
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "alsa",
            "-channels", str(self.channels),
            "-ar", str(self.sample_rate),
            "-i", capture_device,
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "pipe:1"
        ]
        
        print(f"[AudioReceiver] Initializing capture from {capture_device} ({self.sample_rate}Hz)...")
        try:
            self.process = subprocess.Popen(
                arecord_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=4096
            )
            time.sleep(0.2)
            if self.process.poll() is not None:
                print("[AudioReceiver] 'arecord' exited early. Retrying with 'ffmpeg' capture backend...")
                self.process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=4096
                )
        except Exception as e:
            print(f"[AudioReceiver] Failed to start 'arecord' ({e}). Trying 'ffmpeg'...", file=sys.stderr)
            try:
                self.process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=4096
                )
            except Exception as ex:
                print(f"[AudioReceiver] CRITICAL: Both capture backends failed to start: {ex}", file=sys.stderr)
                raise RuntimeError("Could not initialize live audio capture backend.")

        self.thread = threading.Thread(target=self._capture_loop, name="AudioCaptureThread", daemon=True)
        self.thread.start()
        return self

    def _capture_loop(self):
        """Continuously reads PCM bytes from the capture subprocess stdout and stores in the deque."""
        bytes_per_sample = 2  # 16-bit PCM = 2 bytes
        chunk_samples = 1024
        chunk_bytes = chunk_samples * self.channels * bytes_per_sample
        
        while not self.stopped:
            if not self.process or self.process.poll() is not None:
                if not self.stopped:
                    print("[AudioReceiver] WARNING: Audio process died. Re-initializing in 1s...", file=sys.stderr)
                    time.sleep(1.0)
                    try:
                        self.start()
                    except Exception:
                        pass
                break
                
            try:
                raw_data = self.process.stdout.read(chunk_bytes)
                if not raw_data:
                    time.sleep(0.01)
                    continue
                    
                samples = np.frombuffer(raw_data, dtype=np.int16)
                samples_float = samples.astype(np.float32) / 32768.0
                
                with self.lock:
                    self.audio_buffer.extend(samples_float)
            except Exception as e:
                if not self.stopped:
                    print(f"[AudioReceiver] Error reading audio stream: {e}", file=sys.stderr)
                time.sleep(0.1)
                
    def get_audio_window(self, seconds):
        """Returns the most recent N seconds of audio as a NumPy array."""
        samples_needed = int(self.sample_rate * self.channels * seconds)
        with self.lock:
            buffer_len = len(self.audio_buffer)
            if buffer_len == 0:
                return np.zeros(samples_needed, dtype=np.float32)
                
            if buffer_len < samples_needed:
                padding = np.zeros(samples_needed - buffer_len, dtype=np.float32)
                data = np.array(self.audio_buffer, dtype=np.float32)
                return np.concatenate((padding, data))
            else:
                deque_list = list(self.audio_buffer)
                return np.array(deque_list[-samples_needed:], dtype=np.float32)

    def stop(self):
        """Cleanly stops the capture thread and terminates the subprocess."""
        self.stopped = True
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=1.0)
            except Exception:
                try:
                    self.process.kill()
                    self.process.wait()
                except Exception:
                    pass
        if self.thread:
            self.thread.join(timeout=1.0)
        print("[AudioReceiver] Capture stream successfully terminated.")


class VirtualAudioReceiver:
    """
    A high-performance virtual audio receiver that pre-loads the entire audio track
    from an offline video file into memory on startup using a fast, synchronous FFmpeg call.
    Provides sample-accurate, zero-latency real-time retrieval with seamless loop wrap-around.
    Guarantees no missing start seconds in saved video recordings.
    """
    def __init__(self, video_path, sample_rate=48000, channels=1, max_buffer_seconds=10):
        self.video_path = video_path
        self.sample_rate = sample_rate
        self.channels = channels
        
        self.full_audio_track = np.array([], dtype=np.float32)
        self.video_duration = 1.0
        self.start_playback_time = 0.0
        self.system_start_time = 0.0
        self.stopped = False

    def start(self):
        """Synchronously pre-loads the audio track into memory and starts the playback timer."""
        self.stopped = False
        
        cmd = [
            "ffmpeg", "-y",
            "-i", self.video_path,
            "-f", "s16le",
            "-ac", str(self.channels),
            "-ar", str(self.sample_rate),
            "-acodec", "pcm_s16le",
            "pipe:1"
        ]
        
        print(f"[VirtualAudioReceiver] Synchronously pre-loading entire audio track from: '{os.path.basename(self.video_path)}'...")
        try:
            # Decode the entire file audio synchronously in less than 50-100ms
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            raw_bytes, _ = process.communicate()
            
            # Decode int16 PCM bytes to float32
            samples = np.frombuffer(raw_bytes, dtype=np.int16)
            self.full_audio_track = samples.astype(np.float32) / 32768.0
            
            # Fetch video duration from file using OpenCV to ensure sample alignment
            if cv2 is not None:
                cap = cv2.VideoCapture(self.video_path)
                if cap.isOpened():
                    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    self.video_duration = total_frames / fps if (fps and fps > 0) else (len(self.full_audio_track) / self.sample_rate)
                    cap.release()
                else:
                    self.video_duration = len(self.full_audio_track) / self.sample_rate
            else:
                self.video_duration = len(self.full_audio_track) / self.sample_rate
                
            print(f"[VirtualAudioReceiver] Pre-loaded {len(self.full_audio_track)} samples successfully ({self.video_duration:.2f}s duration).")
            self.system_start_time = time.time()
            self.reset_timer()
        except Exception as e:
            print(f"[VirtualAudioReceiver] CRITICAL: Failed to pre-load audio track: {e}", file=sys.stderr)
            self.full_audio_track = np.zeros(self.sample_rate * 10, dtype=np.float32)
            self.video_duration = 10.0
            self.system_start_time = time.time()
            self.reset_timer()
            
        return self

    def reset_timer(self):
        """Resets the playback start timer. Called synchronously when the video loops back to 0."""
        self.start_playback_time = time.time()

    def get_audio_window(self, seconds):
        """
        Returns the exact segment of audio played over the last N seconds.
        Utilizes seamless modulo indexing to wrap around loop points cleanly.
        """
        if len(self.full_audio_track) == 0:
            return np.zeros(int(self.sample_rate * seconds), dtype=np.float32)
            
        # Get real-time elapsed playback duration
        elapsed = time.time() - self.start_playback_time
        
        # Wrap time around video duration
        end_time = elapsed % self.video_duration
        
        total_samples = len(self.full_audio_track)
        end_sample = int(end_time * self.sample_rate)
        duration_samples = int(seconds * self.sample_rate)
        start_sample = end_sample - duration_samples
        
        # Get total elapsed time since the receiver started to avoid zero-padding inside loops
        total_elapsed = time.time() - self.system_start_time
        
        # If in the very first loop segment (since app startup) and start is negative, pad with zeros
        if total_elapsed < seconds:
            pad_len = -start_sample
            if pad_len > 0:
                slice_data = self.full_audio_track[0 : end_sample]
                return np.concatenate((np.zeros(pad_len, dtype=np.float32), slice_data))
            
        # Otherwise, slice the array with modulo wrapping to guarantee seamless continuity
        indices = np.arange(start_sample, end_sample) % total_samples
        return self.full_audio_track[indices]

    def stop(self):
        """Cleans up the virtual receiver state."""
        self.stopped = True
        print("[VirtualAudioReceiver] Virtual audio stream stopped.")
