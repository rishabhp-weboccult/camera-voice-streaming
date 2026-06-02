#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import threading
from collections import deque
import numpy as np

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
        
        # Try arecord first, then fallback to ffmpeg if arecord is missing or fails
        capture_device = self._get_sharing_capture_device()
        
        # We will try arecord command
        arecord_cmd = [
            "arecord",
            "-D", capture_device,
            "-c", str(self.channels),
            "-r", str(self.sample_rate),
            "-f", "S16_LE",
            "-t", "raw",
            "-"
        ]
        
        # Fallback ffmpeg command
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
        
        # Try launching arecord
        print(f"[AudioReceiver] Initializing capture from {capture_device} ({self.sample_rate}Hz)...")
        try:
            self.process = subprocess.Popen(
                arecord_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=4096
            )
            # Short test read
            time.sleep(0.2)
            if self.process.poll() is not None:
                # arecord failed or exited early, let's try ffmpeg
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
                raise RuntimeError("Could not initialize live audio capture backend (arecord and ffmpeg failed).")

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
                    
                # Convert raw bytes to int16 numpy array
                samples = np.frombuffer(raw_data, dtype=np.int16)
                
                # Convert to float32 normalized to [-1.0, 1.0]
                samples_float = samples.astype(np.float32) / 32768.0
                
                # Write to rolling buffer
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
                
            # If we don't have enough samples, pad with zeros
            if buffer_len < samples_needed:
                padding = np.zeros(samples_needed - buffer_len, dtype=np.float32)
                data = np.array(self.audio_buffer, dtype=np.float32)
                return np.concatenate((padding, data))
            else:
                # Return the slice from the end of the deque
                # Converting the whole deque is simple, but we can slice it
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
