#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import threading
from datetime import datetime
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

def get_sharing_capture_device(alsa_device):
    """
    Converts a direct ALSA hardware reference (like hw:1,0) to a sharing-safe
    dsnoop device (like plug:dsnoop:1) to allow multiple concurrent capture handles.
    """
    if alsa_device.startswith("hw:"):
        parts = alsa_device[3:].split(",")
        card_idx = parts[0]
        return f"plug:dsnoop:{card_idx}"
    return alsa_device

class OpenCVHUDRecorder:
    """
    Highly optimized video and audio recorder that writes frames shown on screen
    (including all overlays, watermarks, and matching indicators) directly to MP4 chunks.
    Automatically muxes with either the live ALSA microphone capture or the real-time
    virtual audio track demuxed from the video file in a background thread to prevent stutters.
    """
    def __init__(self, alsa_device, sample_rate=48000, channels=1, 
                 width=1280, height=720, fps=15, interval=10, output_dir="./recordings",
                 audio_receiver=None, cleanup_enabled=True, max_size_mb=1000.0):
        self.alsa_device = alsa_device
        self.sample_rate = sample_rate
        self.channels = channels
        self.width = width
        self.height = height
        self.fps = fps
        self.interval = interval
        self.output_dir = os.path.abspath(output_dir)
        self.audio_receiver = audio_receiver
        self.cleanup_enabled = cleanup_enabled
        self.max_size_mb = max_size_mb
        
        # Check if we should extract virtual audio track from the VirtualAudioReceiver
        # Using string name checking prevents circular import dependencies
        self.use_virtual_audio = (audio_receiver is not None and 
                                  audio_receiver.__class__.__name__ == "VirtualAudioReceiver")
        
        self.video_writer = None
        self.audio_process = None
        self.chunk_start_time = 0
        self.current_batch_id = None
        
        # Temp paths for raw streams before muxing
        self.temp_dir = os.path.join(self.output_dir, "temp")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Keep track of active background threads
        self.threads = []
        self.written_frames = 0

    def start(self):
        """Starts recording the first chunk."""
        if self.interval <= 0:
            return
            
        self.chunk_start_time = time.time()
        self.written_frames = 0
        self.current_batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Setup Video Writer
        temp_video_name = f"temp_video_{self.current_batch_id}.mp4"
        self.temp_video_path = os.path.join(self.temp_dir, temp_video_name)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.temp_video_path, fourcc, self.fps, (self.width, self.height)
        )
        
        # 2. Setup Audio Channel
        temp_audio_name = f"temp_audio_{self.current_batch_id}.wav"
        self.temp_audio_path = os.path.join(self.temp_dir, temp_audio_name)
        
        if not self.use_virtual_audio:
            # Live hardware capture mode: spawn ffmpeg to record physical ALSA microphone
            capture_device = get_sharing_capture_device(self.alsa_device)
            cmd = [
                "ffmpeg", "-y",
                "-f", "alsa",
                "-channels", str(self.channels),
                "-ar", str(self.sample_rate),
                "-i", capture_device,
                "-t", str(self.interval),
                self.temp_audio_path
            ]
            try:
                self.audio_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except Exception as e:
                print(f"[Recorder] ERROR starting background ALSA audio capture: {e}", file=sys.stderr)
                self.audio_process = None
        else:
            # File simulation mode: we extract real-time audio bytes from the decoder's rolling queue
            self.audio_process = None

    def write_frame(self, frame):
        """
        Writes a visual frame to the active VideoWriter, regulated by real-world timestamps
        to ensure perfect synchronization with the audio stream.
        """
        if self.video_writer and self.video_writer.isOpened():
            now = time.time()
            elapsed = now - self.chunk_start_time
            expected_frames = int(elapsed * self.fps)
            frames_to_write = expected_frames - self.written_frames
            
            if frames_to_write > 0:
                h, w, _ = frame.shape
                if w != self.width or h != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                
                for _ in range(frames_to_write):
                    self.video_writer.write(frame)
                    self.written_frames += 1

    def tick(self, current_frame):
        """
        Polls the recording duration. Rollover to a new file chunk
        if the configured interval has elapsed.
        """
        if self.interval <= 0:
            return
            
        elapsed = time.time() - self.chunk_start_time
        if elapsed >= self.interval:
            self._rotate_chunk(current_frame)

    def _save_virtual_audio_wav(self, wav_path, duration):
        """Extracts the corresponding audio segment from the VirtualAudioReceiver and writes it to a WAV."""
        if not self.audio_receiver:
            return False
            
        import wave
        
        # Read the exact duration of audio samples decoded from the video in the last interval
        samples = self.audio_receiver.get_audio_window(duration)
        
        # Convert float32 samples in range [-1.0, 1.0] back to 16-bit signed PCM
        pcm_samples = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        
        try:
            with wave.open(wav_path, 'wb') as w:
                w.setnchannels(self.channels)
                w.setsampwidth(2) # 16-bit
                w.setframerate(self.sample_rate)
                w.writeframes(pcm_samples.tobytes())
            return True
        except Exception as e:
            print(f"[Recorder] ERROR saving virtual WAV chunk: {e}", file=sys.stderr)
            return False

    def _rotate_chunk(self, last_frame):
        """Closes current files and opens the next chunk, starting background muxing."""
        completed_batch_id = self.current_batch_id
        completed_video_path = self.temp_video_path
        completed_audio_path = self.temp_audio_path
        completed_audio_proc = self.audio_process
        
        if self.video_writer:
            self.video_writer.release()
            
        if completed_audio_proc and completed_audio_proc.poll() is None:
            completed_audio_proc.terminate()
            completed_audio_proc.wait()
            
        # In file simulation mode, grab and save the corresponding decoded video audio segment
        if self.use_virtual_audio:
            elapsed = time.time() - self.chunk_start_time
            self._save_virtual_audio_wav(completed_audio_path, elapsed)
            
        self.start()
        
        final_mp4_path = os.path.join(self.output_dir, f"rec_{completed_batch_id}.mp4")
        t = threading.Thread(
            target=self._merge_chunk,
            args=(completed_video_path, completed_audio_path, final_mp4_path),
            daemon=True
        )
        t.start()
        self.threads.append(t)

    def cleanup_old_recordings(self):
        """
        Calculates the total size of files directly under the output recordings directory.
        If it exceeds max_size_mb, deletes the oldest files until the size is under the threshold.
        """
        if not self.cleanup_enabled:
            return

        try:
            max_size_bytes = self.max_size_mb * 1024 * 1024
            
            # Gather files directly under output_dir (skip directories like 'temp')
            files = []
            total_size = 0
            for entry in os.scandir(self.output_dir):
                if entry.is_file():
                    stat = entry.stat()
                    files.append({
                        "path": entry.path,
                        "size": stat.st_size,
                        "mtime": stat.st_mtime
                    })
                    total_size += stat.st_size
            
            # If total size exceeds max size limit, delete oldest files first
            if total_size > max_size_bytes:
                # Sort by modification time ascending (oldest first)
                files.sort(key=lambda x: x["mtime"])
                
                print(f"\n[Recorder Cleanup] Current folder size: {total_size / (1024*1024):.2f} MB. Threshold: {self.max_size_mb:.2f} MB.")
                for f in files:
                    if total_size <= max_size_bytes:
                        break
                    try:
                        os.remove(f["path"])
                        total_size -= f["size"]
                        print(f"[Recorder Cleanup] Deleted old recording: {os.path.basename(f['path'])} ({f['size'] / (1024*1024):.2f} MB)")
                    except Exception as ex:
                        print(f"[Recorder Cleanup] ERROR deleting file {f['path']}: {ex}", file=sys.stderr)
        except Exception as e:
            print(f"[Recorder Cleanup] ERROR during folder scan: {e}", file=sys.stderr)

    def _merge_chunk(self, video_path, audio_path, final_path):
        """Synchronizes and muxes the video and audio chunks together using ffmpeg."""
        audio_ok = os.path.exists(audio_path) and os.path.getsize(audio_path) > 0
        
        if not audio_ok:
            print(f"\n[Recorder] WARNING: Audio stream file is missing. Saving video-only chunk...")
            try:
                import shutil
                shutil.copy2(video_path, final_path)
                if os.path.exists(video_path):
                    os.remove(video_path)
                self.cleanup_old_recordings()
            except Exception as e:
                print(f"[Recorder] ERROR during video-only copy fallback: {e}", file=sys.stderr)
            return

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            final_path
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
            if res.returncode == 0:
                print(f"\n[Recorder] Saved recorded chunk: {os.path.basename(final_path)}")
                if os.path.exists(video_path):
                    os.remove(video_path)
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                self.cleanup_old_recordings()
            else:
                print(f"\n[Recorder] WARNING: ffmpeg merge failed (code {res.returncode}). Falling back to video-only...")
                import shutil
                shutil.copy2(video_path, final_path)
                if os.path.exists(video_path):
                    os.remove(video_path)
                self.cleanup_old_recordings()
        except Exception as e:
            print(f"\n[Recorder] ERROR background muxing chunk: {e}", file=sys.stderr)

    def stop(self):
        """Safely stops active recording and flushes final files."""
        if self.interval <= 0:
            return
            
        print("[Recorder] Finalizing active recording chunk...")
        if self.video_writer:
            self.video_writer.release()
            
        if self.audio_process:
            if self.audio_process.poll() is None:
                self.audio_process.terminate()
                self.audio_process.wait()
                
        # In file simulation mode, write final decoded audio track segment to WAV
        if self.use_virtual_audio:
            elapsed = time.time() - self.chunk_start_time
            self._save_virtual_audio_wav(self.temp_audio_path, elapsed)
            
        final_mp4_path = os.path.join(self.output_dir, f"rec_{self.current_batch_id}.mp4")
        self._merge_chunk(self.temp_video_path, self.temp_audio_path, final_mp4_path)
        
        for t in self.threads:
            t.join(timeout=2.0)
            
        try:
            if os.path.exists(self.temp_dir) and not os.listdir(self.temp_dir):
                os.rmdir(self.temp_dir)
        except Exception:
            pass
        print("[Recorder] Recording system finalized successfully.")
