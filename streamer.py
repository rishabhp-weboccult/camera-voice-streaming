#!/usr/bin/env python3
import os
import sys
import time
import threading
import subprocess
from datetime import datetime

# Try to import cv2, otherwise print a helpful instruction
try:
    import cv2
except ImportError:
    print("WARNING: 'opencv-python' (cv2) is not installed. Please install it with 'pip install opencv-python'.", file=sys.stderr)
    cv2 = None

class OpenCVVideoStream:
    """
    A high-performance, thread-safe video streaming object that captures frames
    from a V4L2 device in a background thread to eliminate read latency.
    """
    def __init__(self, device_path, width=1280, height=720, fps=15):
        if cv2 is None:
            raise ImportError("OpenCV (cv2) must be installed to use OpenCVVideoStream.")
        
        self.device_path = device_path
        self.width = width
        self.height = height
        self.fps = fps
        
        self.cap = None
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = None
        
        # Performance metrics
        self.frame_count = 0
        self.start_time = time.time()
        self.current_fps = 0.0

    def start(self):
        """Starts the background thread to capture video frames."""
        # Initialize video capture
        # Try to use V4L2 API explicitly under Linux
        self.cap = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
        
        # Set parameters
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Verify settings
        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"[VideoStream] Camera initialized: {actual_w}x{actual_h} @ {actual_fps} FPS")
        
        # Initial read
        ret, self.frame = self.cap.read()
        if not ret:
            print("[VideoStream] WARNING: Failed to read initial frame from camera.", file=sys.stderr)
            
        self.stopped = False
        self.start_time = time.time()
        self.frame_count = 0
        
        self.thread = threading.Thread(target=self._update, name="VideoStreamThread", daemon=True)
        self.thread.start()
        return self

    def _update(self):
        """Continuously reads frames from the camera."""
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                print("[VideoStream] ERROR: Failed to read frame from camera stream.", file=sys.stderr)
                time.sleep(0.05)
                continue
                
            with self.lock:
                self.frame = frame
                self.frame_count += 1
                
            # Compute FPS every 15 frames
            if self.frame_count % 15 == 0:
                elapsed = time.time() - self.start_time
                self.current_fps = self.frame_count / elapsed

    def read(self):
        """Returns the most recent frame."""
        with self.lock:
            # Return a copy to avoid multithreading race conditions during frame drawing
            return self.frame.copy() if self.frame is not None else None

    def get_fps(self):
        """Returns the calculated real-time capture FPS."""
        return self.current_fps

    def stop(self):
        """Stops the thread and releases the capture hardware."""
        self.stopped = True
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        print("[VideoStream] Stream stopped and device released.")


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


class FFplayAudioMonitor:
    """
    Manages a live audio stream/monitor from an ALSA capture device
    using ffplay in a managed background subprocess.
    """
    def __init__(self, alsa_device, sample_rate=48000, channels=1):
        self.alsa_device = alsa_device
        self.sample_rate = sample_rate
        self.channels = channels
        self.process = None

    def start(self):
        """Launches ffplay subprocess to playback live ALSA capture."""
        # ffplay flags:
        # -nodisp: Do not open any graphic display window
        # -f alsa: Use ALSA capture driver
        # -ar: Sample rate
        # -ac: Audio channels
        # -i: Input device specifier
        # Convert direct card reference to a sharing-safe dsnoop device
        capture_device = get_sharing_capture_device(self.alsa_device)
        cmd = [
            "ffplay", "-nodisp",
            "-f", "alsa",
            "-ar", str(self.sample_rate),
            "-channels", str(self.channels),
            "-i", capture_device
        ]
        
        print(f"[FFplayAudio] Starting live audio loopback from {capture_device}...")
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,  # Capture stderr to debug problems
                text=True
            )
            # Give it a short moment to initialize and check if it crashed early
            time.sleep(0.5)
            if self.process.poll() is not None:
                # Process exited immediately!
                stderr_output = self.process.stderr.read()
                raise RuntimeError(f"ffplay failed to start. Stderr:\n{stderr_output}")
        except FileNotFoundError:
            raise FileNotFoundError("ffplay executable not found. Make sure ffmpeg/ffplay are installed.")

    def stop(self):
        """Gracefully terminates the ffplay subprocess."""
        if self.process and self.process.poll() is None:
            print("[FFplayAudio] Stopping live audio loopback...")
            self.process.terminate()
            try:
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            print("[FFplayAudio] Stopped.")


# class FFplayCombinedStreamer:
#     """
#     Launches ffplay in full audio/video streaming preview mode
#     to demonstrate live loopback of BOTH camera and microphone using ffmpeg/ffplay.
#     """
#     def __init__(self, video_device, alsa_device, width=1280, height=720, fps=15, sample_rate=48000, channels=1):
#         self.video_device = video_device
#         self.alsa_device = alsa_device
#         self.width = width
#         self.height = height
#         self.fps = fps
#         self.sample_rate = sample_rate
#         self.channels = channels
#         self.process = None
# 
#     def start(self):
#         """
#         Launches ffplay capturing from v4l2 and alsa simultaneously.
#         This provides a zero-latency, hardware-optimized synchronized preview.
#         """
#         # We can construct an ffplay command that takes the webcam and ALSA input.
#         # Alternatively, we can use ffmpeg to capture and pipe to ffplay, or just play directly.
#         # Direct play via ffplay is standard:
#         # ffplay -f v4l2 -input_format mjpeg -video_size 1280x720 -framerate 15 -i /dev/video0 -f alsa -i hw:1,0
#         cmd = [
#             "ffplay",
#             "-f", "v4l2",
#             "-input_format", "mjpeg",
#             "-video_size", f"{self.width}x{self.height}",
#             "-framerate", str(self.fps),
#             "-i", self.video_device,
#             "-f", "alsa",
#             "-ar", str(self.sample_rate),
#             "-channels", str(self.channels),
#             "-i", self.alsa_device
#         ]
#         
#         print(f"[FFplayCombined] Launching synchronized hardware display...")
#         print(f"  Video: {self.video_device}")
#         print(f"  Audio: {self.alsa_device}")
#         
#         try:
#             self.process = subprocess.Popen(
#                 cmd,
#                 stdout=subprocess.DEVNULL,
#                 stderr=subprocess.PIPE,
#                 text=True
#             )
#             time.sleep(0.5)
#             if self.process.poll() is not None:
#                 stderr_output = self.process.stderr.read()
#                 raise RuntimeError(f"ffplay combined failed to start. Stderr:\n{stderr_output}")
#         except FileNotFoundError:
#             raise FileNotFoundError("ffplay executable not found. Make sure ffmpeg/ffplay are installed.")
# 
#             print("[FFplayCombined] Stopped.")


class OpenCVHUDRecorder:
    """
    Highly optimized video and audio recorder that writes frames shown on screen
    (including all overlays, watermarks, and FPS display) directly to MP4 chunks.
    Muxes the finished video with ALSA audio in a separate background thread
    to prevent visual frame drops or UI thread stutter.
    """
    def __init__(self, alsa_device, sample_rate=48000, channels=1, 
                 width=1280, height=720, fps=15, interval=10, output_dir="./recordings"):
        self.alsa_device = alsa_device
        self.sample_rate = sample_rate
        self.channels = channels
        self.width = width
        self.height = height
        self.fps = fps
        self.interval = interval
        self.output_dir = os.path.abspath(output_dir)
        
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
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Fast MPEG-4 codec
        self.video_writer = cv2.VideoWriter(
            self.temp_video_path, fourcc, self.fps, (self.width, self.height)
        )
        
        # 2. Setup Background Audio Process using sharing-safe dsnoop
        temp_audio_name = f"temp_audio_{self.current_batch_id}.wav"
        self.temp_audio_path = os.path.join(self.temp_dir, temp_audio_name)
        
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

    def write_frame(self, frame):
        """
        Writes a visual frame to the active VideoWriter, regulated by real-world timestamps
        to ensure perfect synchronization with the audio stream.
        """
        if self.video_writer and self.video_writer.isOpened():
            now = time.time()
            elapsed = now - self.chunk_start_time
            
            # Target total frames that should be written by this elapsed timestamp
            expected_frames = int(elapsed * self.fps)
            
            # Duplication/skip calculations to maintain real-world timing
            frames_to_write = expected_frames - self.written_frames
            
            if frames_to_write > 0:
                h, w, _ = frame.shape
                if w != self.width or h != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                
                # Write frame(s) to match the elapsed real-world time
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

    def _rotate_chunk(self, last_frame):
        """Closes current files and opens the next chunk, starting background muxing."""
        # 1. Capture details of the completed chunk
        completed_batch_id = self.current_batch_id
        completed_video_path = self.temp_video_path
        completed_audio_path = self.temp_audio_path
        completed_audio_proc = self.audio_process
        
        # 2. Clean up current writers and instantly start the next chunk (minimizes frame gap)
        if self.video_writer:
            self.video_writer.release()
            
        # Cleanly stop the audio process if it hasn't ended
        if completed_audio_proc and completed_audio_proc.poll() is None:
            completed_audio_proc.terminate()
            completed_audio_proc.wait()
            
        # Start new chunk immediately!
        self.start()
        
        # 3. Mux completed files in background thread
        final_mp4_path = os.path.join(self.output_dir, f"rec_{completed_batch_id}.mp4")
        t = threading.Thread(
            target=self._merge_chunk,
            args=(completed_video_path, completed_audio_path, final_mp4_path),
            daemon=True
        )
        t.start()
        self.threads.append(t)

    def _merge_chunk(self, video_path, audio_path, final_path):
        """Synchronizes and muxes the video and audio chunks together using ffmpeg."""
        audio_ok = os.path.exists(audio_path) and os.path.getsize(audio_path) > 0
        
        if not audio_ok:
            print(f"\n[Recorder] WARNING: Audio stream file is missing or empty. Saving video-only chunk...")
            try:
                import shutil
                shutil.copy2(video_path, final_path)
                print(f"[Recorder] Saved video-only chunk successfully: {os.path.basename(final_path)}")
                
                # Cleanup video temp file
                if os.path.exists(video_path):
                    os.remove(video_path)
            except Exception as e:
                print(f"[Recorder] ERROR during video-only copy fallback: {e}", file=sys.stderr)
            return

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",        # Fast copy - zero video transcoding!
            "-c:a", "aac",         # Mux standard audio codec
            "-b:a", "192k",
            final_path
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
            if res.returncode == 0:
                print(f"\n[Recorder] Saved recorded chunk: {os.path.basename(final_path)}")
                
                # Safe cleanup of raw streams
                if os.path.exists(video_path):
                    os.remove(video_path)
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            else:
                print(f"\n[Recorder] WARNING: ffmpeg merge failed (code {res.returncode}). Falling back to video-only copy...")
                import shutil
                shutil.copy2(video_path, final_path)
                if os.path.exists(video_path):
                    os.remove(video_path)
        except Exception as e:
            print(f"\n[Recorder] ERROR background muxing chunk: {e}", file=sys.stderr)

    def stop(self):
        """Safely stops active recording and flushes final files."""
        if self.interval <= 0:
            return
            
        print("[Recorder] Finalizing active recording chunk...")
        
        # Release writer
        if self.video_writer:
            self.video_writer.release()
            
        # Kill audio capture
        if self.audio_process:
            if self.audio_process.poll() is None:
                self.audio_process.terminate()
                self.audio_process.wait()
                
        # Merge final chunk synchronously (or spawn final thread and wait)
        final_mp4_path = os.path.join(self.output_dir, f"rec_{self.current_batch_id}.mp4")
        self._merge_chunk(self.temp_video_path, self.temp_audio_path, final_mp4_path)
        
        # Wait briefly for other background threads to complete
        for t in self.threads:
            t.join(timeout=2.0)
            
        # Clear empty temp folders if possible
        try:
            if os.path.exists(self.temp_dir) and not os.listdir(self.temp_dir):
                os.rmdir(self.temp_dir)
        except Exception:
            pass
        print("[Recorder] Recording system finalized successfully.")

