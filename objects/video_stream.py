#!/usr/bin/env python3
import time
import threading
import sys

try:
    import cv2
except ImportError:
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
        """Returns a thread-safe copy of the most recent frame."""
        with self.lock:
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
        print("[VideoStream] Camera stream stopped and hardware device released.")


class VirtualVideoStream:
    """
    A high-performance virtual video stream that decodes frames from a video file
    at its native frame rate in a background thread, simulating a live camera.
    Automatically loops the video seamlessly when it reaches the end.
    """
    def __init__(self, video_path):
        if cv2 is None:
            raise ImportError("OpenCV (cv2) must be installed to use VirtualVideoStream.")
        
        self.video_path = video_path
        self.cap = None
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = None
        
        self.current_fps = 0.0
        self.video_fps = 30.0
        self.frame_count = 0
        self.start_time = time.time()
        self.audio_receiver = None

    def start(self):
        """Opens the video file and starts the background decoding thread."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open virtual video input file: {self.video_path}")
            
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            self.video_fps = fps
            
        w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"[VirtualVideoStream] Video file initialized: {self.video_path} ({int(w)}x{int(h)} @ {self.video_fps} FPS)")
        
        ret, self.frame = self.cap.read()
        self.stopped = False
        self.start_time = time.time()
        self.frame_count = 0
        
        self.thread = threading.Thread(target=self._update, name="VirtualVideoStreamThread", daemon=True)
        self.thread.start()
        return self

    def _update(self):
        """Decodes frames from the file in background at native frame rate."""
        delay = 1.0 / self.video_fps
        while not self.stopped:
            start_frame_time = time.time()
            ret, frame = self.cap.read()
            
            if not ret:
                # Loop video seamlessly by rewinding pos to 0
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # Reset play timer on the audio receiver to guarantee sample-accurate sync
                if hasattr(self, "audio_receiver") and self.audio_receiver:
                    self.audio_receiver.reset_timer()
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                    
            with self.lock:
                self.frame = frame
                self.frame_count += 1
                
            if self.frame_count % 15 == 0:
                elapsed = time.time() - self.start_time
                self.current_fps = self.frame_count / elapsed
                
            # Regulate frame rate
            elapsed_frame = time.time() - start_frame_time
            sleep_time = max(0.001, delay - elapsed_frame)
            time.sleep(sleep_time)

    def read(self):
        """Returns a thread-safe copy of the most recent video frame."""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def get_fps(self):
        """Returns the calculated virtual stream FPS."""
        return self.current_fps

    def stop(self):
        """Stops the thread and releases the file handle."""
        self.stopped = True
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        print("[VirtualVideoStream] Virtual video stream stopped.")
