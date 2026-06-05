#!/usr/bin/env python3
import os
import sys
import time
from datetime import datetime

try:
    import cv2
except ImportError:
    cv2 = None

from objects import OpenCVVideoStream, AudioReceiver, AudioMatcher, FFplayAudioMonitor, OpenCVHUDRecorder, VirtualVideoStream, VirtualAudioReceiver, StopSignLogic
from onnx_pipeline.pipeline import YOLOv8Detector
import numpy as np
from .device_manager import DeviceManager
import subprocess

class StreamManager:
    """
    Orchestrates the entire high-performance streaming session.
    Initializes hardware and algorithmic objects, manages the rendering frame loop,
    overlays the real-time audio matching progress bar / alert indicators, and handles shutdowns.
    """
    def __init__(self, config):
        self.config = config
        
        # Pull configurations by sub-section
        camera_cfg = config.get("camera", {})
        mic_cfg = config.get("microphone", {})
        rec_cfg = config.get("recording", {})
        match_cfg = config.get("audio_matching", {})
        sim_cfg = config.get("simulation", {})
        model_cfg = config.get("model", {})
        safety_cfg = config.get("safety_logic", {})

        # Configuration parameters
        self.video_device = camera_cfg.get("video_device")
        self.alsa_device = mic_cfg.get("alsa_device")
        self.video_size = camera_cfg.get("video_size", "1280x720")
        self.show_window = camera_cfg.get("show_window", True)
        
        try:
            w_str, h_str = self.video_size.split("x")
            self.width, self.height = int(w_str), int(h_str)
        except Exception:
            self.width, self.height = 1280, 720
            
        self.fps = int(camera_cfg.get("video_fps", 15))
        self.sample_rate = int(mic_cfg.get("sampling_rate", 48000))
        self.channels = int(mic_cfg.get("channels", 1))
        
        self.record_interval = int(rec_cfg.get("record_interval", 10))
        self.recordings_dir = rec_cfg.get("recordings_dir", "./recordings")
        self.recording_enabled = rec_cfg.get("recording_enabled", True)
        
        self.matching_enabled = match_cfg.get("audio_matching_enabled", True)
        self.matching_threshold = float(match_cfg.get("audio_matching_threshold", 0.50))
        self.matching_target_path = match_cfg.get("audio_matching_target_path", "/home/wot-rishabh/Downloads/recording.wav")
        
        # Video file simulation parameters
        self.run_on_video = sim_cfg.get("run_on_video", False)
        self.video_input_path = sim_cfg.get("video_input_path", "")
        self.audio_monitor_process = None
        
        # Application entities
        self.device_manager = DeviceManager()
        self.video_stream = None
        self.audio_monitor = None
        self.hud_recorder = None
        self.audio_receiver = None
        self.audio_matcher = None
        
        # Detection & Stop Sign Logic parameters
        self.detection_enabled = model_cfg.get("detection_enabled", True)
        self.vehicle_stationary_logic_enabled = safety_cfg.get("vehicle_stationary_logic_enabled", True)
        self.overlay_hud_on_frame = rec_cfg.get("overlay_hud_on_frame", True)
        self.recording_cleanup_enabled = rec_cfg.get("recording_cleanup_enabled", True)
        self.max_recordings_size_mb = float(rec_cfg.get("max_recordings_size_mb", 1000.0))
        self.detector = None
        self.stop_sign_logic = None

        self.window_name = "Camera & Voice - High Performance Streamer"
        self.screenshots_dir = "./screenshots"
        
        # Performance metrics
        self.current_full_fps = 0.0
        self.current_det_fps = 0.0
        self.full_frame_count = 0
        self.last_full_fps_time = time.time()
        
    def start_session(self):
        """Prepares resources, clears hardware locks or starts video file demuxers."""
        # Initialize Object Detection and Stop Sign Logic if enabled
        if self.detection_enabled:
            try:
                print("\n[Detection] Initializing YOLOv8 ONNX Detector...")
                self.detector = YOLOv8Detector(self.config)
                print("[Detection] Detector initialized successfully!")
                
                # Initialize stop sign logic
                safety_cfg = self.config.get("safety_logic", {})
                stop_class_name = safety_cfg.get("stop_class_name", "stop")
                required_stop_time = float(safety_cfg.get("required_stop_time", 1.0))
                self.stop_sign_logic = StopSignLogic(
                    stop_class_name=stop_class_name,
                    required_stop_time=required_stop_time
                )
            except Exception as e:
                print(f"[Detection] ERROR starting YOLOv8 Detector: {e}", file=sys.stderr)
                self.detection_enabled = False

        # Check if running in Video File Simulation mode
        if self.run_on_video:
            if not self.video_input_path or not os.path.exists(self.video_input_path):
                print(f"[StreamManager] WARNING: Video input path '{self.video_input_path}' not found! Falling back to physical camera mode.", file=sys.stderr)
                self.run_on_video = False
                
        if self.run_on_video:
            print("\n" + "="*70)
            print("Starting File Simulation & Voice Alignment Session...")
            print(f"  Input Video File: '{self.video_input_path}'")
            print("="*70 + "\n")
            
            # 1. Start audio monitor in background (ffplay video file audio track in real time)
            try:
                cmd = ["ffplay", "-nodisp", "-autoexit", "-i", self.video_input_path]
                print(f"[FFplayAudio] Playing virtual audio track from video in real time...")
                self.audio_monitor_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                print(f"[Error] Failed to initialize virtual audio playback: {e}", file=sys.stderr)
                self.audio_monitor_process = None

            # 2. Start virtual video stream (decodes file frames in real time with looping)
            try:
                self.video_stream = VirtualVideoStream(video_path=self.video_input_path).start()
            except Exception as e:
                print(f"[Critical Error] Failed to initialize Virtual Video Stream: {e}", file=sys.stderr)
                self.stop_session()
                return False

            # 3. Start Virtual Audio Matcher & Receiver (FFmpeg demuxed pipe)
            if self.matching_enabled:
                try:
                    self.audio_matcher = AudioMatcher(
                        target_wav_path=self.matching_target_path,
                        threshold=self.matching_threshold,
                        sample_rate=self.sample_rate,
                        enabled=True
                    )
                    
                    if self.audio_matcher.target_loaded:
                        buffer_sec = max(5.0, self.audio_matcher.target_duration + 2.0)
                        self.audio_receiver = VirtualAudioReceiver(
                            video_path=self.video_input_path,
                            sample_rate=self.sample_rate,
                            channels=self.channels,
                            max_buffer_seconds=buffer_sec
                        ).start()
                        print(f"[AudioMatcher] Successfully loaded matching model target WAV!")
                    else:
                        print("[AudioMatcher] WAV load error. Matching will be bypassed.")
                        self.matching_enabled = False
                except Exception as e:
                    print(f"[AudioMatcher] ERROR starting matching system: {e}", file=sys.stderr)
                    self.matching_enabled = False

            # Link the virtual video stream to the virtual audio receiver for sample-accurate loop sync
            if self.video_stream and self.audio_receiver:
                self.video_stream.audio_receiver = self.audio_receiver

            # 4. Start background HUD chunk recorder if enabled (passing virtual receiver)
            if self.recording_enabled and self.record_interval > 0:
                try:
                    self.hud_recorder = OpenCVHUDRecorder(
                        alsa_device=self.alsa_device,
                        sample_rate=self.sample_rate,
                        channels=self.channels,
                        width=self.width,
                        height=self.height,
                        fps=self.fps,
                        interval=self.record_interval,
                        output_dir=self.recordings_dir,
                        audio_receiver=self.audio_receiver,
                        cleanup_enabled=self.recording_cleanup_enabled,
                        max_size_mb=self.max_recordings_size_mb
                    )
                    self.hud_recorder.start()
                except Exception as e:
                    print(f"[Error] Failed to initialize video/audio recorder: {e}", file=sys.stderr)
                    self.hud_recorder = None
            else:
                self.hud_recorder = None

            os.makedirs(self.screenshots_dir, exist_ok=True)
            return True

        # --- Standard Hardware Capture Mode ---
        # 1. Clean hardware locks held by other zombie subprocesses
        self.device_manager.release_device_locks(self.video_device, self.alsa_device)
        
        print("\n" + "="*70)
        print("Starting Interactive Streaming & Voice Alignment Session...")
        print("="*70 + "\n")
        
        # 2. Start audio monitor in background (ffplay playback)
        try:
            self.audio_monitor = FFplayAudioMonitor(
                alsa_device=self.alsa_device,
                sample_rate=self.sample_rate,
                channels=self.channels
            )
            self.audio_monitor.start()
        except Exception as e:
            print(f"[Error] Failed to initialize live audio monitoring: {e}", file=sys.stderr)
            print("[Info] Continuing without live monitor playbacks...")

        # 3. Start high-performance video stream (OpenCV background capture)
        try:
            self.video_stream = OpenCVVideoStream(
                device_path=self.video_device,
                width=self.width,
                height=self.height,
                fps=self.fps
            ).start()
        except Exception as e:
            print(f"[Critical Error] Failed to initialize OpenCV Video Stream: {e}", file=sys.stderr)
            self.stop_session()
            return False

        # 4. Start Real-time Audio Matcher & Receiver
        if self.matching_enabled:
            try:
                self.audio_matcher = AudioMatcher(
                    target_wav_path=self.matching_target_path,
                    threshold=self.matching_threshold,
                    sample_rate=self.sample_rate,
                    enabled=True
                )
                
                if self.audio_matcher.target_loaded:
                    buffer_sec = max(5.0, self.audio_matcher.target_duration + 2.0)
                    self.audio_receiver = AudioReceiver(
                        alsa_device=self.alsa_device,
                        sample_rate=self.sample_rate,
                        channels=self.channels,
                        max_buffer_seconds=buffer_sec
                    ).start()
                    print(f"[AudioMatcher] Successfully loaded matching model target WAV!")
                else:
                    print("[AudioMatcher] WAV load error. Matching will be bypassed.")
                    self.matching_enabled = False
            except Exception as e:
                print(f"[AudioMatcher] ERROR starting matching system: {e}", file=sys.stderr)
                self.matching_enabled = False

        # 5. Start background HUD chunk recorder (passing physical receiver)
        if self.recording_enabled and self.record_interval > 0:
            try:
                self.hud_recorder = OpenCVHUDRecorder(
                    alsa_device=self.alsa_device,
                    sample_rate=self.sample_rate,
                    channels=self.channels,
                    width=self.width,
                    height=self.height,
                    fps=self.fps,
                    interval=self.record_interval,
                    output_dir=self.recordings_dir,
                    audio_receiver=self.audio_receiver,
                    cleanup_enabled=self.recording_cleanup_enabled,
                    max_size_mb=self.max_recordings_size_mb
                )
                self.hud_recorder.start()
            except Exception as e:
                print(f"[Error] Failed to initialize video/audio recorder: {e}", file=sys.stderr)
                self.hud_recorder = None

        os.makedirs(self.screenshots_dir, exist_ok=True)
        return True

    def _draw_all_hud_overlays(
        self,
        frame,
        detections,
        logic_status,
        timestamp,
        is_stationary,
        flow_mag,
        current_score,
        last_match_time,
        match_cooldown_sec,
        roi_x1,
        roi_y,
        roi_x2,
        w,
        h,
        show_controls=True,
        full_fps=None,
        detection_fps=None
    ):
        """Draws all HUD overlays onto the frame."""

        fps_val = full_fps if full_fps is not None else self.video_stream.get_fps()

        # 1. Run YOLO detections overlay (Safety HUD, boxes)
        if self.detection_enabled and self.detector:
            frame = self.detector._draw_detections_cv2(
                frame=frame,
                detections=detections,
                fps=fps_val,
                logic_status=logic_status,
                timestamp=timestamp,
                is_stationary=is_stationary if self.vehicle_stationary_logic_enabled else None,
                required_stop_time=self.config.get("safety_logic", {}).get("required_stop_time", 1.0),
                flow_mag=flow_mag if self.vehicle_stationary_logic_enabled else 0.0,
                roi_x1=roi_x1,
                roi_y=roi_y,
                roi_x2=roi_x2,
                height=h,
                y_offset=60,
                detection_fps=detection_fps
            )

        # 2. Draw Semi-transparent Header Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Pulse status badge
        cv2.circle(frame, (20, 28), 6, (0, 255, 0), -1)
        
        fps_val = full_fps if full_fps is not None else self.video_stream.get_fps()
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cv2.putText(frame, "LIVE REC", (35, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"Cam: {os.path.basename(self.video_device)} ({w}x{h})", (140, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"Mic: {self.alsa_device}", (450, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, time_str, (w - 290, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Render FPS count
        cv2.putText(frame, f"FPS: {fps_val:.1f}", (w - 110, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2)
        
        # 3. Draw Audio Matcher HUD Panel Widget
        center_x = w // 2
        box_x1 = center_x - 160
        box_y1 = h - 85
        box_x2 = center_x + 160
        box_y2 = h - 45
        
        # Matcher background
        overlay_box = frame.copy()
        cv2.rectangle(overlay_box, (box_x1, box_y1), (box_x2, box_y2), (10, 10, 10), -1)
        cv2.addWeighted(overlay_box, 0.6, frame, 0.4, 0, frame)
        
        match_active = (time.time() - last_match_time) < match_cooldown_sec and current_score >= self.matching_threshold
        
        if not self.matching_enabled:
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (50, 50, 50), 1)
            cv2.putText(frame, "Audio Matcher: DISABLED", (center_x - 85, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1)
        elif not self.audio_matcher.target_loaded:
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (50, 50, 150), 1)
            cv2.putText(frame, "Matcher: Target Load Error", (center_x - 90, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 200), 1)
        else:
            if match_active:
                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 4) # screen glow
                cv2.putText(frame, f"MATCH DETECTED! Correlation: {current_score:.2f}", (center_x - 145, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                
                # Full green progress bar
                cv2.rectangle(frame, (center_x - 140, h - 52), (center_x + 140, h - 48), (20, 80, 20), -1)
                cv2.rectangle(frame, (center_x - 140, h - 52), (center_x + 140, h - 48), (0, 255, 0), -1)
            else:
                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (150, 150, 150), 1)
                cv2.putText(frame, f"Listening... Correlation: {current_score:.2f} / {self.matching_threshold}", (center_x - 130, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
                
                # Cyan bar matching correlation level
                bar_w = int(280 * min(1.0, current_score / self.matching_threshold))
                cv2.rectangle(frame, (center_x - 140, h - 52), (center_x + 140, h - 48), (40, 40, 40), -1)
                if bar_w > 0:
                    cv2.rectangle(frame, (center_x - 140, h - 52), (center_x - 140 + bar_w, h - 48), (255, 180, 50), -1)
        
        # 4. Draw Footer Instructions
        if show_controls:
            cv2.rectangle(frame, (0, h - 30), (w, h), (30, 30, 30), -1)
            cv2.putText(frame, "[q]: Quit cleanly  |  [s]: Take Screenshot", (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
        return frame

    def run_stream_loop(self):
        """Executes the active interactive OpenCV screen rendering loop."""
        # If show_window is disabled in config, bypass interactive loop and run headless
        if not self.show_window:
            frame = self.video_stream.read()
            while frame is None:
                time.sleep(0.01)
                frame = self.video_stream.read()
            h, w, _ = frame.shape
            self._run_headless_logging_fallback(frame, w, h)
            return

        print("\n--- Interactive Controls ---")
        print("  'q' : Quit stream cleanly")
        print("  's' : Save live screenshot")
        if self.matching_enabled:
            print(f"  * Audio Matcher Active against '{os.path.basename(self.matching_target_path)}' *")
        if self.record_interval > 0:
            print(f"  * MP4 Chunk Recording Active (Interval: {self.record_interval}s) *")
        if self.detection_enabled:
            model_cfg = self.config.get("model", {})
            print(f"  * YOLOv8 Object Detection Active (Model: {os.path.basename(model_cfg.get('model_path', ''))}) *")
            print(f"  * Stationary Logic: {'ENABLED' if self.vehicle_stationary_logic_enabled else 'DISABLED'} *")
        print("----------------------------\n")
        
        frame_idx = 0
        current_score = 0.0
        is_matched = False
        last_match_time = 0.0
        match_cooldown_sec = 2.0

        # Optical Flow states
        prev_gray = None
        flow_mag = 0.0
        safety_cfg = self.config.get("safety_logic", {})
        flow_skip = int(safety_cfg.get("flow_skip", 3))
        flow_threshold = float(safety_cfg.get("flow_threshold", 0.5))
        
        roi_calculated = False
        roi_y = 0
        roi_x1 = 0
        roi_x2 = 0
        
        loop_start_time = time.time()
        self.last_full_fps_time = time.time()
        self.full_frame_count = 0
        
        try:
            while True:
                frame = self.video_stream.read()
                if frame is None:
                    time.sleep(0.01)
                    continue
                    
                h, w, _ = frame.shape
                if not roi_calculated:
                    roi_y = int(h * 0.6)
                    roi_x1 = int(w * 0.2)
                    roi_x2 = int(w * 0.8)
                    roi_calculated = True

                frame_idx += 1

                # Update full loop FPS
                self.full_frame_count += 1
                if self.full_frame_count % 15 == 0:
                    now_time = time.time()
                    elapsed_full = now_time - self.last_full_fps_time
                    if elapsed_full > 0:
                        self.current_full_fps = 15.0 / elapsed_full
                    self.last_full_fps_time = now_time
                
                if self.run_on_video:
                    timestamp = frame_idx / self.fps
                else:
                    timestamp = time.time() - loop_start_time

                # --- 1a. Run Object Detection and Stop Sign compliance logic ---
                detections = []
                logic_status = None
                is_stationary = True
                
                if self.detection_enabled and self.detector and self.stop_sign_logic:
                    # Convert to RGB for detector
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    t_det_start = time.time()
                    detections = self.detector.predict(rgb_frame)
                    t_det_end = time.time()
                    
                    det_elapsed = t_det_end - t_det_start
                    if not hasattr(self, "_det_latencies"):
                        self._det_latencies = []
                    self._det_latencies.append(det_elapsed)
                    if len(self._det_latencies) > 15:
                        self._det_latencies.pop(0)
                    avg_det_latency = sum(self._det_latencies) / len(self._det_latencies)
                    self.current_det_fps = 1.0 / avg_det_latency if avg_det_latency > 0 else 0.0
                    
                    if self.vehicle_stationary_logic_enabled:
                        if frame_idx % flow_skip == 0 or frame_idx == 1:
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            if prev_gray is not None:
                                h_small, w_small = prev_gray[roi_y:, roi_x1:roi_x2].shape[:2]
                                prev_small = cv2.resize(prev_gray[roi_y:, roi_x1:roi_x2], (w_small // 4, h_small // 4))
                                gray_small = cv2.resize(gray[roi_y:, roi_x1:roi_x2], (w_small // 4, h_small // 4))
                                
                                flow = cv2.calcOpticalFlowFarneback(
                                    prev_small, gray_small,
                                    None, 0.5, 2, 8, 2, 5, 1.0, 0
                                )
                                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                                flow_mag = float(np.mean(mag))
                            prev_gray = gray
                        
                        is_stationary = flow_mag < flow_threshold
                    else:
                        is_stationary = True
                    
                    logic_status = self.stop_sign_logic.update(timestamp, detections, is_stationary)

                    # Log on changes or alerts
                    if logic_status:
                        if logic_status["alert_fired"] and not getattr(self, "_last_alert", False):
                            print(f"\n>>> [VIOLATION ALERT] Stop Sign violation alert fired at time {timestamp:.2f}s! <<<")
                            self._last_alert = True
                        if not logic_status["alert_fired"]:
                            self._last_alert = False
                            
                        if logic_status["vehicle_stopped"] and not getattr(self, "_last_stopped", False):
                            print(f"\n>>> [COMPLIANCE VERIFIED] Vehicle successfully stopped at sign (time: {timestamp:.2f}s) <<<")
                            self._last_stopped = True
                        if not logic_status["vehicle_stopped"]:
                            self._last_stopped = False

                # --- 1. Compute Audio Matching in real time (every 3 frames to save CPU) ---
                if self.matching_enabled and self.audio_receiver and self.audio_matcher and self.audio_matcher.target_loaded:
                    if frame_idx % 3 == 0:
                        query_sec = self.audio_matcher.target_duration + 1.0
                        live_audio_window = self.audio_receiver.get_audio_window(query_sec)
                        current_score, is_matched = self.audio_matcher.match_live_audio(live_audio_window)
                        
                        if is_matched:
                            last_match_time = time.time()
                            print(f"\r[MATCH DETECTED] Correlation: {current_score:.2f} at {datetime.now().strftime('%H:%M:%S')}")
                
                # Keep a copy of the clean frame for recording before drawing overlays on it
                raw_frame = frame.copy() if not self.overlay_hud_on_frame else None

                # --- 2. Draw HUD/overlays on frame for live window display ---
                frame = self._draw_all_hud_overlays(
                    frame=frame,
                    detections=detections,
                    logic_status=logic_status,
                    timestamp=timestamp,
                    is_stationary=is_stationary,
                    flow_mag=flow_mag,
                    current_score=current_score,
                    last_match_time=last_match_time,
                    match_cooldown_sec=match_cooldown_sec,
                    roi_x1=roi_x1,
                    roi_y=roi_y,
                    roi_x2=roi_x2,
                    w=w,
                    h=h,
                    show_controls=True,
                    full_fps=self.current_full_fps,
                    detection_fps=self.current_det_fps
                )
                
                # --- 5. Record display frame (with or without HUD depending on overlay_hud_on_frame) ---
                if self.hud_recorder:
                    frame_to_record = frame if self.overlay_hud_on_frame else raw_frame
                    self.hud_recorder.write_frame(frame_to_record)
                    self.hud_recorder.tick(frame_to_record)
                    
                # --- 6. Show interactive window or fallback to Headless ---
                try:
                    cv2.imshow(self.window_name, frame)
                except Exception as e:
                    self._run_headless_logging_fallback(frame, w, h)
                    break
                
                # --- 7. Poll keyboard inputs ---
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    shot_name = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    shot_path = os.path.join(self.screenshots_dir, shot_name)
                    cv2.imwrite(shot_path, frame)
                    print(f"[Screenshot] Frame successfully saved to: {shot_path}")
                    
        except KeyboardInterrupt:
            print("\n[Signal] User terminate interrupt. Shutting down streaming shell...")
        finally:
            self.stop_session()

    def _run_headless_logging_fallback(self, first_frame, w, h):
        """Headless capture loop fallback if display environment is not present or window is disabled."""
        if not self.show_window:
            print("\n[Info] Running in Headless mode (show_window=false). Realtime capture active.")
        else:
            print("\n[Warning] Cannot open display window.")
            print("[Info] Running in Headless Logging mode. Realtime capture active.")
        print("[Info] Press Ctrl+C to terminate.")
        
        current_score = 0.0
        is_matched = False
        last_match_time = 0.0
        frame = first_frame
        
        delay = 1.0 / self.fps
        last_log_time = 0.0

        # Optical Flow states
        prev_gray = None
        flow_mag = 0.0
        safety_cfg = self.config.get("safety_logic", {})
        flow_skip = int(safety_cfg.get("flow_skip", 3))
        flow_threshold = float(safety_cfg.get("flow_threshold", 0.5))
        
        roi_y = int(h * 0.6)
        roi_x1 = int(w * 0.2)
        roi_x2 = int(w * 0.8)
        
        loop_start_time = time.time()
        self.last_full_fps_time = time.time()
        self.full_frame_count = 0
        
        try:
            frame_idx = 0
            while True:
                start_time = time.time()
                
                # Fetch latest frame from the video stream
                new_frame = self.video_stream.read()
                if new_frame is not None:
                    frame = new_frame
                
                frame_idx += 1

                # Update full loop FPS
                self.full_frame_count += 1
                if self.full_frame_count % 15 == 0:
                    now_time = time.time()
                    elapsed_full = now_time - self.last_full_fps_time
                    if elapsed_full > 0:
                        self.current_full_fps = 15.0 / elapsed_full
                    self.last_full_fps_time = now_time

                if self.run_on_video:
                    timestamp = frame_idx / self.fps
                else:
                    timestamp = time.time() - loop_start_time

                # --- 1a. Run Object Detection and Stop Sign compliance logic ---
                detections = []
                logic_status = None
                is_stationary = True
                
                if self.detection_enabled and self.detector and self.stop_sign_logic:
                    # Convert to RGB for detector
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    t_det_start = time.time()
                    detections = self.detector.predict(rgb_frame)
                    t_det_end = time.time()
                    
                    det_elapsed = t_det_end - t_det_start
                    if not hasattr(self, "_det_latencies"):
                        self._det_latencies = []
                    self._det_latencies.append(det_elapsed)
                    if len(self._det_latencies) > 15:
                        self._det_latencies.pop(0)
                    avg_det_latency = sum(self._det_latencies) / len(self._det_latencies)
                    self.current_det_fps = 1.0 / avg_det_latency if avg_det_latency > 0 else 0.0
                    
                    if self.vehicle_stationary_logic_enabled:
                        if frame_idx % flow_skip == 0 or frame_idx == 1:
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            if prev_gray is not None:
                                h_small, w_small = prev_gray[roi_y:, roi_x1:roi_x2].shape[:2]
                                prev_small = cv2.resize(prev_gray[roi_y:, roi_x1:roi_x2], (w_small // 4, h_small // 4))
                                gray_small = cv2.resize(gray[roi_y:, roi_x1:roi_x2], (w_small // 4, h_small // 4))
                                
                                flow = cv2.calcOpticalFlowFarneback(
                                    prev_small, gray_small,
                                    None, 0.5, 2, 8, 2, 5, 1.0, 0
                                )
                                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                                flow_mag = float(np.mean(mag))
                            prev_gray = gray
                        
                        is_stationary = flow_mag < flow_threshold
                    else:
                        is_stationary = True
                    
                    logic_status = self.stop_sign_logic.update(timestamp, detections, is_stationary)

                    # Log on changes or alerts
                    if logic_status:
                        if logic_status["alert_fired"] and not getattr(self, "_last_alert", False):
                            print(f"\n>>> [VIOLATION ALERT] Stop Sign violation alert fired at time {timestamp:.2f}s! <<<")
                            self._last_alert = True
                        if not logic_status["alert_fired"]:
                            self._last_alert = False
                            
                        if logic_status["vehicle_stopped"] and not getattr(self, "_last_stopped", False):
                            print(f"\n>>> [COMPLIANCE VERIFIED] Vehicle successfully stopped at sign (time: {timestamp:.2f}s) <<<")
                            self._last_stopped = True
                        if not logic_status["vehicle_stopped"]:
                            self._last_stopped = False

                if self.matching_enabled and self.audio_receiver and self.audio_matcher:
                    # Match every 3 frames to save CPU, same as in interactive mode
                    if frame_idx % 3 == 0:
                        query_sec = self.audio_matcher.target_duration + 1.0
                        live_audio_window = self.audio_receiver.get_audio_window(query_sec)
                        current_score, is_matched = self.audio_matcher.match_live_audio(live_audio_window)
                        if is_matched:
                            last_match_time = time.time()
                            print(f"\n>>> [MATCH DETECTED] Correlation: {current_score:.2f} <<<")

                # --- 2. Draw HUD/overlays on frame if layout flag is active ---
                frame = self._draw_all_hud_overlays(
                    frame=frame,
                    detections=detections,
                    logic_status=logic_status,
                    timestamp=timestamp,
                    is_stationary=is_stationary,
                    flow_mag=flow_mag,
                    current_score=current_score,
                    last_match_time=last_match_time,
                    match_cooldown_sec=2.0,
                    roi_x1=roi_x1,
                    roi_y=roi_y,
                    roi_x2=roi_x2,
                    w=w,
                    h=h,
                    show_controls=False,
                    full_fps=self.current_full_fps,
                    detection_fps=self.current_det_fps
                )

                if self.hud_recorder:
                    self.hud_recorder.write_frame(frame)
                    self.hud_recorder.tick(frame)
                
                # Log once per second to prevent stdout spam
                now = time.time()
                if now - last_log_time >= 1.0:
                    fps_val = self.current_full_fps
                    det_fps_val = self.current_det_fps
                    detection_text = f" | Det FPS: {det_fps_val:.1f} | Detections: {len(detections)}" if self.detection_enabled else ""
                    print(f"\rStreaming live :: Full FPS: {fps_val:.1f}{detection_text} | Match Correlation: {current_score:.2f} | Audio capture active.", end="", flush=True)
                    last_log_time = now
                
                # Regulate frame rate
                elapsed = time.time() - start_time
                sleep_time = max(0.001, delay - elapsed)
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\n[Headless] Shutting down gracefully...")

    def stop_session(self):
        """Cleanly tears down background processes, monitors, streams, and releases locks."""
        print("[Shutdown] Stopping active managers, objects, and hardware handles...")
        if self.audio_receiver:
            self.audio_receiver.stop()
        if self.hud_recorder:
            self.hud_recorder.stop()
        if self.video_stream:
            self.video_stream.stop()
        if self.audio_monitor:
            self.audio_monitor.stop()
        if hasattr(self, "audio_monitor_process") and self.audio_monitor_process:
            print("[Shutdown] Stopping virtual audio playback...")
            try:
                self.audio_monitor_process.terminate()
                self.audio_monitor_process.wait(timeout=1.0)
            except Exception:
                try:
                    self.audio_monitor_process.kill()
                    self.audio_monitor_process.wait()
                except Exception:
                    pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("[Shutdown] Session closed cleanly.")
