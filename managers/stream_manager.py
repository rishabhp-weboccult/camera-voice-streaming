#!/usr/bin/env python3
import os
import sys
import time
import queue
import threading
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
        self.save_telemetry_csv = rec_cfg.get("save_telemetry_csv", False)
        self.csv_file = None
        self.csv_writer = None
        
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
        self.tracker = None
        self.active_journeys = {}
        self.last_processed_timestamp = 0.0

        self.window_name = "Camera & Voice - High Performance Streamer"
        self.screenshots_dir = "./screenshots"
        
        # Performance metrics
        self.current_full_fps = 0.0
        self.current_det_fps = 0.0
        self.full_frame_count = 0
        self.last_full_fps_time = time.time()
        
        # Additional profiling metrics
        self._flow_latencies = []
        self._match_latencies = []
        self.avg_flow_time = 0.0
        self.avg_match_time = 0.0

        # Threading & Queue structures for pipelined processing
        self.stopped = False
        self.det_in_queue = queue.Queue(maxsize=4)
        self.det_out_queue = queue.Queue(maxsize=4)
        self.flow_queue = queue.Queue(maxsize=1)
        
        self.latest_is_stationary = False
        self.latest_flow_mag = 0.0
        self.flow_lock = threading.Lock()
        
        self.det_thread = None
        self.flow_thread = None
        
    def start_session(self):
        """Prepares resources, clears hardware locks or starts video file demuxers."""
        # Initialize Object Detection and Stop Sign Logic if enabled
        if self.detection_enabled:
            try:
                print("\n[Detection] Initializing YOLOv8 ONNX Detector...")
                self.detector = YOLOv8Detector(self.config)
                print("[Detection] Detector initialized successfully!")
                
                # Initialize BYTETracker
                from byte_tracker.byte_tracker_model import BYTETracker
                tracker_cfg = self.config.get("tracker", {})
                self.tracker = BYTETracker(
                    fps=self.fps,
                    first_track_thresh=float(tracker_cfg.get("first_track_thresh", 0.4)),
                    second_track_thresh=float(tracker_cfg.get("second_track_thresh", 0.1)),
                    match_thresh=float(tracker_cfg.get("match_thresh", 0.7)),
                    track_buffer=int(tracker_cfg.get("track_buffer", 30)),
                    resize_width_height=(self.width, self.height)
                )
                self.active_journeys = {}
                
                # Initialize stop sign logic
                safety_cfg = self.config.get("safety_logic", {})
                stop_class_name = safety_cfg.get("stop_class_name", "stop")
                required_stop_time = float(safety_cfg.get("required_stop_time", 1.0))
                self.stop_sign_logic = StopSignLogic(
                    stop_class_name=stop_class_name,
                    required_stop_time=required_stop_time
                )
                
                # Start worker threads
                self.stopped = False
                self.det_thread = threading.Thread(target=self._detection_worker, name="DetectionWorkerThread", daemon=True)
                self.det_thread.start()
                
                if self.vehicle_stationary_logic_enabled:
                    self.flow_thread = threading.Thread(target=self._optical_flow_worker, name="OpticalFlowWorkerThread", daemon=True)
                    self.flow_thread.start()
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
            self._init_telemetry_csv()
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
        self._init_telemetry_csv()
        return True

    def _init_telemetry_csv(self):
        """Initializes and opens the telemetry CSV file if enabled."""
        if self.save_telemetry_csv:
            import csv
            os.makedirs(self.recordings_dir, exist_ok=True)
            csv_path = os.path.join(self.recordings_dir, "telemetry.csv")
            file_exists = os.path.exists(csv_path)
            try:
                self.csv_file = open(csv_path, mode='a', newline='', encoding='utf-8')
                self.csv_writer = csv.writer(self.csv_file)
                if not file_exists or os.path.getsize(csv_path) == 0:
                    self.csv_writer.writerow([
                        "System_Time", "Video_Time_Sec", "Full_FPS", "Detection_FPS",
                        "Vehicle_Status", "Flow_Magnitude", "Stop_Sign_Status", "Stop_Sign_Duration",
                        "Detections_Count", "Detections_Detail", "Audio_Match_Score",
                        "Prep_Latency_ms", "Inference_Latency_ms", "Postprocess_Latency_ms",
                        "Flow_Latency_ms", "Audio_Match_Latency_ms", "Inference_Provider"
                    ])
                    self.csv_file.flush()
                print(f"[Telemetry] Logging HUD telemetry data to: {csv_path}")
            except Exception as e:
                print(f"[Telemetry] ERROR: Failed to open telemetry CSV: {e}", file=sys.stderr)

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
            from utils import _draw_detections_cv2
            frame = _draw_detections_cv2(
                detector=self.detector,
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
                detection_fps=detection_fps,
                flow_time=self.avg_flow_time,
                match_time=self.audio_matcher.avg_match_time if (self.audio_matcher and self.matching_enabled) else None
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
        
        # 5. Draw Active Journeys and Events Tracker Panel
        x1 = w - 280
        x2 = w - 10
        tracker_y = 70
        panel_h = 55
        
        has_journey = False
        event_conducted = False
        journey_id_str = ""
        
        if hasattr(self, "active_journeys") and self.active_journeys:
            active_list = list(self.active_journeys.items())
            if active_list:
                has_journey = True
                j_id, j_mem = active_list[0]
                journey_id_str = f" #{j_id}"
                if j_mem.get("event_captured"):
                    event_conducted = True

        # Semi-transparent background
        overlay_p = frame.copy()
        cv2.rectangle(overlay_p, (x1, tracker_y), (x2, tracker_y + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay_p, 0.7, frame, 0.3, 0, frame)
        
        # Gray border
        cv2.rectangle(frame, (x1, tracker_y), (x2, tracker_y + panel_h), (100, 100, 100), 1)
        
        # Title / Header
        cv2.putText(frame, "TRACKER STATUS", (x1 + 10, tracker_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (x1 + 10, tracker_y + 24), (x2 - 10, tracker_y + 24), (80, 80, 80), 1)
        
        # Journey indicator dot: GREEN if active, RED if inactive
        j_color = (0, 255, 0) if has_journey else (0, 0, 255)
        j_text = f"Journey Start{journey_id_str}" if has_journey else "No Active Journey"
        cv2.circle(frame, (x1 + 20, tracker_y + 38), 5, j_color, -1)
        cv2.putText(frame, j_text, (x1 + 32, tracker_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Event indicator dot: GREEN if event conducted, RED otherwise
        e_color = (0, 255, 0) if event_conducted else (0, 0, 255)
        e_text = "Event Conducted" if event_conducted else "Event Pending"
        cv2.circle(frame, (x1 + 150, tracker_y + 38), 5, e_color, -1)
        cv2.putText(frame, e_text, (x1 + 162, tracker_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        # 4. Draw Footer Instructions
        if show_controls:
            cv2.rectangle(frame, (0, h - 30), (w, h), (30, 30, 30), -1)
            cv2.putText(frame, "[q]: Quit cleanly  |  [s]: Take Screenshot", (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Write telemetry data to CSV if enabled
        if self.save_telemetry_csv and self.csv_writer:
            try:
                system_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                
                # Format detections detail
                det_details = []
                for det in detections:
                    det_details.append(f"{det.class_name}({det.confidence:.2%})")
                det_details_str = "; ".join(det_details)
                
                # Stop sign details
                status_text = "CLEAR"
                stop_duration = 0.0
                if logic_status:
                    if logic_status["alert_fired"]:
                        status_text = "VIOLATION"
                    elif logic_status["vehicle_stopped"]:
                        status_text = "COMPLIED"
                    elif logic_status["stop_seen"]:
                        status_text = "VISIBLE"
                        stop_duration = logic_status["stop_duration"]
                
                # Latency details
                prep_ms = self.detector.avg_preprocess_time * 1000 if self.detector else 0.0
                inf_ms = self.detector.avg_inference_time * 1000 if self.detector else 0.0
                post_ms = self.detector.avg_postprocess_time * 1000 if self.detector else 0.0
                flow_ms = self.avg_flow_time * 1000
                match_ms = self.audio_matcher.avg_match_time * 1000 if (self.audio_matcher and self.matching_enabled) else 0.0
                
                provider = self.detector.session.get_providers()[0] if self.detector else "N/A"
                motion_state = "STATIONARY" if is_stationary else "MOVING" if is_stationary is not None else "N/A"
                
                self.csv_writer.writerow([
                    system_time,
                    f"{timestamp:.3f}" if timestamp is not None else "0.000",
                    f"{fps_val:.1f}",
                    f"{detection_fps:.1f}" if detection_fps is not None else "0.0",
                    motion_state,
                    f"{flow_mag:.6f}",
                    status_text,
                    f"{stop_duration:.3f}",
                    len(detections),
                    det_details_str,
                    f"{current_score:.4f}",
                    f"{prep_ms:.2f}",
                    f"{inf_ms:.2f}",
                    f"{post_ms:.2f}",
                    f"{flow_ms:.2f}",
                    f"{match_ms:.2f}",
                    provider
                ])
                # Periodically flush
                if self.full_frame_count % 15 == 0:
                    self.csv_file.flush()
            except Exception:
                pass
            
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
                start_time = time.time()
                img_src = self.video_stream.read()
                if img_src is None:
                    time.sleep(0.01)
                    continue
                
                frame_idx += 1
                
                if self.run_on_video:
                    timestamp = frame_idx / self.fps
                else:
                    timestamp = time.time() - loop_start_time
                
                # Push frame to background detection queue (non-blocking)
                if self.detection_enabled:
                    if not self.det_in_queue.full():
                        try:
                            self.det_in_queue.put_nowait((img_src.copy(), timestamp, frame_idx))
                        except queue.Full:
                            pass
                
                # Retrieve the matched frame and its corresponding detections from the pipeline
                new_det_ready = False
                if self.detection_enabled:
                    try:
                        # Blocking read with timeout to keep UI responsive
                        frame, timestamp_det, frame_idx_det, detections = self.det_out_queue.get(timeout=0.1)
                        new_det_ready = True
                    except queue.Empty:
                        # Fallback if queue is empty: reuse source frame with empty detections
                        frame = img_src.copy()
                        timestamp_det = timestamp
                        frame_idx_det = frame_idx
                        detections = []
                else:
                    frame = img_src.copy()
                    timestamp_det = timestamp
                    frame_idx_det = frame_idx
                    detections = []

                self.last_processed_timestamp = timestamp_det

                # Run BYTETracker
                draw_detections = detections
                if self.detection_enabled and new_det_ready and self.tracker:
                    tracker_inputs = []
                    for det in detections:
                        tracker_inputs.append([
                            det.box[0], det.box[1], det.box[2], det.box[3],
                            det.confidence, det.class_name
                        ])
                    try:
                        tracked_stracks = self.tracker.update(tracker_inputs)
                        self._process_journeys_and_events(tracked_stracks, timestamp_det, current_score)
                        
                        # Build draw-only tracked detections list (for visualization)
                        from dataclass.detection import Detection
                        draw_detections = []
                        for track in tracked_stracks:
                            x1, y1, x2, y2 = track.tlbr
                            draw_detections.append(Detection(
                                class_id=0,
                                class_name=f"{track.class_name} #{track.track_id}",
                                confidence=track.score,
                                box=[float(x1), float(y1), float(x2), float(y2)]
                            ))
                    except Exception as e:
                        print(f"[Tracker] Error in tracker update: {e}", file=sys.stderr)
                
                h, w, _ = frame.shape
                if not roi_calculated:
                    roi_y = int(h * 0.6)
                    roi_x1 = int(w * 0.2)
                    roi_x2 = int(w * 0.8)
                    roi_calculated = True

                # Update full loop FPS
                self.full_frame_count += 1
                if self.full_frame_count % 15 == 0:
                    now_time = time.time()
                    elapsed_full = now_time - self.last_full_fps_time
                    if elapsed_full > 0:
                        self.current_full_fps = 15.0 / elapsed_full
                    self.last_full_fps_time = now_time

                # --- 1a. Run Stop Sign compliance logic and optical flow ---
                logic_status = None
                is_stationary = True
                
                if self.detection_enabled and self.stop_sign_logic:
                    if self.vehicle_stationary_logic_enabled:
                        if frame_idx % flow_skip == 0 or frame_idx == 1:
                            gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
                            if prev_gray is not None:
                                h_small, w_small = prev_gray[roi_y:, roi_x1:roi_x2].shape[:2]
                                prev_small = cv2.resize(prev_gray[roi_y:, roi_x1:roi_x2], (w_small // 4, h_small // 4))
                                gray_small = cv2.resize(gray[roi_y:, roi_x1:roi_x2], (w_small // 4, h_small // 4))
                                
                                # Push frames to optical flow queue (non-blocking)
                                if self.flow_queue.empty():
                                    try:
                                        self.flow_queue.put_nowait((prev_small.copy(), gray_small.copy(), flow_threshold))
                                    except queue.Full:
                                        pass
                            prev_gray = gray
                        
                        # Get latest stationary state
                        with self.flow_lock:
                            is_stationary = self.latest_is_stationary
                            flow_mag = self.latest_flow_mag
                    else:
                        is_stationary = True
                    
                    logic_status = self.stop_sign_logic.update(timestamp_det, detections, is_stationary)

                    # Log on changes or alerts
                    if logic_status:
                        if logic_status["alert_fired"] and not getattr(self, "_last_alert", False):
                            print(f"\n>>> [VIOLATION ALERT] Stop Sign violation alert fired at time {timestamp_det:.2f}s! <<<")
                            self._last_alert = True
                        if not logic_status["alert_fired"]:
                            self._last_alert = False
                            
                        if logic_status["vehicle_stopped"] and not getattr(self, "_last_stopped", False):
                            print(f"\n>>> [COMPLIANCE VERIFIED] Vehicle successfully stopped at sign (time: {timestamp_det:.2f}s) <<<")
                            self._last_stopped = True
                        if not logic_status["vehicle_stopped"]:
                            self._last_stopped = False

                # --- 1. Compute Audio Matching in real time (every 3 frames to save CPU) ---
                if self.matching_enabled and self.audio_receiver and self.audio_matcher and self.audio_matcher.target_loaded:
                    if frame_idx_det % 3 == 0:
                        query_sec = self.audio_matcher.target_duration + 1.0
                        live_audio_window = self.audio_receiver.get_audio_window(query_sec)
                        current_score, is_matched = self.audio_matcher.match_live_audio(live_audio_window)
                        
                        if is_matched:
                            last_match_time = time.time()
                            print(f"\r[MATCH DETECTED] Correlation: {current_score:.2f} at {datetime.now().strftime('%H:%M:%S')}")
                
                # Keep a copy of the clean frame for recording before drawing overlays on it
                raw_frame = frame if not self.overlay_hud_on_frame else None

                # --- 2. Draw HUD/overlays on frame for live window display ---
                frame = self._draw_all_hud_overlays(
                    frame=frame,
                    detections=draw_detections,
                    logic_status=logic_status,
                    timestamp=timestamp_det,
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
                delay = 1.0 / self.fps
                elapsed = time.time() - start_time
                delay_ms = max(1, int((delay - elapsed) * 1000))
                key = cv2.waitKey(delay_ms) & 0xFF
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
        img_src = first_frame
        frame = img_src.copy()
        
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
                    img_src = new_frame
                
                frame_idx += 1
                
                if self.run_on_video:
                    timestamp = frame_idx / self.fps
                else:
                    timestamp = time.time() - loop_start_time
                
                # Push frame to background detection queue (non-blocking)
                if self.detection_enabled:
                    if not self.det_in_queue.full():
                        try:
                            self.det_in_queue.put_nowait((img_src.copy(), timestamp, frame_idx))
                        except queue.Full:
                            pass
                
                # Retrieve the matched frame and its corresponding detections from the pipeline
                new_det_ready = False
                if self.detection_enabled:
                    try:
                        # Blocking read with timeout to keep UI responsive
                        frame, timestamp_det, frame_idx_det, detections = self.det_out_queue.get(timeout=0.1)
                        new_det_ready = True
                    except queue.Empty:
                        # Fallback if queue is empty: reuse source frame with empty detections
                        frame = img_src.copy()
                        timestamp_det = timestamp
                        frame_idx_det = frame_idx
                        detections = []
                else:
                    frame = img_src.copy()
                    timestamp_det = timestamp
                    frame_idx_det = frame_idx
                    detections = []

                self.last_processed_timestamp = timestamp_det

                # Run BYTETracker
                draw_detections = detections
                if self.detection_enabled and new_det_ready and self.tracker:
                    tracker_inputs = []
                    for det in detections:
                        tracker_inputs.append([
                            det.box[0], det.box[1], det.box[2], det.box[3],
                            det.confidence, det.class_name
                        ])
                    try:
                        tracked_stracks = self.tracker.update(tracker_inputs)
                        self._process_journeys_and_events(tracked_stracks, timestamp_det, current_score)
                        
                        # Build draw-only tracked detections list (for visualization)
                        from dataclass.detection import Detection
                        draw_detections = []
                        for track in tracked_stracks:
                            x1, y1, x2, y2 = track.tlbr
                            draw_detections.append(Detection(
                                class_id=0,
                                class_name=f"{track.class_name} #{track.track_id}",
                                confidence=track.score,
                                box=[float(x1), float(y1), float(x2), float(y2)]
                            ))
                    except Exception as e:
                        print(f"[Tracker] Error in tracker update: {e}", file=sys.stderr)

                # Update full loop FPS
                self.full_frame_count += 1
                if self.full_frame_count % 15 == 0:
                    now_time = time.time()
                    elapsed_full = now_time - self.last_full_fps_time
                    if elapsed_full > 0:
                        self.current_full_fps = 15.0 / elapsed_full
                    self.last_full_fps_time = now_time

                # --- 1a. Run Stop Sign compliance logic and optical flow ---
                logic_status = None
                is_stationary = True
                
                if self.detection_enabled and self.stop_sign_logic:
                    if self.vehicle_stationary_logic_enabled:
                        if frame_idx % flow_skip == 0 or frame_idx == 1:
                            gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
                            if prev_gray is not None:
                                h_small, w_small = prev_gray[roi_y:, roi_x1:roi_x2].shape[:2]
                                prev_small = cv2.resize(prev_gray[roi_y:, roi_x1:roi_x2], (w_small // 4, h_small // 4))
                                gray_small = cv2.resize(gray[roi_y:, roi_x1:roi_x2], (w_small // 4, h_small // 4))
                                
                                # Push frames to optical flow queue (non-blocking)
                                if self.flow_queue.empty():
                                    try:
                                        self.flow_queue.put_nowait((prev_small.copy(), gray_small.copy(), flow_threshold))
                                    except queue.Full:
                                        pass
                            prev_gray = gray
                        
                        # Get latest stationary state
                        with self.flow_lock:
                            is_stationary = self.latest_is_stationary
                            flow_mag = self.latest_flow_mag
                    else:
                        is_stationary = True
                    
                    logic_status = self.stop_sign_logic.update(timestamp_det, detections, is_stationary)

                    # Log on changes or alerts
                    if logic_status:
                        if logic_status["alert_fired"] and not getattr(self, "_last_alert", False):
                            print(f"\n>>> [VIOLATION ALERT] Stop Sign violation alert fired at time {timestamp_det:.2f}s! <<<")
                            self._last_alert = True
                        if not logic_status["alert_fired"]:
                            self._last_alert = False
                            
                        if logic_status["vehicle_stopped"] and not getattr(self, "_last_stopped", False):
                            print(f"\n>>> [COMPLIANCE VERIFIED] Vehicle successfully stopped at sign (time: {timestamp_det:.2f}s) <<<")
                            self._last_stopped = True
                        if not logic_status["vehicle_stopped"]:
                            self._last_stopped = False

                if self.matching_enabled and self.audio_receiver and self.audio_matcher:
                    # Match every 3 frames to save CPU, same as in interactive mode
                    if frame_idx_det % 3 == 0:
                        query_sec = self.audio_matcher.target_duration + 1.0
                        live_audio_window = self.audio_receiver.get_audio_window(query_sec)
                        current_score, is_matched = self.audio_matcher.match_live_audio(live_audio_window)
                        if is_matched:
                            last_match_time = time.time()
                            print(f"\n>>> [MATCH DETECTED] Correlation: {current_score:.2f} <<<")

                # --- 2. Draw HUD/overlays on frame if layout flag is active ---
                frame = self._draw_all_hud_overlays(
                    frame=frame,
                    detections=draw_detections,
                    logic_status=logic_status,
                    timestamp=timestamp_det,
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
        
        # Finalize all remaining active journeys
        if hasattr(self, "active_journeys") and self.active_journeys:
            active_ids = list(self.active_journeys.keys())
            for j_id in active_ids:
                self._finalize_journey(j_id)
                
        self.stopped = True
        
        # Join worker threads
        if self.det_thread:
            self.det_thread.join(timeout=1.0)
            self.det_thread = None
        if self.flow_thread:
            self.flow_thread.join(timeout=1.0)
            self.flow_thread = None
            
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

        if self.csv_file:
            try:
                self.csv_file.close()
                print("[Telemetry] Telemetry CSV file closed.")
            except Exception:
                pass
            self.csv_file = None
            self.csv_writer = None

        print("[Shutdown] Session closed cleanly.")

    def _detection_worker(self):
        """Background worker thread for running model inference in a pipeline."""
        import queue
        print("[Detection] Background worker thread started.")
        self._det_latencies = []
        while not self.stopped:
            try:
                # Get the frame package from the input queue
                item = self.det_in_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            try:
                img, ts, idx = item
                rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                t_det_start = time.perf_counter()
                detections = self.detector.predict(rgb_frame)
                t_det_end = time.perf_counter()
                
                det_elapsed = t_det_end - t_det_start
                self._det_latencies.append(det_elapsed)
                if len(self._det_latencies) > 15:
                    self._det_latencies.pop(0)
                avg_det_latency = sum(self._det_latencies) / len(self._det_latencies)
                self.current_det_fps = 1.0 / avg_det_latency if avg_det_latency > 0 else 0.0
                
                # Push the processed frame and its matching detections to the output queue
                self.det_out_queue.put((img, ts, idx, detections))
            except Exception as e:
                print(f"[Detection] Error in worker thread: {e}", file=sys.stderr)
            finally:
                self.det_in_queue.task_done()
        print("[Detection] Background worker thread stopped.")

    def _optical_flow_worker(self):
        """Background worker thread for calculating optical flow."""
        import queue
        print("[OpticalFlow] Background worker thread started.")
        while not self.stopped:
            try:
                prev_small, gray_small, flow_threshold = self.flow_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            try:
                t_flow_start = time.perf_counter()
                flow = cv2.calcOpticalFlowFarneback(
                    prev_small, gray_small,
                    None, 0.5, 2, 8, 2, 5, 1.0, 0
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                flow_mag = float(np.mean(mag))
                is_stationary = flow_mag < flow_threshold
                t_flow_end = time.perf_counter()
                
                flow_latency = t_flow_end - t_flow_start
                self._flow_latencies.append(flow_latency)
                if len(self._flow_latencies) > 15:
                    self._flow_latencies.pop(0)
                self.avg_flow_time = sum(self._flow_latencies) / len(self._flow_latencies)
                
                with self.flow_lock:
                    self.latest_is_stationary = is_stationary
                    self.latest_flow_mag = flow_mag
            except Exception as e:
                print(f"[OpticalFlow] Error in worker thread: {e}", file=sys.stderr)
            finally:
                self.flow_queue.task_done()
        print("[OpticalFlow] Background worker thread stopped.")

    def _load_json_file(self, filename):
        """Loads data from a JSON file in the recordings directory."""
        import json
        filepath = os.path.join(self.recordings_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_json_file(self, filename, data):
        """Saves data to a JSON file in the recordings directory."""
        import json
        filepath = os.path.join(self.recordings_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"[Error] Failed to save {filename}: {e}", file=sys.stderr)

    def _process_journeys_and_events(self, tracked_stracks, timestamp, current_score):
        """Processes BYTETracker outputs, manages journeys and events lifecycle, and updates JSON files."""
        # Get active tracks that are stop signs
        stop_tracks = [t for t in tracked_stracks if t.class_name == "stop"]
        current_stop_ids = {int(t.track_id) for t in stop_tracks}
        
        # 1. Start new journeys for new stop sign detections
        for track in stop_tracks:
            track_id = int(track.track_id)
            if track_id not in self.active_journeys:
                # Create journey
                system_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.active_journeys[track_id] = {
                    "journey_id": track_id,
                    "start_time": system_time,
                    "end_time": None,
                    "event_status": "no event conducted",
                    "event_captured": False,
                    "frames_tracked": 1,
                    "audio_matched_during_journey": False,
                    "vehicle_stationary_during_journey": False,
                    "has_stopped": False,
                    "horn_while_stopped": False,
                    "stationary_start_time": None
                }
                
                # Append to journey.json
                journeys = self._load_json_file("journey.json")
                if not any(j.get("journey_id") == track_id for j in journeys):
                    journeys.append({
                        "journey_id": track_id,
                        "start_time": system_time,
                        "end_time": None,
                        "event_status": "no event conducted",
                        "reason": "didnt get stable, audio not matching(didnt blow horn)"
                    })
                    self._save_json_file("journey.json", journeys)
                    print(f"[Journey] Created new journey for Stop Sign ID: {track_id}")
            else:
                # Increment frame count
                self.active_journeys[track_id]["frames_tracked"] += 1
                
            # 2. Check if track is stable, vehicle is stationary, and audio match crosses threshold to trigger event
            journey = self.active_journeys[track_id]
            is_stable = journey["frames_tracked"] >= 3
            
            is_stationary = True
            if self.vehicle_stationary_logic_enabled:
                with self.flow_lock:
                    is_stationary = self.latest_is_stationary
            
            # Record if vehicle is stationary and track stop duration
            if is_stationary:
                journey["vehicle_stationary_during_journey"] = True
                if journey.get("stationary_start_time") is None:
                    journey["stationary_start_time"] = timestamp
                else:
                    duration = timestamp - journey["stationary_start_time"]
                    if is_stable and duration >= self.stop_sign_logic.required_stop_time:
                        journey["has_stopped"] = True
            else:
                journey["stationary_start_time"] = None
            
            match_crossed = current_score >= self.matching_threshold
            if match_crossed:
                journey["audio_matched_during_journey"] = True
                if is_stationary and journey.get("has_stopped", False):
                    journey["horn_while_stopped"] = True
            
            if journey.get("has_stopped", False) and journey.get("horn_while_stopped", False) and not journey["event_captured"]:
                # Trigger Event
                journey["event_captured"] = True
                journey["event_status"] = "conducted"
                
                system_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                event_obj = {
                    "event_id": track_id,  # "u can take the object id only"
                    "journey_id": track_id,
                    "timestamp": system_time,
                    "audio_match_score": float(current_score)
                }
                
                # Append to event.json
                events = self._load_json_file("event.json")
                if not any(e.get("event_id") == track_id for e in events):
                    events.append(event_obj)
                    self._save_json_file("event.json", events)
                    print(f"[Event] Captured event for Stop Sign ID: {track_id} (Score: {current_score:.2f})")
                
                # Update event_status in journey.json and remove reason field since event was successfully conducted
                journeys = self._load_json_file("journey.json")
                for j in journeys:
                    if j.get("journey_id") == track_id:
                        j["event_status"] = "conducted"
                        if "reason" in j:
                            del j["reason"]
                        break
                self._save_json_file("journey.json", journeys)

        # 3. Handle ended journeys (seen previously but not in current frame)
        active_ids = list(self.active_journeys.keys())
        for j_id in active_ids:
            if j_id not in current_stop_ids:
                j_mem = self.active_journeys[j_id]
                j_mem["missed_count"] = j_mem.get("missed_count", 0) + 1
                if j_mem["missed_count"] >= 15: # allow 15 frames of dropout
                    self._finalize_journey(j_id)
            else:
                if "missed_count" in self.active_journeys[j_id]:
                    self.active_journeys[j_id]["missed_count"] = 0

    def _finalize_journey(self, journey_id):
        """Finalizes an active journey, sets its end time and updates the JSON file."""
        if journey_id in self.active_journeys:
            j_mem = self.active_journeys[journey_id]
            system_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            
            # Load and update journey.json
            journeys = self._load_json_file("journey.json")
            for j in journeys:
                if j.get("journey_id") == journey_id:
                    j["end_time"] = system_time
                    j["event_status"] = j_mem["event_status"]
                    
                    # If event was not conducted, record the reason
                    if j["event_status"] != "conducted":
                        reasons = []
                        if j_mem["frames_tracked"] < 3:
                            reasons.append("didnt get stable")
                        
                        if self.vehicle_stationary_logic_enabled and not j_mem.get("vehicle_stationary_during_journey", False):
                            reasons.append("vehicle not stationary")
                        
                        # Check audio/horn sequence
                        if not j_mem.get("audio_matched_during_journey", False):
                            reasons.append("audio not matching(didnt blow horn)")
                        elif j_mem.get("vehicle_stationary_during_journey", False) and not j_mem.get("horn_while_stopped", False):
                            reasons.append("audio not matching(horn not blown while stopped)")
                        
                        if reasons:
                            j["reason"] = ", ".join(reasons)
                        else:
                            j["reason"] = "unknown reason"
                    else:
                        if "reason" in j:
                            del j["reason"]
                    break
            self._save_json_file("journey.json", journeys)
            
            # Remove from active
            del self.active_journeys[journey_id]
            print(f"[Journey] Finalized journey for Stop Sign ID: {journey_id} (Status: {j_mem['event_status']})")
