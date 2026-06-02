#!/usr/bin/env python3
import os
import sys
import time
from datetime import datetime

try:
    import cv2
except ImportError:
    cv2 = None

from objects import OpenCVVideoStream, AudioReceiver, AudioMatcher, FFplayAudioMonitor, OpenCVHUDRecorder, VirtualVideoStream, VirtualAudioReceiver
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
        
        # Configuration parameters
        self.video_device = config.get("video_device")
        self.alsa_device = config.get("alsa_device")
        self.video_size = config.get("video_size", "1280x720")
        self.show_window = config.get("show_window", True)
        
        try:
            w_str, h_str = self.video_size.split("x")
            self.width, self.height = int(w_str), int(h_str)
        except Exception:
            self.width, self.height = 1280, 720
            
        self.fps = int(config.get("video_fps", 15))
        self.sample_rate = int(config.get("sampling_rate", 48000))
        self.channels = int(config.get("channels", 1))
        
        self.record_interval = int(config.get("record_interval", 10))
        self.recordings_dir = config.get("recordings_dir", "./recordings")
        self.recording_enabled = config.get("recording_enabled", True)
        
        self.matching_enabled = config.get("audio_matching_enabled", True)
        self.matching_threshold = float(config.get("audio_matching_threshold", 0.50))
        self.matching_target_path = config.get("audio_matching_target_path", "/home/wot-rishabh/Downloads/recording.wav")
        
        # Video file simulation parameters
        self.run_on_video = config.get("run_on_video", False)
        self.video_input_path = config.get("video_input_path", "")
        self.audio_monitor_process = None
        
        # Application entities
        self.device_manager = DeviceManager()
        self.video_stream = None
        self.audio_monitor = None
        self.hud_recorder = None
        self.audio_receiver = None
        self.audio_matcher = None
        
        self.window_name = "Camera & Voice - High Performance Streamer"
        self.screenshots_dir = "./screenshots"
        
    def start_session(self):
        """Prepares resources, clears hardware locks or starts video file demuxers."""
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
                        audio_receiver=self.audio_receiver
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
                    audio_receiver=self.audio_receiver
                )
                self.hud_recorder.start()
            except Exception as e:
                print(f"[Error] Failed to initialize video/audio recorder: {e}", file=sys.stderr)
                self.hud_recorder = None

        os.makedirs(self.screenshots_dir, exist_ok=True)
        return True

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
        print("----------------------------\n")
        
        frame_idx = 0
        current_score = 0.0
        is_matched = False
        last_match_time = 0.0
        match_cooldown_sec = 2.0
        
        try:
            while True:
                frame = self.video_stream.read()
                if frame is None:
                    time.sleep(0.01)
                    continue
                    
                h, w, _ = frame.shape
                frame_idx += 1
                
                # --- 1. Compute Audio Matching in real time (every 3 frames to save CPU) ---
                if self.matching_enabled and self.audio_receiver and self.audio_matcher and self.audio_matcher.target_loaded:
                    if frame_idx % 3 == 0:
                        query_sec = self.audio_matcher.target_duration + 1.0
                        live_audio_window = self.audio_receiver.get_audio_window(query_sec)
                        current_score, is_matched = self.audio_matcher.match_live_audio(live_audio_window)
                        
                        if is_matched:
                            last_match_time = time.time()
                            print(f"\r[MATCH DETECTED] Correlation: {current_score:.2f} at {datetime.now().strftime('%H:%M:%S')}")
                
                # --- 2. Draw Semi-transparent Header Overlay ---
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 55), (15, 15, 15), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                
                # Pulse status badge
                cv2.circle(frame, (20, 28), 6, (0, 255, 0), -1)
                
                fps_val = self.video_stream.get_fps()
                time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                cv2.putText(frame, "LIVE REC", (35, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, f"Cam: {os.path.basename(self.video_device)} ({w}x{h})", (140, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(frame, f"Mic: {self.alsa_device}", (450, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(frame, time_str, (w - 290, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                # Render FPS count
                cv2.putText(frame, f"FPS: {fps_val:.1f}", (w - 110, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2)
                
                # --- 3. Draw Audio Matcher HUD Panel Widget ---
                center_x = w // 2
                box_x1 = center_x - 160
                box_y1 = h - 85
                box_x2 = center_x + 160
                box_y2 = h - 45
                
                # Matcher background
                overlay_box = frame.copy()
                cv2.rectangle(overlay_box, (box_x1, box_y1), (box_x2, box_y2), (10, 10, 10), -1)
                cv2.addWeighted(overlay_box, 0.6, frame, 0.4, 0, frame)
                
                match_active = (time.time() - last_match_time) < match_cooldown_sec
                
                if not self.matching_enabled:
                    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (50, 50, 50), 1)
                    cv2.putText(frame, "Audio Matcher: DISABLED", (center_x - 85, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1)
                elif not self.audio_matcher.target_loaded:
                    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (50, 50, 150), 1)
                    cv2.putText(frame, "Matcher: Target Load Error", (center_x - 90, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 200), 1)
                else:
                    if match_active:
                        # Pulse green overlays
                        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)
                        cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 4) # screen glow
                        
                        cv2.putText(frame, f"MATCH DETECTED! Correlation: {current_score:.2f}", (center_x - 145, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                        
                        # Full green progress bar
                        cv2.rectangle(frame, (center_x - 140, h - 52), (center_x + 140, h - 48), (20, 80, 20), -1)
                        cv2.rectangle(frame, (center_x - 140, h - 52), (center_x + 140, h - 48), (0, 255, 0), -1)
                    else:
                        # Monitor listening overlay
                        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (150, 150, 150), 1)
                        cv2.putText(frame, f"Listening... Correlation: {current_score:.2f} / {self.matching_threshold}", (center_x - 130, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
                        
                        # Cyan bar matching correlation level
                        bar_w = int(280 * min(1.0, current_score / self.matching_threshold))
                        cv2.rectangle(frame, (center_x - 140, h - 52), (center_x + 140, h - 48), (40, 40, 40), -1)
                        if bar_w > 0:
                            cv2.rectangle(frame, (center_x - 140, h - 52), (center_x - 140 + bar_w, h - 48), (255, 180, 50), -1)
                
                # --- 4. Draw Footer Instructions ---
                cv2.rectangle(frame, (0, h - 30), (w, h), (30, 30, 30), -1)
                cv2.putText(frame, "[q]: Quit cleanly  |  [s]: Take Screenshot", (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                # --- 5. Record display frame to active chunk ---
                if self.hud_recorder:
                    self.hud_recorder.write_frame(frame)
                    self.hud_recorder.tick(frame)
                    
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
        frame = first_frame
        
        delay = 1.0 / self.fps
        last_log_time = 0.0
        
        try:
            frame_idx = 0
            while True:
                start_time = time.time()
                
                # Fetch latest frame from the video stream
                new_frame = self.video_stream.read()
                if new_frame is not None:
                    frame = new_frame
                
                frame_idx += 1
                if self.hud_recorder:
                    self.hud_recorder.write_frame(frame)
                    self.hud_recorder.tick(frame)
                
                if self.matching_enabled and self.audio_receiver and self.audio_matcher:
                    # Match every 3 frames to save CPU, same as in interactive mode
                    if frame_idx % 3 == 0:
                        query_sec = self.audio_matcher.target_duration + 1.0
                        live_audio_window = self.audio_receiver.get_audio_window(query_sec)
                        current_score, is_matched = self.audio_matcher.match_live_audio(live_audio_window)
                        if is_matched:
                            print(f"\n>>> [MATCH DETECTED] Correlation: {current_score:.2f} <<<")
                
                # Log once per second to prevent stdout spam
                now = time.time()
                if now - last_log_time >= 1.0:
                    fps_val = self.video_stream.get_fps()
                    print(f"\rStreaming live :: FPS: {fps_val:.1f} | Match Correlation: {current_score:.2f} | Audio capture active.", end="", flush=True)
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
