#!/usr/bin/env python3
import os
import sys
import time
import json
import glob
import re
try:
    import cv2
except ImportError:
    cv2 = None
from datetime import datetime

# Add the current directory to python path to import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import hardware_details
import streamer

DEFAULT_CONFIG_PATH = os.path.join(current_dir, "config.json")
DEFAULT_TXT_PATH = os.path.join(current_dir, "devices.txt")

def setup_first_run():
    """
    Called when config.json does not exist.
    Runs hardware discovery, writes devices.txt, establishes optimal defaults,
    and writes config.json.
    """
    print("[First Run] Configuration not found. Running initial hardware probe...")
    hw_data = hardware_details.discover_all()
    hardware_details.save_to_text(hw_data, DEFAULT_TXT_PATH)
    
    # Establish defaults based on discovered devices
    # Video Selection
    video_device = "/dev/video0"
    if hw_data["video_devices"]:
        # Try to find a USB device first
        usb_video = next((vd for vd in hw_data["video_devices"] if vd["by_id_path"]), None)
        if usb_video:
            video_device = usb_video["by_id_path"]
        else:
            video_device = hw_data["video_devices"][0]["path"]
            
    # Audio Selection
    alsa_device = "hw:1,0"
    if hw_data["audio_devices"]:
        # Try to find hw:1,0 first
        has_hw1 = any(ad["alsa_device"] == "hw:1,0" for ad in hw_data["audio_devices"])
        if not has_hw1:
            alsa_device = hw_data["audio_devices"][0]["alsa_device"]
            
    # Compile Config Profile
    config_data = {
        "video_device": video_device,
        "alsa_device": alsa_device,
        "video_size": "1280x720",
        "video_fps": 15,
        "sampling_rate": 48000,
        "channels": 1,
        "record_interval": 10,
        "recordings_dir": "./recordings",
        "force_discover": False,
        "__meta__": "Configuration profile loaded from config.json instead of .env. Modify values to change defaults."
    }
    
    # Save config.json
    try:
        with open(DEFAULT_CONFIG_PATH, "w") as f:
            json.dump(config_data, f, indent=4)
        print(f"[First Run] Created default configuration file at: {DEFAULT_CONFIG_PATH}")
    except Exception as e:
        print(f"[First Run] ERROR writing config.json: {e}", file=sys.stderr)
        
    return config_data

def load_config():
    """Loads active configurations from config.json."""
    if not os.path.exists(DEFAULT_CONFIG_PATH):
        return setup_first_run()
        
    try:
        with open(DEFAULT_CONFIG_PATH, "r") as f:
            config = json.load(f)
            # Ensure essential keys exist
            required_keys = ["video_device", "alsa_device", "video_size", "video_fps", "sampling_rate", "channels"]
            if all(k in config for k in required_keys):
                return config
            else:
                print("[Config] WARNING: config.json is missing required keys. Re-initializing...", file=sys.stderr)
                return setup_first_run()
    except Exception as e:
        print(f"[Config] Error loading config.json: {e}. Re-initializing...", file=sys.stderr)
        return setup_first_run()

def release_hardware_devices(video_device, alsa_device):
    """
    Scans /proc to find and terminate any zombie processes holding exclusive locks
    on either the ALSA audio capture card or the V4L2 video camera node.
    """
    pids_to_kill = set()
    my_pid = os.getpid()
    
    # 1. Determine audio capture file name pattern
    card = "*"
    if "hw:" in alsa_device or "dsnoop:" in alsa_device:
        match = re.search(r"(\d+)", alsa_device)
        if match:
            card = match.group(1)
    audio_pattern = f"pcmC{card}D"
    
    # 2. Determine video device name pattern
    real_video_path = ""
    if video_device and os.path.exists(video_device):
        real_video_path = os.path.realpath(video_device)
        
    print(f"[DeviceManager] Probing system locks for Video ({os.path.basename(video_device)}) and Audio ({alsa_device})...")
    
    for pid_dir in glob.glob("/proc/[0-9]*"):
        try:
            pid = int(os.path.basename(pid_dir))
            if pid == my_pid:
                continue
            
            fd_path = os.path.join(pid_dir, "fd")
            if not os.path.exists(fd_path):
                continue
                
            for fd in os.listdir(fd_path):
                try:
                    link = os.readlink(os.path.join(fd_path, fd))
                    # Check audio device match (c means capture)
                    if "pcmC" in link and link.endswith("c") and audio_pattern in link:
                        pids_to_kill.add(pid)
                    # Check video device match
                    elif real_video_path and real_video_path in link:
                        pids_to_kill.add(pid)
                except Exception:
                    pass
        except Exception:
            pass
            
    if pids_to_kill:
        print(f"[DeviceManager] Identified {len(pids_to_kill)} conflicting process(es) holding hardware locks: {pids_to_kill}")
        for pid in pids_to_kill:
            try:
                with open(f"/proc/{pid}/comm", "r") as f:
                    comm = f.read().strip()
                print(f"  -> Releasing lock: terminating process {pid} ({comm})...")
                os.kill(pid, 15)  # SIGTERM
                time.sleep(0.1)
                if os.path.exists(f"/proc/{pid}"):
                    os.kill(pid, 9)   # SIGKILL
            except Exception as e:
                print(f"  -> Failed to terminate process {pid}: {e}", file=sys.stderr)
        time.sleep(0.5)  # Give drivers time to recycle card state
        print("[DeviceManager] Conflict resolution complete. Devices successfully released.")
    else:
        print("[DeviceManager] No conflicting device locks detected.")

def run_opencv_and_ffplay_mode(video_device, alsa_device, width, height, fps, sample_rate, channels, record_interval=0, recordings_dir="./recordings"):
    """
    Mode 1:
    - Runs a thread-safe OpenCV stream to capture and show video frames.
    - Draws visual overlays, including FPS on top-right, resolution, active configuration.
    - Runs ffplay in a background subprocess to monitor live audio feed.
    - Writes the displayed screen frames (with overlays) and ALSA audio to MP4 chunks.
    """
    # 0. Automatically release hardware locks held by conflicting processes
    release_hardware_devices(video_device, alsa_device)
    
    print("\n" + "="*70)
    print("Mode 1: Interactive OpenCV Video Stream + FFplay Background Audio Monitor")
    print("="*70 + "\n")
    
    # 1. Start audio monitor in background
    audio_monitor = None
    try:
        audio_monitor = streamer.FFplayAudioMonitor(
            alsa_device=alsa_device,
            sample_rate=sample_rate,
            channels=channels
        )
        audio_monitor.start()
    except Exception as e:
        print(f"[Error] Failed to initialize live audio monitoring: {e}", file=sys.stderr)
        print("[Info] Continuing with video stream only...")

    # 2. Start high-performance video stream
    video_stream = None
    try:
        video_stream = streamer.OpenCVVideoStream(
            device_path=video_device,
            width=width,
            height=height,
            fps=fps
        )
        video_stream.start()
    except Exception as e:
        print(f"[Critical Error] Failed to initialize OpenCV Video Stream: {e}", file=sys.stderr)
        if audio_monitor:
            audio_monitor.stop()
        return

    # 2.5 Start background HUD recorder if configured
    hud_recorder = None
    if record_interval > 0:
        try:
            hud_recorder = streamer.OpenCVHUDRecorder(
                alsa_device=alsa_device,
                sample_rate=sample_rate,
                channels=channels,
                width=width,
                height=height,
                fps=fps,
                interval=record_interval,
                output_dir=recordings_dir
            )
            hud_recorder.start()
        except Exception as e:
            print(f"[Error] Failed to initialize video/audio recorder: {e}", file=sys.stderr)
            hud_recorder = None

    # 3. Create interactive window
    window_name = "Camera & Voice - High Performance Streamer"
    screenshots_dir = os.path.join(current_dir, "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)
    
    print("\n--- Controls ---")
    print("  'q' : Quit stream")
    print("  's' : Save screenshot")
    if record_interval > 0:
        print(f"  * Note: Automatic MP4 recording active (Interval: {record_interval}s) *")
    print("----------------\n")
    
    try:
        while True:
            frame = video_stream.read()
            if frame is None:
                time.sleep(0.01)
                continue
                
            h, w, _ = frame.shape
            
            # semi-transparent HUD header background
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 55), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Status badge (Green pulsing indicator mock)
            cv2.circle(frame, (20, 28), 6, (0, 255, 0), -1)
            
            # Info texts
            fps_val = video_stream.get_fps()
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            cv2.putText(frame, "LIVE REC", (35, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f"Cam: {os.path.basename(video_device)} ({w}x{h})", (140, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(frame, f"Mic: {alsa_device}", (450, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(frame, time_str, (w - 290, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # FPS on top right corner (bright green and bold)
            fps_text = f"FPS: {fps_val:.1f}"
            cv2.putText(frame, fps_text, (w - 110, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2)
            
            # Instruction footer overlay
            cv2.rectangle(frame, (0, h - 30), (w, h), (30, 30, 30), -1)
            cv2.putText(frame, "[q]: Quit  |  [s]: Take Screenshot", (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Write to recording chunk and tick interval checks
            if hud_recorder:
                hud_recorder.write_frame(frame)
                hud_recorder.tick(frame)
                
            # Show processed frame
            try:
                cv2.imshow(window_name, frame)
            except Exception as e:
                print(f"\n[Warning] Cannot display window: {e}")
                print("[Info] Running in Headless Logging mode. Realtime capture active.")
                print("[Info] Press Ctrl+C to terminate.")
                while True:
                    # Tick recorder even in headless mode
                    if hud_recorder:
                        hud_recorder.write_frame(frame)
                        hud_recorder.tick(frame)
                    fps_val = video_stream.get_fps()
                    print(f"\rStreaming live :: FPS: {fps_val:.1f} | Frame Dimensions: {w}x{h} | Audio: {alsa_device} active.", end="")
                    time.sleep(1.0)
            
            # Key polling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                shot_name = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                shot_path = os.path.join(screenshots_dir, shot_name)
                cv2.imwrite(shot_path, frame)
                print(f"[Screenshot] Frame saved successfully to: {shot_path}")
                
    except KeyboardInterrupt:
        print("\n[Signal] Interrupt detected. Exiting gracefully...")
    finally:
        if hud_recorder:
            hud_recorder.stop()
        if video_stream:
            video_stream.stop()
        if audio_monitor:
            audio_monitor.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("[Shutdown] Mode 1 terminated cleanly.")

# def run_combined_ffplay_mode(video_device, alsa_device, width, height, fps, sample_rate, channels):
#     """
#     Mode 2:
#     - Launches a unified FFplay subprocess capturing both V4L2 and ALSA inputs.
#     - Synchronized playback window with lower level kernel latency.
#     """
#     print("\n" + "="*70)
#     print("Mode 2: Synchronized FFplay Audio/Video Combined Preview")
#     print("="*70 + "\n")
#     
#     streamer_obj = None
#     try:
#         streamer_obj = streamer.FFplayCombinedStreamer(
#             video_device=video_device,
#             alsa_device=alsa_device,
#             width=width,
#             height=height,
#             fps=fps,
#             sample_rate=sample_rate,
#             channels=channels
#         )
#         streamer_obj.start()
#         
#         print("\nSynchronized media feed launched in active player window.")
#         print("To close, close the window or press Ctrl+C in this console terminal.")
#         
#         while True:
#             if streamer_obj.process.poll() is not None:
#                 print("[FFplay] Preview window closed by user.")
#                 break
#             time.sleep(0.5)
#             
#     except KeyboardInterrupt:
#         print("\n[Signal] Interrupt received. Exiting...")
#     finally:
#         if streamer_obj:
#             streamer_obj.stop()
#         print("[Shutdown] Mode 2 terminated cleanly.")

def main():
    print("="*60)
    print("   CAMERA & VOICE HIGH PERFORMANCE STREAMING SHELL")
    print("="*60)
    
    # 1. Load Config from JSON in place of .env
    config = load_config()
    
    # 2. Handle Devices Text discovery if requested in config.json
    force_discover = config.get("force_discover", False)
    if force_discover or not os.path.exists(DEFAULT_TXT_PATH):
        print("\n[Discovery] Running hardware discovery probe...")
        hw_data = hardware_details.discover_all()
        hardware_details.save_to_text(hw_data, DEFAULT_TXT_PATH)
        
        # Reset force_discover flag inside config.json so it doesn't loop
        if force_discover:
            config["force_discover"] = False
            try:
                with open(DEFAULT_CONFIG_PATH, "w") as f:
                    json.dump(config, f, indent=4)
            except Exception:
                pass
    elif not os.path.exists(DEFAULT_TXT_PATH):
        print(f"\n[Discovery] {DEFAULT_TXT_PATH} not found. Running hardware discovery...")
        hw_data = hardware_details.discover_all()
        hardware_details.save_to_text(hw_data, DEFAULT_TXT_PATH)

    # Extract settings
    video_device = config.get("video_device")
    alsa_device = config.get("alsa_device")
    video_size = config.get("video_size", "1280x720")
    
    try:
        w_str, h_str = video_size.split("x")
        width, height = int(w_str), int(h_str)
    except Exception:
        width, height = 1280, 720
        
    fps = int(config.get("video_fps", 15))
    sample_rate = int(config.get("sampling_rate", 48000))
    channels = int(config.get("channels", 1))
    
    # Pull recording interval and directories
    record_interval = int(config.get("record_interval", 10))
    recordings_dir = config.get("recordings_dir", "./recordings")
    
    print(f"\n[Config] Active Configuration loaded from {DEFAULT_CONFIG_PATH}:")
    print(f"  Active Video: {video_device}")
    print(f"  Active Audio: {alsa_device}")
    print(f"  Profile     : {width}x{height} @ {fps} FPS, {sample_rate} Hz, {channels} ch")
    if record_interval > 0:
        print(f"  Recording   : Active (Muxed MP4 chunks saved every {record_interval}s to '{recordings_dir}')")
    else:
        print(f"  Recording   : Disabled")
    print("--------------------------------------------------")
    
    # 3. Launch interactive streaming session
    run_opencv_and_ffplay_mode(
        video_device=video_device,
        alsa_device=alsa_device,
        width=width,
        height=height,
        fps=fps,
        sample_rate=sample_rate,
        channels=channels,
        record_interval=record_interval,
        recordings_dir=recordings_dir
    )

if __name__ == "__main__":
    main()
