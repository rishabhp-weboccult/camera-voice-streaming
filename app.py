#!/usr/bin/env python3
import os
import sys
import json

# Add current directory to path to locate objects and managers
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from managers import DeviceManager, StreamManager

from utils import load_config, CONFIG_PATH, TXT_PATH

def main():
    print("="*65)
    print("      CAMERA & VOICE ALIGNMENT CORE APPLICATION SHELL")
    print("="*65)
    
    # 1. Load configuration
    config = load_config()
    
    # 2. Trigger fresh hardware probe if requested
    force_discover = config.get("force_discover", False)
    if force_discover or not os.path.exists(TXT_PATH):
        print("\n[Discovery] Running hardware discovery probe...")
        dm = DeviceManager(devices_txt_path=TXT_PATH)
        dm.discover_and_report()
        
        # Reset force_discover flag
        if force_discover:
            config["force_discover"] = False
            try:
                with open(CONFIG_PATH, "w") as f:
                    json.dump(config, f, indent=4)
            except Exception:
                pass

    # 3. Print loaded variables
    video_device = config.get("video_device")
    alsa_device = config.get("alsa_device")
    video_size = config.get("video_size", "1280x720")
    fps = int(config.get("video_fps", 15))
    sample_rate = int(config.get("sampling_rate", 48000))
    channels = int(config.get("channels", 1))
    
    record_interval = int(config.get("record_interval", 10))
    recordings_dir = config.get("recordings_dir", "./recordings")
    
    matching_enabled = config.get("audio_matching_enabled", True)
    matching_threshold = float(config.get("audio_matching_threshold", 0.50))
    matching_target_path = config.get("audio_matching_target_path", "/home/wot-rishabh/Downloads/recording.wav")
    
    show_window = config.get("show_window", True)
    recording_enabled = config.get("recording_enabled", True)
    
    run_on_video = config.get("run_on_video", False)
    video_input_path = config.get("video_input_path", "")
    
    print(f"\n[Config] Active Configuration loaded from config.json:")
    if run_on_video:
        print(f"  Execution Mode  : FILE SIMULATION MODE")
        print(f"  Input Video File: {video_input_path}")
    else:
        print(f"  Execution Mode  : LIVE HARDWARE CAPTURE MODE")
        print(f"  Active Video    : {video_device} ({video_size} @ {fps} FPS)")
        print(f"  Active Audio    : {alsa_device} ({sample_rate}Hz, {channels} ch)")
    print(f"  Display Window  : {'ENABLED' if show_window else 'DISABLED (Headless)'}")
    print(f"  Audio Matching  : {'ENABLED' if matching_enabled else 'DISABLED'}")
    if matching_enabled:
        print(f"  Match Target    : {matching_target_path}")
        print(f"  Match Threshold : {matching_threshold}")
    if recording_enabled and record_interval > 0:
        print(f"  Recording       : Active (MP4 chunks saved every {record_interval}s to '{recordings_dir}')")
    else:
        print(f"  Recording       : Disabled")
    print("-----------------------------------------------------------------")
    
    # 4. Bootstrap and start session
    sm = StreamManager(config)
    if sm.start_session():
        sm.run_stream_loop()

if __name__ == "__main__":
    main()
