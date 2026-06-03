import os
import sys
import json

# Ensure workspace root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(current_dir)
if workspace_dir not in sys.path:
    sys.path.append(workspace_dir)

from managers import DeviceManager

CONFIG_PATH = os.path.join(workspace_dir, "config.json")
TXT_PATH = os.path.join(workspace_dir, "devices.txt")

def setup_first_run():
    """Establishes config.json and devices.txt optimal defaults based on discovered hardware."""
    print("[First Run] Configuration not found. Running hardware discovery probe...")
    dm = DeviceManager(devices_txt_path=TXT_PATH)
    hw_data = dm.discover_and_report()
    
    # Establish defaults based on discovered devices
    video_device = "/dev/video0"
    if hw_data and hw_data["video_devices"]:
        usb_video = next((vd for vd in hw_data["video_devices"] if vd.get("by_id_path")), None)
        if usb_video:
            video_device = usb_video["by_id_path"]
        else:
            video_device = hw_data["video_devices"][0]["path"]
            
    alsa_device = "hw:1,0"
    if hw_data and hw_data["audio_devices"]:
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
        "recording_enabled": True,
        "recording_cleanup_enabled": True,
        "max_recordings_size_mb": 1000.0,
        "force_discover": False,
        "show_window": True,
        "audio_matching_enabled": True,
        "audio_matching_threshold": 0.50,
        "audio_matching_target_path": "/home/wot-rishabh/Downloads/recording.wav",
        "run_on_video": False,
        "video_input_path": "/home/wot-rishabh/Downloads/video_file.mp4",
        "detection_enabled": True,
        "vehicle_stationary_logic_enabled": True,
        "overlay_hud_on_frame": True,
        "stop_class_name": "stop",
        "required_stop_time": 1.0,
        "flow_threshold": 0.5,
        "flow_skip": 3,
        "model_path": "MODEL/v2-yolov8n-det-480-20260501-datav1.onnx",
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
        "providers": [
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ],
        "classes": {
            "0": "stop"
        },
        "__meta__": "Configuration profile loaded from config.json. Modify values to change defaults."
    }
    
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(config_data, f, indent=4)
        print(f"[First Run] Created configuration profile at: {CONFIG_PATH}")
    except Exception as e:
        print(f"[First Run] ERROR writing config.json: {e}", file=sys.stderr)
        
    return config_data

def load_config():
    """Loads active configurations from config.json."""
    if not os.path.exists(CONFIG_PATH):
        return setup_first_run()
        
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
            required_keys = ["video_device", "alsa_device", "video_size", "video_fps", "sampling_rate", "channels"]
            if all(k in config for k in required_keys):
                return config
            else:
                print("[Config] WARNING: config.json is missing required keys. Re-initializing...", file=sys.stderr)
                return setup_first_run()
    except Exception as e:
        print(f"[Config] Error loading config.json: {e}. Re-initializing...", file=sys.stderr)
        return setup_first_run()
