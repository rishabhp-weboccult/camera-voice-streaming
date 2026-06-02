#!/usr/bin/env python3
import os
import sys
import glob
import subprocess
import re
from datetime import datetime

DEFAULT_TXT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "devices.txt")

def get_video_devices():
    """Finds all video capture devices via sysfs and dev paths."""
    video_devices = []
    by_id_paths = glob.glob("/dev/v4l/by-id/*")
    video_nodes = sorted(glob.glob("/dev/video*"))
    
    for node in video_nodes:
        device_info = {
            "path": node,
            "name": os.path.basename(node),
            "by_id_path": None,
        }
        
        # Link back to a by-id path if it exists
        for by_id in by_id_paths:
            if os.path.exists(by_id) and os.path.samefile(node, by_id):
                device_info["by_id_path"] = by_id
                break
                
        # Try to use v4l2-ctl to get the device card name
        try:
            res = subprocess.run(["v4l2-ctl", "--device=" + node, "--info"], capture_output=True, text=True, timeout=2)
            if res.returncode == 0:
                card_match = re.search(r"Card type\s*:\s*(.+)", res.stdout)
                if card_match:
                    device_info["name"] = card_match.group(1).strip()
        except Exception:
            pass
            
        video_devices.append(device_info)
        
    return video_devices

def get_audio_devices():
    """Finds ALSA capture devices by running arecord -l and parsing output."""
    audio_devices = []
    
    try:
        res = subprocess.run(["arecord", "-l"], capture_output=True, text=True, timeout=3)
        if res.returncode == 0:
            lines = res.stdout.splitlines()
            for line in lines:
                match = re.match(r"card\s+(\d+):\s+(.+),\s+device\s+(\d+):\s+(.+)", line, re.IGNORECASE)
                if match:
                    card_num = match.group(1)
                    card_name = match.group(2).strip()
                    dev_num = match.group(3)
                    dev_name = match.group(4).strip()
                    
                    alsa_id = f"hw:{card_num},{dev_num}"
                    
                    device_info = {
                        "alsa_device": alsa_id,
                        "card_name": card_name,
                        "device_name": dev_name,
                        "sampling_rates": [],
                        "channels": []
                    }
                    
                    hw_params = query_alsa_hw_params(alsa_id)
                    if hw_params:
                        device_info.update(hw_params)
                        
                    audio_devices.append(device_info)
    except FileNotFoundError:
        if os.path.exists("/proc/asound/cards"):
            try:
                with open("/proc/asound/cards", "r") as f:
                    content = f.read()
                cards = re.findall(r"\s*(\d+)\s+\[([^\]]+)\]:\s+(.+)", content)
                for card_num, short_name, long_name in cards:
                    alsa_id = f"hw:{card_num},0"
                    audio_devices.append({
                        "alsa_device": alsa_id,
                        "card_name": short_name.strip(),
                        "device_name": long_name.strip(),
                        "sampling_rates": [44100, 48000],
                        "channels": [1, 2]
                    })
            except Exception:
                pass
    except Exception as e:
        print(f"Error listing audio devices: {e}", file=sys.stderr)
        
    return audio_devices

def query_alsa_hw_params(alsa_device):
    """Checks hardware parameters using arecord dump-hw-params."""
    try:
        res = subprocess.run(["arecord", "-D", alsa_device, "--dump-hw-params", "-d", "1"], 
                             capture_output=True, text=True, timeout=2)
        text = res.stdout + "\n" + res.stderr
        
        rates = []
        channels = []
        
        rate_match = re.search(r"RATE:\s*(.+)", text)
        if rate_match:
            rate_str = rate_match.group(1)
            range_match = re.match(r"\[(\d+)\s+(\d+)\]", rate_str)
            if range_match:
                min_r, max_r = int(range_match.group(1)), int(range_match.group(2))
                for r in [8000, 16000, 22050, 32000, 44100, 48000, 96000]:
                    if min_r <= r <= max_r:
                        rates.append(r)
            else:
                rates = [int(r) for r in re.findall(r"\d+", rate_str)]
                
        chan_match = re.search(r"CHANNELS:\s*(.+)", text)
        if chan_match:
            chan_str = chan_match.group(1)
            range_match = re.match(r"\[(\d+)\s+(\d+)\]", chan_str)
            if range_match:
                min_c, max_c = int(range_match.group(1)), int(range_match.group(2))
                channels = list(range(min_c, max_c + 1))
            else:
                channels = [int(c) for c in re.findall(r"\d+", chan_str)]
                
        return {
            "sampling_rates": rates if rates else [8000, 16000, 44100, 48000],
            "channels": channels if channels else [1, 2]
        }
    except Exception:
        return None

def discover_all():
    """Discovers all hardware resources and returns a structured dictionary."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "timestamp": timestamp,
        "video_devices": get_video_devices(),
        "audio_devices": get_audio_devices()
    }

def save_to_text(data, filepath=DEFAULT_TXT_PATH):
    """Saves hardware data as a clean human-readable text file."""
    try:
        with open(filepath, "w") as f:
            f.write("==================================================\n")
            f.write("          DISCOVERED HARDWARE DEVICES             \n")
            f.write("==================================================\n")
            f.write(f"Generated at: {data['timestamp']}\n\n")
            
            f.write("--- Video Devices ---\n")
            if not data["video_devices"]:
                f.write("  No video devices found.\n")
            for idx, vd in enumerate(data["video_devices"], 1):
                f.write(f"[{idx}] Name: {vd['name']}\n")
                f.write(f"    Path: {vd['path']}\n")
                if vd.get('by_id_path'):
                    f.write(f"    By-ID Path: {vd['by_id_path']}\n")
                f.write("\n")
                
            f.write("--- Audio Capture Devices (ALSA) ---\n")
            if not data["audio_devices"]:
                f.write("  No audio capture devices found.\n")
            for idx, ad in enumerate(data["audio_devices"], 1):
                f.write(f"[{idx}] Card Name: {ad['card_name']}\n")
                f.write(f"    Device Name: {ad['device_name']}\n")
                f.write(f"    ALSA Identifier: {ad['alsa_device']}\n")
                f.write(f"    Supported Rates: {', '.join(map(str, ad['sampling_rates']))}\n")
                f.write(f"    Supported Channels: {', '.join(map(str, ad['channels']))}\n")
                f.write("\n")
                
        print(f"Successfully wrote hardware listing to: {filepath}")
        return True
    except Exception as e:
        print(f"Error saving TXT report to {filepath}: {e}", file=sys.stderr)
        return False

def parse_devices_text(filepath=DEFAULT_TXT_PATH):
    """
    Parses the human-readable devices.txt back into a structured dictionary
    so the runner script can read it without running another live hardware probe.
    """
    if not os.path.exists(filepath):
        return None
        
    try:
        with open(filepath, "r") as f:
            content = f.read()
            
        data = {
            "video_devices": [],
            "audio_devices": []
        }
        
        # Split sections
        video_sec = ""
        audio_sec = ""
        
        parts = content.split("--- Audio Capture Devices (ALSA) ---")
        if len(parts) == 2:
            video_sec = parts[0]
            audio_sec = parts[1]
        else:
            video_sec = content
            
        # Parse video devices
        # Matches blocks like:
        # [1] Name: Droidcam
        #     Path: /dev/video0
        #     By-ID Path: ...
        video_blocks = re.findall(r"\[\d+\] Name:\s*(.+?)\n\s+Path:\s*(.+?)(?:\n\s+By-ID Path:\s*(.+?))?\n\n", video_sec, re.MULTILINE)
        for name, path, by_id in video_blocks:
            data["video_devices"].append({
                "name": name.strip(),
                "path": path.strip(),
                "by_id_path": by_id.strip() if by_id else None
            })
            
        # Parse audio devices
        # Matches blocks like:
        # [1] Card Name: ...
        #     Device Name: ...
        #     ALSA Identifier: ...
        #     Supported Rates: ...
        #     Supported Channels: ...
        audio_blocks = re.findall(
            r"\[\d+\] Card Name:\s*(.+?)\n\s+Device Name:\s*(.+?)\n\s+ALSA Identifier:\s*(.+?)\n\s+Supported Rates:\s*(.+?)\n\s+Supported Channels:\s*(.+?)\n\n",
            audio_sec, re.MULTILINE
        )
        for card, device, alsa_id, rates_str, chans_str in audio_blocks:
            rates = [int(r.strip()) for r in rates_str.split(",") if r.strip().isdigit()]
            chans = [int(c.strip()) for c in chans_str.split(",") if c.strip().isdigit()]
            data["audio_devices"].append({
                "card_name": card.strip(),
                "device_name": device.strip(),
                "alsa_device": alsa_id.strip(),
                "sampling_rates": rates,
                "channels": chans
            })
            
        return data
    except Exception as e:
        print(f"Error parsing {filepath}: {e}", file=sys.stderr)
        return None

if __name__ == "__main__":
    print("--- Starting Hardware Discovery Probe ---")
    hw_data = discover_all()
    save_to_text(hw_data)
