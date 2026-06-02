#!/usr/bin/env python3
import os
import sys
import glob
import subprocess
import re
import time
from datetime import datetime

class DeviceManager:
    """
    Manages physical hardware capture devices: performs discovery probes, 
    compiles config listings, and releases exclusive locks held by conflicting processes.
    """
    def __init__(self, devices_txt_path="./devices.txt"):
        self.devices_txt_path = devices_txt_path

    def release_device_locks(self, video_device, alsa_device):
        """
        Scans /proc to identify and terminate any processes holding exclusive file descriptors
        on the configured V4L2 node or the ALSA card.
        """
        pids_to_kill = set()
        my_pid = os.getpid()
        
        # Audio card index pattern
        card = "*"
        if "hw:" in alsa_device or "dsnoop:" in alsa_device:
            match = re.search(r"(\d+)", alsa_device)
            if match:
                card = match.group(1)
        audio_pattern = f"pcmC{card}D"
        
        # Real video card path
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
                        # Match audio device descriptor (ends in 'c' for capture)
                        if "pcmC" in link and link.endswith("c") and audio_pattern in link:
                            pids_to_kill.add(pid)
                        # Match video device node path
                        elif real_video_path and real_video_path in link:
                            pids_to_kill.add(pid)
                    except Exception:
                        pass
            except Exception:
                pass
                
        if pids_to_kill:
            print(f"[DeviceManager] Found {len(pids_to_kill)} conflicting process(es) holding hardware: {pids_to_kill}")
            for pid in pids_to_kill:
                try:
                    with open(f"/proc/{pid}/comm", "r") as f:
                        comm = f.read().strip()
                    print(f"  -> Releasing device: terminating process {pid} ({comm})...")
                    os.kill(pid, 15)  # SIGTERM
                    time.sleep(0.1)
                    if os.path.exists(f"/proc/{pid}"):
                        os.kill(pid, 9)   # SIGKILL
                except Exception as e:
                    print(f"  -> Failed to terminate process {pid}: {e}", file=sys.stderr)
            time.sleep(0.5)
            print("[DeviceManager] Conflicting device locks successfully released.")
        else:
            print("[DeviceManager] No conflicting hardware locks detected.")

    def get_video_devices(self):
        """Finds all available video capture nodes and card type names."""
        video_devices = []
        by_id_paths = glob.glob("/dev/v4l/by-id/*")
        video_nodes = sorted(glob.glob("/dev/video*"))
        
        for node in video_nodes:
            device_info = {
                "path": node,
                "name": os.path.basename(node),
                "by_id_path": None,
            }
            for by_id in by_id_paths:
                if os.path.exists(by_id) and os.path.samefile(node, by_id):
                    device_info["by_id_path"] = by_id
                    break
                    
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

    def get_audio_devices(self):
        """Finds ALSA capture cards and details rates/channels."""
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
                            "sampling_rates": [44100, 48000],
                            "channels": [1, 2]
                        }
                        audio_devices.append(device_info)
        except Exception:
            if os.path.exists("/proc/asound/cards"):
                try:
                    with open("/proc/asound/cards", "r") as f:
                        content = f.read()
                    cards = re.findall(r"\s*(\d+)\s+\[([^\]]+)\]:\s+(.+)", content)
                    for card_num, short_name, long_name in cards:
                        audio_devices.append({
                            "alsa_device": f"hw:{card_num},0",
                            "card_name": short_name.strip(),
                            "device_name": long_name.strip(),
                            "sampling_rates": [44100, 48000],
                            "channels": [1, 2]
                        })
                except Exception:
                    pass
        return audio_devices

    def discover_and_report(self):
        """Performs full system hardware probe and saves result report to devices.txt."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        v_devices = self.get_video_devices()
        a_devices = self.get_audio_devices()
        
        try:
            with open(self.devices_txt_path, "w") as f:
                f.write("==================================================\n")
                f.write("          DISCOVERED HARDWARE DEVICES             \n")
                f.write("==================================================\n")
                f.write(f"Generated at: {timestamp}\n\n")
                
                f.write("--- Video Devices ---\n")
                if not v_devices:
                    f.write("  No video devices found.\n")
                for idx, vd in enumerate(v_devices, 1):
                    f.write(f"[{idx}] Name: {vd['name']}\n")
                    f.write(f"    Path: {vd['path']}\n")
                    if vd.get('by_id_path'):
                        f.write(f"    By-ID Path: {vd['by_id_path']}\n")
                    f.write("\n")
                    
                f.write("--- Audio Capture Devices (ALSA) ---\n")
                if not a_devices:
                    f.write("  No audio capture devices found.\n")
                for idx, ad in enumerate(a_devices, 1):
                    f.write(f"[{idx}] Card Name: {ad['card_name']}\n")
                    f.write(f"    Device Name: {ad['device_name']}\n")
                    f.write(f"    ALSA Identifier: {ad['alsa_device']}\n")
                    f.write(f"    Supported Rates: {', '.join(map(str, ad['sampling_rates']))}\n")
                    f.write(f"    Supported Channels: {', '.join(map(str, ad['channels']))}\n")
                    f.write("\n")
            print(f"[DeviceManager] Successfully wrote devices list to: {self.devices_txt_path}")
            return {
                "video_devices": v_devices,
                "audio_devices": a_devices
            }
        except Exception as e:
            print(f"[DeviceManager] ERROR saving report to {self.devices_txt_path}: {e}", file=sys.stderr)
            return None
