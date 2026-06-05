#!/usr/bin/env python3
import subprocess
import time


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
                stderr=subprocess.PIPE,
                text=True
            )
            time.sleep(0.5)
            if self.process.poll() is not None:
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
