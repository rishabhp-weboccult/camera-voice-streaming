#!/usr/bin/env python3
import os
import sys
import wave
import numpy as np

class AudioMatcher:
    """
    Performs real-time audio template matching using frame-by-frame zero-centered Mel-spectrogram
    spectral correlation with local temporal jitter robustness.
    Ensures that all frames contribute equally, completely preventing single high-pitched sounds,
    frequency spikes, or pitch variations of different phrases from triggering false matches.
    Includes an RMS-energy voice activity detection (VAD) gate.
    """
    def __init__(self, target_wav_path="/home/wot-rishabh/Downloads/recording.wav", 
                 threshold=0.50, sample_rate=48000, enabled=True):
        self.target_wav_path = target_wav_path
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.enabled = enabled
        
        # Audio feature configuration
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 40
        
        # Target data
        self.target_audio = None
        self.target_duration = 0.0
        self.target_rms = 0.0
        self.target_features = None
        self.target_loaded = False
        
        if self.enabled:
            self.load_target_template()
            
    def load_target_template(self):
        """Loads the target WAV file and extracts its feature embeddings."""
        if not os.path.exists(self.target_wav_path):
            print(f"[AudioMatcher] WARNING: Target WAV file not found at '{self.target_wav_path}'. Matching will be disabled.", file=sys.stderr)
            self.target_loaded = False
            return
            
        try:
            with wave.open(self.target_wav_path, 'rb') as w:
                n_channels = w.getnchannels()
                sampwidth = w.getsampwidth()
                framerate = w.getframerate()
                n_frames = w.getnframes()
                
                raw_bytes = w.readframes(n_frames)
                
                # Check sample format
                if sampwidth == 2:
                    dtype = np.int16
                elif sampwidth == 1:
                    dtype = np.uint8
                elif sampwidth == 4:
                    dtype = np.int32
                else:
                    raise ValueError(f"Unsupported sample width: {sampwidth}")
                    
                samples = np.frombuffer(raw_bytes, dtype=dtype)
                
                # Convert to mono if multi-channel
                if n_channels > 1:
                    samples = samples.reshape(-1, n_channels).mean(axis=1)
                
                # Convert to float32 and normalize to [-1.0, 1.0]
                if sampwidth == 2:
                    self.target_audio = samples.astype(np.float32) / 32768.0
                elif sampwidth == 4:
                    self.target_audio = samples.astype(np.float32) / 2147483648.0
                else:
                    self.target_audio = (samples.astype(np.float32) - 128.0) / 128.0
                
                # Resample target audio if the rate doesn't match our stream rate
                if framerate != self.sample_rate:
                    print(f"[AudioMatcher] Resampling target WAV from {framerate}Hz to stream rate {self.sample_rate}Hz...")
                    duration = len(self.target_audio) / framerate
                    new_samples = int(duration * self.sample_rate)
                    self.target_audio = np.interp(
                        np.linspace(0, len(self.target_audio), new_samples),
                        np.arange(len(self.target_audio)),
                        self.target_audio
                    ).astype(np.float32)
                
                self.target_duration = len(self.target_audio) / self.sample_rate
                
                # Calculate template energy
                self.target_rms = np.sqrt(np.mean(self.target_audio**2)) + 1e-8
                
                # Extract reference feature spectrogram (with CMS & frame normalization)
                self.target_features = self.extract_mel_spectrogram(self.target_audio)
                self.target_loaded = True
                print(f"[AudioMatcher] Successfully loaded target '{os.path.basename(self.target_wav_path)}':")
                print(f"  Duration    : {self.target_duration:.2f} seconds")
                print(f"  Target RMS  : {self.target_rms:.6f}")
                print(f"  Embed Shape : {self.target_features.shape[0]} frames x {self.target_features.shape[1]} Mel bands")
                
        except Exception as e:
            print(f"[AudioMatcher] ERROR loading target WAV file: {e}", file=sys.stderr)
            self.target_loaded = False

    def extract_mel_spectrogram(self, audio):
        """
        Extracts channel-independent and mean-centered Mel-spectrogram features in pure NumPy.
        Normalizes EACH individual spectral frame to unit L2 norm, guaranteeing that every
        temporal segment contributes equally to the final correlation score.
        """
        # Hanning window
        window = np.hanning(self.n_fft)
        
        # Calculate number of frames
        n_frames = (len(audio) - self.n_fft) // self.hop_length + 1
        if n_frames <= 0:
            return np.zeros((1, self.n_mels), dtype=np.float32)
            
        stft = []
        for i in range(0, n_frames * self.hop_length, self.hop_length):
            frame = audio[i : i + self.n_fft]
            if len(frame) < self.n_fft:
                frame = np.pad(frame, (0, self.n_fft - len(frame)), mode='constant')
            
            # Apply window and FFT
            fft_mag = np.abs(np.fft.rfft(frame * window))
            stft.append(fft_mag)
            
        stft = np.array(stft, dtype=np.float32) # shape: (frames, n_fft // 2 + 1)
        
        # Build Mel filterbank matrix
        f_min = 0
        f_max = self.sample_rate / 2
        
        mel_min = 2595 * np.log10(1 + f_min / 700)
        mel_max = 2595 * np.log10(1 + f_max / 700)
        
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = 700 * (10**(mel_points / 2595) - 1)
        
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        
        filters = np.zeros((self.n_fft // 2 + 1, self.n_mels), dtype=np.float32)
        for m in range(1, self.n_mels + 1):
            d1 = max(1, bin_points[m] - bin_points[m - 1])
            d2 = max(1, bin_points[m + 1] - bin_points[m])
            
            for k in range(bin_points[m - 1], bin_points[m]):
                filters[k, m - 1] = (k - bin_points[m - 1]) / d1
            for k in range(bin_points[m], bin_points[m + 1]):
                filters[k, m - 1] = (bin_points[m + 1] - k) / d2
                
        # Project to Mel scale
        mel_spec = np.dot(stft, filters)
        
        # Logarithmic amplitude scaling
        log_mel = np.log1p(mel_spec)
        
        # 1. Cepstral Mean Subtraction (CMS) over the time axis:
        # Erases static microphone response hums.
        mean_over_time = np.mean(log_mel, axis=0, keepdims=True)
        cms_mel = log_mel - mean_over_time
        
        # 2. Subtract row-wise mean (frame-wise centering):
        # Shifts values into positive and negative space to center correlation at 0.0.
        frame_means = np.mean(cms_mel, axis=1, keepdims=True)
        centered = cms_mel - frame_means
        
        # 3. Frame-wise L2 Normalization:
        # Standardizes every frame to unit length. This enforces strict temporal sequence shape
        # matching, preventing loud high-pitched clicks/vowels from dominating the overall score.
        norms = np.linalg.norm(centered, axis=1, keepdims=True) + 1e-9
        normalized = centered / norms
        
        return normalized

    def match_live_audio(self, live_audio_window):
        """
        Compares live audio window with target template frame-by-frame.
        Utilizes an RMS-energy gate, then slides the target template along the live window.
        At each offset, computes the average frame correlation using a local jitter window
        to absorb slight speech rate changes without losing temporal sequence ordering.
        Returns (max_score, is_match).
        """
        if not self.enabled or not self.target_loaded or self.target_features is None:
            return 0.0, False
            
        # 1. Voice Activity Detection (VAD) Energy Gate:
        # Ignore calculations during silence.
        live_rms = np.sqrt(np.mean(live_audio_window**2))
        if live_rms < 0.0015 or live_rms < 0.03 * self.target_rms:
            return 0.0, False
            
        # 2. Feature Extraction of live sliding window
        live_features = self.extract_mel_spectrogram(live_audio_window)
        
        T_target = self.target_features.shape[0]
        T_live = live_features.shape[0]
        
        if T_live < T_target:
            return 0.0, False
            
        # 3. Sliding-window average frame correlation search with local temporal jitter
        max_similarity = -1.0
        jitter = 1 # allows a local shift of +/- 1 frame (~10ms) for speech rate robustness
        
        step = max(1, (T_live - T_target) // 12)
        for offset in range(0, T_live - T_target + 1, step):
            live_sub = live_features[offset : offset + T_target, :]
            
            # Since live_sub L2 frame norms might have drifted during window CMS,
            # we re-normalize each live slice frame to unit norm to ensure perfect dot-product cosine similarity
            live_sub_centered = live_sub - np.mean(live_sub, axis=1, keepdims=True)
            norms = np.linalg.norm(live_sub_centered, axis=1, keepdims=True) + 1e-9
            live_sub_normalized = live_sub_centered / norms
            
            frame_scores = []
            for t in range(T_target):
                # Local temporal search range to absorb timing jitter
                t_min = max(0, t - jitter)
                t_max = min(T_target - 1, t + jitter)
                
                best_frame_sim = -1.0
                for tj in range(t_min, t_max + 1):
                    # Dot product of unit vectors is the cosine similarity (Pearson Correlation)
                    sim = np.dot(live_sub_normalized[t, :], self.target_features[tj, :])
                    if sim > best_frame_sim:
                        best_frame_sim = sim
                frame_scores.append(best_frame_sim)
                
            # Average frame-wise correlation score across the entire timeline
            mean_score = float(np.mean(frame_scores))
            
            if mean_score > max_similarity:
                max_similarity = mean_score
                
        max_similarity = max(0.0, max_similarity)
        is_match = max_similarity >= self.threshold
        return max_similarity, is_match
