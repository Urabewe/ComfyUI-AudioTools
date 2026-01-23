"""ComfyUI Audio Enhancement Node
Based on DSRE v2.0 Enhanced Audio Processing Suite

Installation:
1. Place this file in: ComfyUI/custom_nodes/ComfyUI-AudioTools/audio_enhance.py
2. Install dependencies: pip install soundfile resampy
3. Restart ComfyUI
"""

import os
import numpy as np
from scipy import signal
import librosa
import resampy
import soundfile as sf
import tempfile
import torch
import folder_paths

# ======== CORE AUDIO PROCESSING FUNCTIONS ========

def spectral_noise_reduction(x: np.ndarray, sr: int, noise_level: int = 5) -> np.ndarray:
    """Spectral noise reduction using spectral gating technique"""
    if x is None or x.size == 0:
        return x
    
    reduction_strength = 0.1 + (noise_level - 1) * 0.08
    enhanced = np.zeros_like(x)
    
    for ch in range(x.shape[0]):
        signal_ch = x[ch]
        fft = np.fft.fft(signal_ch)
        freqs = np.fft.fftfreq(len(signal_ch), 1/sr)
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        
        sorted_magnitude = np.sort(magnitude)
        noise_floor = np.mean(sorted_magnitude[:len(sorted_magnitude)//10])
        threshold = noise_floor * (1 + reduction_strength)
        
        gate = np.ones_like(magnitude)
        gate[magnitude < threshold] = reduction_strength
        gated_magnitude = magnitude * gate
        gated_fft = gated_magnitude * np.exp(1j * phase)
        enhanced[ch] = np.real(np.fft.ifft(gated_fft))
        
        if noise_level > 5:
            cutoff_freq = sr // 4
            cutoff_bin = int(cutoff_freq * len(signal_ch) / sr)
            rolloff = np.ones(len(fft))
            rolloff[cutoff_bin:] *= np.exp(-np.arange(len(fft) - cutoff_bin) * 0.01 * (noise_level - 5))
            rolloff[-cutoff_bin:] = rolloff[cutoff_bin:cutoff_bin*2][::-1]
            fft_rolled = fft * rolloff
            enhanced[ch] = np.real(np.fft.ifft(fft_rolled))
    
    return enhanced

def generate_harmonics(signal_band, fundamental_freq, sr, num_harmonics=5, harmonic_strength=0.3):
    """Generate harmonic content for a frequency band"""
    if len(signal_band) == 0 or np.any(np.isnan(signal_band)):
        return signal_band
    
    enhanced = signal_band.copy()
    
    for h in range(2, num_harmonics + 2):
        harmonic_freq = fundamental_freq * h
        if harmonic_freq < sr / 2:
            phase_increment = 2 * np.pi * harmonic_freq / sr
            if np.isnan(phase_increment) or np.isinf(phase_increment):
                continue
            
            harmonic_oscillator = np.sin(phase_increment * np.arange(len(signal_band)))
            if np.any(np.isnan(harmonic_oscillator)):
                continue
            
            harmonic_content = signal_band * harmonic_oscillator * (harmonic_strength / h)
            if np.any(np.isnan(harmonic_content)):
                continue
            
            enhanced += harmonic_content
    
    if np.any(np.isnan(enhanced)):
        return signal_band
    
    return enhanced

def multiband_exciter(x, sr, bass_boost=1.0, presence_boost=1.0):
    """Multi-band harmonic exciter with adjustable frequency response"""
    if x.ndim == 1:
        x = x[np.newaxis, :]
    
    enhanced = np.zeros_like(x)
    nyquist = sr // 2
    
    band_definitions = [
        {"name": "Sub Bass", "low": 20, "high": 80, "gain": 1.5 * bass_boost, "harmonics": 3, "strength": 0.3},
        {"name": "Bass", "low": 80, "high": 250, "gain": 1.6 * bass_boost, "harmonics": 3, "strength": 0.35},
        {"name": "Low Mid", "low": 250, "high": 500, "gain": 1.4, "harmonics": 2, "strength": 0.25},
        {"name": "Mid", "low": 500, "high": 2000, "gain": 1.3, "harmonics": 2, "strength": 0.2},
        {"name": "High Mid", "low": 2000, "high": 5000, "gain": 1.2 * presence_boost, "harmonics": 1, "strength": 0.15},
        {"name": "Upper Mid", "low": 5000, "high": 8000, "gain": 0.8 * presence_boost, "harmonics": 1, "strength": 0.1},
        {"name": "Presence", "low": 8000, "high": 12000, "gain": 0.6, "harmonics": 0, "strength": 0.0},
        {"name": "Air", "low": 12000, "high": min(20000, nyquist - 1000), "gain": 0.5, "harmonics": 0, "strength": 0.0}
    ]
    
    bands = [band for band in band_definitions 
             if band["low"] < nyquist and band["high"] < nyquist and band["high"] > band["low"]]
    
    if not bands:
        return x
    
    for ch in range(x.shape[0]):
        channel_enhanced = x[ch].copy()
        
        for band in bands:
            if band["low"] >= sr // 2:
                continue
            
            low_norm = band["low"] / (sr / 2)
            high_norm = min(band["high"] / (sr / 2), 0.99)
            
            if low_norm >= high_norm or low_norm <= 0 or high_norm >= 1.0:
                continue
            
            min_separation = 0.0001 if low_norm < 0.01 else (0.001 if low_norm < 0.1 else 0.01)
            if high_norm - low_norm < min_separation:
                continue
            
            try:
                filter_order = min(4, max(2, int(4 * (high_norm - low_norm))))
                b, a = signal.butter(filter_order, [low_norm, high_norm], btype='band')
                
                if np.any(np.isnan(b)) or np.any(np.isnan(a)):
                    continue
                
                band_signal = signal.filtfilt(b, a, x[ch])
                if np.any(np.isnan(band_signal)):
                    continue
                
                center_freq = (band["low"] + band["high"]) / 2
                harmonics_added = generate_harmonics(
                    band_signal, center_freq, sr, 
                    band["harmonics"], band["strength"] * 0.8
                )
                
                if np.any(np.isnan(harmonics_added)):
                    continue
                
                saturated = np.tanh(harmonics_added * 1.2) * 0.85
                if np.any(np.isnan(saturated)):
                    continue
                
                band_enhanced = saturated * band["gain"]
                if np.any(np.isnan(band_enhanced)):
                    continue
                
                channel_enhanced = channel_enhanced + band_enhanced * 0.25
                
            except Exception:
                continue
        
        enhanced[ch] = channel_enhanced
    
    return enhanced

def psychoacoustic_enhancer(x, sr):
    """Psychoacoustic enhancement targeting human hearing sensitivity - voice optimized"""
    if x.ndim == 1:
        x = x[np.newaxis, :]
    
    enhanced = np.zeros_like(x)
    
    critical_bands = [
        {"freq": 200, "boost": 1.5, "q": 1.0},
        {"freq": 500, "boost": 1.8, "q": 1.5},
        {"freq": 1000, "boost": 2.0, "q": 1.5},
        {"freq": 2500, "boost": 1.6, "q": 1.2},
        {"freq": 4000, "boost": 1.2, "q": 0.8},
    ]
    
    for ch in range(x.shape[0]):
        channel_enhanced = x[ch].copy()
        
        for band in critical_bands:
            if band["freq"] >= sr // 2:
                continue
            
            try:
                freq_norm = band["freq"] / (sr / 2)
                if freq_norm >= 0.99:
                    continue
                
                w = 2 * np.pi * band["freq"] / sr
                cosw = np.cos(w)
                sinw = np.sin(w)
                alpha = sinw / (2 * band["q"])
                A = 10**(band["boost"]/40)
                
                b0 = 1 + alpha * A
                b1 = -2 * cosw
                b2 = 1 - alpha * A
                a0 = 1 + alpha / A
                a1 = -2 * cosw
                a2 = 1 - alpha / A
                
                b = np.array([b0, b1, b2]) / a0
                a = np.array([1, a1/a0, a2/a0])
                
                filtered = signal.lfilter(b, a, x[ch])
                blend_factor = 0.4
                channel_enhanced = channel_enhanced * (1 - blend_factor) + filtered * blend_factor
                
            except Exception:
                continue
        
        enhanced[ch] = channel_enhanced
    
    return enhanced

def stereo_width_enhancer(x, width_factor=1.4):
    """Enhance stereo width using M/S processing"""
    if x.shape[0] != 2:
        return x
    
    left = x[0]
    right = x[1]
    
    mid = (left + right) / 2
    side = (left - right) / 2
    side_enhanced = side * width_factor
    
    left_enhanced = mid + side_enhanced
    right_enhanced = mid - side_enhanced
    
    return np.array([left_enhanced, right_enhanced])

def dynamic_range_enhancer(x, ratio=1.3, attack_ms=5, release_ms=50, sr=44100):
    """Gentle upward expansion to increase dynamic range"""
    attack_samples = int(attack_ms * sr / 1000)
    release_samples = int(release_ms * sr / 1000)
    enhanced = np.zeros_like(x)
    
    for ch in range(x.shape[0]):
        signal_ch = x[ch]
        envelope = np.abs(signal_ch)
        
        if len(envelope) > 0:
            smoothed_env = np.zeros_like(envelope)
            current_env = envelope[0]
            
            for i in range(len(envelope)):
                if envelope[i] > current_env:
                    current_env += (envelope[i] - current_env) / attack_samples
                else:
                    current_env -= (current_env - envelope[i]) / release_samples
                smoothed_env[i] = current_env
            
            threshold = 0.1
            gain = np.ones_like(smoothed_env)
            above_threshold = smoothed_env > threshold
            gain[above_threshold] = (smoothed_env[above_threshold] / threshold) ** (ratio - 1)
            gain = np.clip(gain, 1.0, 3.0)
            enhanced[ch] = signal_ch * gain
    
    return enhanced

def de_esser(x, sr, threshold_db=-20, reduction_db=6):
    """Reduce sibilance and harsh high frequencies in vocal range"""
    if x.ndim == 1:
        x = x[np.newaxis, :]
    
    de_essed = np.zeros_like(x)
    
    sibilance_low = 4000
    sibilance_high = 10000
    
    for ch in range(x.shape[0]):
        signal_ch = x[ch]
        
        low_norm = sibilance_low / (sr / 2)
        high_norm = min(sibilance_high / (sr / 2), 0.99)
        
        if low_norm < high_norm and high_norm < 1.0:
            try:
                b, a = signal.butter(4, [low_norm, high_norm], btype='band')
                sibilance_band = signal.filtfilt(b, a, signal_ch)
                
                threshold = 10 ** (threshold_db / 20)
                envelope = np.abs(sibilance_band)
                
                window = int(sr * 0.01)
                if window > 0:
                    envelope = np.convolve(envelope, np.ones(window)/window, mode='same')
                
                reduction_factor = 10 ** (-reduction_db / 20)
                gain = np.ones_like(envelope)
                mask = envelope > threshold
                gain[mask] = reduction_factor + (1 - reduction_factor) * (threshold / envelope[mask])
                
                reduced_sibilance = sibilance_band * gain
                de_essed[ch] = signal_ch - sibilance_band + reduced_sibilance
                
            except Exception:
                de_essed[ch] = signal_ch
        else:
            de_essed[ch] = signal_ch
    
    return de_essed

def low_pass_filter(x, sr, cutoff_freq=16000):
    """Apply gentle low-pass filter to remove extreme high frequency hiss"""
    if x.ndim == 1:
        x = x[np.newaxis, :]
    
    filtered = np.zeros_like(x)
    nyquist = sr // 2
    
    if cutoff_freq >= nyquist:
        return x
    
    for ch in range(x.shape[0]):
        try:
            cutoff_norm = cutoff_freq / nyquist
            if cutoff_norm < 0.99:
                b, a = signal.butter(2, cutoff_norm, btype='low')
                filtered[ch] = signal.filtfilt(b, a, x[ch])
            else:
                filtered[ch] = x[ch]
        except Exception:
            filtered[ch] = x[ch]
    
    return filtered

def analog_warmth(x, amount=0.5):
    """Add analog-style warmth with gentle saturation and frequency shaping"""
    if amount <= 0:
        return x
    
    warmed = np.zeros_like(x)
    
    for ch in range(x.shape[0]):
        signal_ch = x[ch]
        
        drive = 1 + (amount * 0.5)
        saturated = np.tanh(signal_ch * drive) / drive
        
        harmonic2 = np.tanh(signal_ch * 2) * 0.15 * amount
        harmonic4 = np.tanh(signal_ch * 4) * 0.08 * amount
        
        warmed[ch] = saturated + harmonic2 + harmonic4
    
    return warmed

def enhanced_audio_algorithm(
    x: np.ndarray,
    sr: int,
    enhancement_strength: float = 0.7,
    harmonic_intensity: float = 0.6,
    stereo_width: float = 1.3,
    dynamic_enhancement: float = 1.2,
    bass_boost: float = 1.0,
    presence_boost: float = 1.0,
    warmth: float = 0.5,
) -> np.ndarray:
    """Complete enhanced audio processing algorithm"""
    
    if x is None or x.size == 0:
        raise ValueError("Input audio data is empty")
    
    # Check for extremely silent audio (much more lenient threshold)
    max_amplitude = np.max(np.abs(x))
    if max_amplitude < 1e-12:
        # Audio is essentially silent, just return it as-is
        return x
    
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("Input audio contains invalid values")
    
    de_essed = de_esser(x, sr, threshold_db=-20, reduction_db=8)
    if np.any(np.isnan(de_essed)):
        de_essed = x
    
    lp_filtered = low_pass_filter(de_essed, sr, cutoff_freq=16000)
    if np.any(np.isnan(lp_filtered)):
        lp_filtered = de_essed
    
    if warmth > 0:
        warmed = analog_warmth(lp_filtered, warmth)
        if np.any(np.isnan(warmed)):
            warmed = lp_filtered
    else:
        warmed = lp_filtered
    
    enhanced = multiband_exciter(warmed, sr, bass_boost, presence_boost)
    
    if np.any(np.isnan(enhanced)):
        enhanced = warmed.copy()
        for ch in range(x.shape[0]):
            signal_ch = warmed[ch]
            enhanced_ch = signal_ch.copy()
            for harmonic in [2, 3]:
                if harmonic * 500 < sr // 2:
                    phase = 2 * np.pi * harmonic * 500 / sr * np.arange(len(signal_ch))
                    harmonic_content = signal_ch * np.sin(phase) * 0.1
                    enhanced_ch += harmonic_content
            enhanced_ch = np.tanh(enhanced_ch * 1.2) * 0.9
            enhanced[ch] = signal_ch * 0.7 + enhanced_ch * 0.3
    
    psycho_enhanced = psychoacoustic_enhancer(enhanced, sr)
    if np.any(np.isnan(psycho_enhanced)):
        psycho_enhanced = enhanced
    
    dynamic_enhanced = dynamic_range_enhancer(psycho_enhanced, dynamic_enhancement, sr=sr)
    if np.any(np.isnan(dynamic_enhanced)):
        dynamic_enhanced = psycho_enhanced
    
    if x.shape[0] == 2 and stereo_width != 1.0:
        stereo_enhanced = stereo_width_enhancer(dynamic_enhanced, stereo_width)
    else:
        stereo_enhanced = dynamic_enhanced
    
    if np.max(np.abs(stereo_enhanced)) < 1e-10:
        final = x.copy()
    else:
        if enhancement_strength == 0:
            final = x.copy()
        else:
            blend_factor = min(enhancement_strength * 0.8, 0.7)
            final = x * (1 - blend_factor) + stereo_enhanced * blend_factor
    
    peak = np.max(np.abs(final))
    if peak > 0.95:
        final = final * (0.95 / peak)
    
    if np.max(np.abs(final)) < 1e-10:
        final = x.copy()
    
    return final


# ======== COMFYUI NODE DEFINITIONS ========

class AudioEnhancementNode:
    """
    Enhanced Audio Processing Node for ComfyUI
    Applies multi-band harmonic excitement, psychoacoustic enhancement,
    and dynamic range processing to audio files.
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enhancement_mode": (["manual", "auto_enhance"], {
                    "default": "manual",
                    "tooltip": "Manual: use your settings | Auto: automatically analyze and enhance based on audio characteristics"
                }),
                "enhancement_strength": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Overall enhancement intensity. 0.0 = bypass, 0.7 = balanced, 1.0 = maximum enhancement"
                }),
                "harmonic_intensity": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Controls harmonic generation for richer sound. Higher values add more harmonic content and richness"
                }),
                "stereo_width": ("FLOAT", {
                    "default": 1.3,
                    "min": 0.5,
                    "max": 2.5,
                    "step": 0.1,
                    "tooltip": "Stereo image width. 1.0 = unchanged, <1.0 = narrower (more mono), >1.0 = wider soundstage. Only affects stereo audio"
                }),
                "dynamic_enhancement": ("FLOAT", {
                    "default": 1.2,
                    "min": 0.8,
                    "max": 2.5,
                    "step": 0.1,
                    "tooltip": "Dynamic range adjustment. 1.0 = unchanged, <1.0 = compressed, >1.0 = expanded (more dynamic)"
                }),
                "bass_boost": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Bass frequency control (20-250Hz). 1.0 = neutral, <1.0 = reduce bass, >1.0 = boost bass"
                }),
                "presence_boost": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Presence/clarity control (2.5-8kHz). 1.0 = neutral, <1.0 = reduce presence, >1.0 = boost clarity and vocal definition"
                }),
                "warmth": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Analog-style warmth with tube saturation. 0.0 = off, 0.5 = subtle warmth, 1.0 = maximum analog character"
                }),
                "target_sample_rate": ([
                    "keep_original",
                    "44100",
                    "48000",
                    "96000",
                    "192000"
                ], {
                    "default": "keep_original",
                    "tooltip": "Resample audio to target sample rate. Higher rates preserve more frequency information but increase file size"
                }),
                "enable_noise_reduction": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable spectral noise reduction to remove hiss and background noise"
                }),
                "noise_reduction_level": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Noise reduction intensity. 1 = gentle, 5 = balanced, 10 = aggressive (may affect audio quality)"
                }),
                "start_time": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.01,
                    "tooltip": "Start time in seconds for processing region (0 = start of file)"
                }),
                "end_time": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.01,
                    "tooltip": "End time in seconds for processing region (0 = end of file)"
                }),
                "apply_to": (["full_track", "selection_only", "auto_enhance"], {
                    "default": "full_track",
                    "tooltip": "Full track: enhance entire audio | Selection only: enhance only the time range | Auto enhance: analyze reference before selection and enhance to match"
                })
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "enhance_audio"
    CATEGORY = "audio/processing"
    
    WINDOW_SEC = 3.0  # Reference window for auto_enhance mode
    
    def analyze_audio_characteristics(self, audio_np, sr):
        """
        Analyze audio to determine optimal enhancement parameters.
        Returns dict of recommended settings.
        """
        rms = np.sqrt(np.mean(audio_np ** 2))
        
        fft = np.fft.fft(audio_np[0] if audio_np.ndim > 1 else audio_np)
        magnitude = np.abs(fft[:len(fft)//2])
        freqs = np.fft.fftfreq(len(fft), 1/sr)[:len(fft)//2]
        
        bass_mask = (freqs >= 20) & (freqs < 250)
        mid_mask = (freqs >= 250) & (freqs < 2000)
        high_mask = (freqs >= 2000) & (freqs < 8000)
        air_mask = (freqs >= 8000) & (freqs < 16000)
        
        bass_energy = np.mean(magnitude[bass_mask]) if np.any(bass_mask) else 0
        mid_energy = np.mean(magnitude[mid_mask]) if np.any(mid_mask) else 0
        high_energy = np.mean(magnitude[high_mask]) if np.any(high_mask) else 0
        air_energy = np.mean(magnitude[air_mask]) if np.any(air_mask) else 0
        
        total_energy = bass_energy + mid_energy + high_energy + air_energy
        
        if total_energy > 0:
            bass_ratio = bass_energy / total_energy
            mid_ratio = mid_energy / total_energy
            high_ratio = high_energy / total_energy
            air_ratio = air_energy / total_energy
        else:
            return {
                "enhancement_strength": 0.0,
                "bass_boost": 1.0,
                "presence_boost": 1.0,
                "warmth": 0.5,
                "enable_noise_reduction": False,
                "noise_reduction_level": 5
            }
        
        is_bass_light = bass_ratio < 0.15
        is_harsh = high_ratio > 0.35
        is_muddy = bass_ratio > 0.4
        is_dull = high_ratio < 0.15
        
        sorted_mag = np.sort(magnitude)
        noise_floor = np.mean(sorted_mag[:len(sorted_mag)//20])
        signal_floor = np.mean(sorted_mag[len(sorted_mag)//2:])
        snr_estimate = signal_floor / (noise_floor + 1e-10)
        
        has_noise = snr_estimate < 100
        
        recommendations = {}
        
        if rms < 0.05:
            recommendations["enhancement_strength"] = 0.8
        elif rms < 0.15:
            recommendations["enhancement_strength"] = 0.7
        else:
            recommendations["enhancement_strength"] = 0.6
        
        if is_bass_light:
            recommendations["bass_boost"] = 1.4
        elif is_muddy:
            recommendations["bass_boost"] = 0.8
        else:
            recommendations["bass_boost"] = 1.1
        
        if is_harsh:
            recommendations["presence_boost"] = 0.8
        elif is_dull:
            recommendations["presence_boost"] = 1.3
        else:
            recommendations["presence_boost"] = 1.0
        
        if is_bass_light or is_harsh:
            recommendations["warmth"] = 0.7
        else:
            recommendations["warmth"] = 0.5
        
        if has_noise:
            recommendations["enable_noise_reduction"] = True
            if snr_estimate < 20:
                recommendations["noise_reduction_level"] = 8
            elif snr_estimate < 50:
                recommendations["noise_reduction_level"] = 6
            else:
                recommendations["noise_reduction_level"] = 4
        else:
            recommendations["enable_noise_reduction"] = False
            recommendations["noise_reduction_level"] = 5
        
        return recommendations
    
    def enhance_audio(self, audio, enhancement_mode, enhancement_strength, harmonic_intensity, 
                     stereo_width, dynamic_enhancement, bass_boost, presence_boost, warmth,
                     target_sample_rate, enable_noise_reduction, noise_reduction_level,
                     start_time, end_time, apply_to):
        
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        enhanced_batch = []
        
        for batch_idx in range(waveform.shape[0]):
            audio_data = waveform[batch_idx].cpu().numpy()
            
            num_samples = audio_data.shape[-1]
            duration = num_samples / sample_rate
            
            if end_time <= 0 or end_time > duration:
                end_time = duration
            if start_time >= end_time:
                start_time = 0.0
            
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # -------------------------
            # AUTO ENHANCE MODE (like auto_balance)
            # -------------------------
            if apply_to == "auto_enhance":
                # Get reference window before selection
                window_samples = int(self.WINDOW_SEC * sample_rate)
                ref_start = max(0, start_sample - window_samples)
                reference = audio_data[..., ref_start:start_sample]
                selection = audio_data[..., start_sample:end_sample]
                
                # Analyze reference audio to determine baseline characteristics
                if reference.size > 0 and np.max(np.abs(reference)) > 1e-6:
                    ref_params = self.analyze_audio_characteristics(reference, sample_rate)
                else:
                    ref_params = None
                
                # Analyze selection audio
                if selection.size > 0 and np.max(np.abs(selection)) > 1e-6:
                    sel_params = self.analyze_audio_characteristics(selection, sample_rate)
                else:
                    sel_params = None
                
                # Determine enhancement needed based on comparison
                if sel_params is not None:
                    if ref_params is not None:
                        # We have both reference and selection - enhance selection to match/improve reference
                        # Use selection's auto params but moderate them based on reference
                        enhancement_strength = sel_params["enhancement_strength"]
                        bass_boost = sel_params["bass_boost"]
                        presence_boost = sel_params["presence_boost"]
                        warmth = sel_params["warmth"]
                        enable_noise_reduction = sel_params["enable_noise_reduction"]
                        noise_reduction_level = sel_params["noise_reduction_level"]
                        
                        # If selection is noisier than reference, boost noise reduction
                        if sel_params["enable_noise_reduction"] and not ref_params["enable_noise_reduction"]:
                            noise_reduction_level = min(noise_reduction_level + 2, 10)
                    else:
                        # No reference, use selection analysis only
                        enhancement_strength = sel_params["enhancement_strength"]
                        bass_boost = sel_params["bass_boost"]
                        presence_boost = sel_params["presence_boost"]
                        warmth = sel_params["warmth"]
                        enable_noise_reduction = sel_params["enable_noise_reduction"]
                        noise_reduction_level = sel_params["noise_reduction_level"]
                else:
                    # Selection is silent/invalid - no enhancement
                    enhancement_strength = 0.0
                
                process_audio = selection
                
            # -------------------------
            # SELECTION ONLY MODE
            # -------------------------
            elif apply_to == "selection_only":
                selection = audio_data[..., start_sample:end_sample]
                process_audio = selection
                
                # Use manual mode or auto mode analysis
                if enhancement_mode == "auto_enhance":
                    auto_params = self.analyze_audio_characteristics(process_audio, sample_rate)
                    enhancement_strength = auto_params["enhancement_strength"]
                    bass_boost = auto_params["bass_boost"]
                    presence_boost = auto_params["presence_boost"]
                    warmth = auto_params["warmth"]
                    enable_noise_reduction = auto_params["enable_noise_reduction"]
                    noise_reduction_level = auto_params["noise_reduction_level"]
            
            # -------------------------
            # FULL TRACK MODE
            # -------------------------
            else:  # full_track
                process_audio = audio_data
                
                # Use manual mode or auto mode analysis
                if enhancement_mode == "auto_enhance":
                    auto_params = self.analyze_audio_characteristics(process_audio, sample_rate)
                    enhancement_strength = auto_params["enhancement_strength"]
                    bass_boost = auto_params["bass_boost"]
                    presence_boost = auto_params["presence_boost"]
                    warmth = auto_params["warmth"]
                    enable_noise_reduction = auto_params["enable_noise_reduction"]
                    noise_reduction_level = auto_params["noise_reduction_level"]
            
            # Resample if needed
            if target_sample_rate != "keep_original":
                new_sr = int(target_sample_rate)
                if sample_rate != new_sr:
                    process_audio = resampy.resample(process_audio, sample_rate, new_sr, filter='kaiser_fast')
                    current_sr = new_sr
                else:
                    current_sr = sample_rate
            else:
                current_sr = sample_rate
            
            # Apply noise reduction if enabled
            if enable_noise_reduction:
                process_audio = spectral_noise_reduction(
                    process_audio, 
                    current_sr, 
                    noise_level=noise_reduction_level
                )
            
            # Apply enhancement algorithm
            enhanced_data = enhanced_audio_algorithm(
                process_audio,
                current_sr,
                enhancement_strength=enhancement_strength,
                harmonic_intensity=harmonic_intensity,
                stereo_width=stereo_width,
                dynamic_enhancement=dynamic_enhancement,
                bass_boost=bass_boost,
                presence_boost=presence_boost,
                warmth=warmth
            )
            
            # Handle different apply modes
            if apply_to in ["selection_only", "auto_enhance"]:
                output_data = audio_data.copy()
                
                if target_sample_rate != "keep_original" and sample_rate != current_sr:
                    output_data = resampy.resample(output_data, sample_rate, current_sr, filter='kaiser_fast')
                    start_sample_new = int(start_time * current_sr)
                    end_sample_new = int(end_time * current_sr)
                    output_data[..., start_sample_new:end_sample_new] = enhanced_data
                else:
                    output_data[..., start_sample:end_sample] = enhanced_data
                
                final_sr = current_sr if target_sample_rate != "keep_original" else sample_rate
            else:  # full_track
                output_data = enhanced_data
                final_sr = current_sr if target_sample_rate != "keep_original" else sample_rate
            
            enhanced_tensor = torch.from_numpy(output_data).float()
            enhanced_batch.append(enhanced_tensor)
        
        enhanced_waveform = torch.stack(enhanced_batch)
        
        return ({
            "waveform": enhanced_waveform,
            "sample_rate": final_sr
        },)


# ======== NODE REGISTRATION ========

NODE_CLASS_MAPPINGS = {
    "AudioEnhancementNode": AudioEnhancementNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioEnhancementNode": "Audio Enhancement (DSRE)",
} 
                