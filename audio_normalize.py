# audio_normalize.py
# Place this file in: ComfyUI/custom_nodes/ComfyUI-AudioNormalize/audio_normalize.py

import numpy as np
import torch

class AudioNormalizeLUFS:
    """
    A ComfyUI node for normalizing audio to a target LUFS level with safe fallbacks.
    If no valid source audio is available for analysis, normalization defaults to
    user-specified target behavior without extreme gain.
    """
    
    MIN_REF_LUFS = -30.0
    MAX_DB_CHANGE = 6.0
    WINDOW_SEC = 3.0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_lufs": ("FLOAT", {
                    "default": -20.0,
                    "min": -70.0,
                    "max": 0.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "start_time": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "end_time": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "apply_to": (["full_track", "selection_only", "auto_balance"], {
                    "default": "full_track"
                })
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "normalize_audio"
    CATEGORY = "audio/processing"

    # -------------------------
    # Utility helpers
    # -------------------------

    def is_valid_audio(self, audio_np):
        """
        Returns False if audio is empty or effectively silent.
        """
        if audio_np.size == 0:
            return False
        if np.max(np.abs(audio_np)) < 1e-6:
            return False
        return True

    def calculate_lufs(self, audio_data, sample_rate):
        """
        Calculate approximate integrated LUFS using RMS.
        Returns None for silent or invalid audio.
        """
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.cpu().numpy()

        if audio_data.size == 0:
            return None

        rms = np.sqrt(np.mean(audio_data ** 2))

        if rms < 1e-10:
            return None

        # Approximate LUFS (not true K-weighted LUFS)
        lufs = 20 * np.log10(rms) - 0.691
        return lufs

    # -------------------------
    # Main processing
    # -------------------------

    def normalize_audio(self, audio, target_lufs, start_time, end_time, apply_to):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        audio_np = waveform.cpu().numpy()

        num_samples = audio_np.shape[-1]
        duration = num_samples / sample_rate

        # Clamp time range
        if end_time <= 0 or end_time > duration:
            end_time = duration
        if start_time >= end_time:
            start_time = 0.0

        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)

        audio_normalized = audio_np.copy()
        selection = audio_np[..., start_sample:end_sample]

        # -------------------------
        # AUTO BALANCE MODE
        # -------------------------
        if apply_to == "auto_balance":
            # Use sliding window before selection
            window_samples = int(self.WINDOW_SEC * sample_rate)
            ref_start = max(0, start_sample - window_samples)
            reference = audio_np[..., ref_start:start_sample]

            # Calculate LUFS
            ref_lufs = self.calculate_lufs(reference, sample_rate) if self.is_valid_audio(reference) else None
            sel_lufs = self.calculate_lufs(selection, sample_rate) if self.is_valid_audio(selection) else None

            if sel_lufs is None:
                # Nothing usable → no change
                gain_linear = 1.0

            elif ref_lufs is not None:
                # Reference-based matching
                ref_lufs = max(ref_lufs, self.MIN_REF_LUFS)
                lufs_diff = ref_lufs - sel_lufs

                lufs_diff = np.clip(
                    lufs_diff,
                    -self.MAX_DB_CHANGE,
                    self.MAX_DB_CHANGE
                )

                gain_linear = 10 ** (lufs_diff / 20)

            else:
                # Fallback: normalize selection to target LUFS
                lufs_diff = target_lufs - sel_lufs

                lufs_diff = np.clip(
                    lufs_diff,
                    -self.MAX_DB_CHANGE,
                    self.MAX_DB_CHANGE
                )

                gain_linear = 10 ** (lufs_diff / 20)

            audio_normalized[..., start_sample:end_sample] *= gain_linear


        # -------------------------
        # NORMAL NORMALIZATION
        # -------------------------
        else:
            if self.is_valid_audio(selection):
                current_lufs = self.calculate_lufs(selection, sample_rate)
            else:
                current_lufs = None

            # Fallback: no valid reference audio
            if current_lufs is None:
                gain_linear = 1.0
            else:
                gain_linear = 10 ** ((target_lufs - current_lufs) / 20)

            if apply_to == "full_track":
                audio_normalized *= gain_linear
            else:  # selection_only
                audio_normalized[..., start_sample:end_sample] *= gain_linear

        # -------------------------
        # Anti-clipping safeguard
        # -------------------------
        max_val = np.max(np.abs(audio_normalized))
        if max_val > 1.0:
            audio_normalized = audio_normalized / max_val * 0.99

        normalized_waveform = torch.from_numpy(audio_normalized).to(waveform.device)

        return ({
            "waveform": normalized_waveform,
            "sample_rate": sample_rate
        },)


# -------------------------
# Node registration
# -------------------------

NODE_CLASS_MAPPINGS = {
    "AudioNormalizeLUFS": AudioNormalizeLUFS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioNormalizeLUFS": "Audio Normalize (LUFS)"
}
