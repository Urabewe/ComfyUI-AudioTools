## README PRODUCED BY CLAUDE! I HAVE NOT GONE OVER IT YET NOR HAD TIME TO MAKE MY OWN.





# ComfyUI-AudioTools

Professional audio processing nodes for ComfyUI, featuring intelligent enhancement and normalization capabilities.

## Overview

ComfyUI-AudioTools provides two essential audio processing nodes:

- **Audio Enhancement (DSRE)** - Advanced multi-band audio enhancement with automatic analysis
- **Audio Normalize (LUFS)** - Intelligent loudness normalization with LUFS-based measurement

These nodes are designed to work seamlessly in ComfyUI workflows, particularly for video-to-video generation and audio post-processing tasks.

---

## Installation

### Requirements
- ComfyUI (latest version recommended)
- Python 3.8+

### Install Dependencies

```bash
pip install soundfile resampy
```

*Note: torch, numpy, scipy, and librosa are already included in standard ComfyUI installations.*

### Manual Installation

1. Navigate to your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes/
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ComfyUI-AudioTools.git
   ```

3. Restart ComfyUI

---

## Audio Enhancement Node

### Features

- **Multi-band harmonic enhancement** - Processes 8 frequency bands independently for optimal results
- **Automatic audio analysis** - Detects and corrects thin, harsh, muddy, or noisy audio
- **De-essing and high-frequency control** - Reduces sibilance and removes unwanted hiss
- **Analog warmth simulation** - Adds tube-style saturation for natural character
- **Stereo width enhancement** - M/S processing for immersive soundstage
- **Dynamic range control** - Gentle upward expansion for more lively audio
- **Spectral noise reduction** - Intelligent removal of background noise
- **Time-based selection** - Process specific segments or full tracks
- **V2V Mode** - Reference-based enhancement for video-to-video workflows

### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| enhancement_mode | manual | manual / auto_enhance | Choose manual settings or automatic analysis |
| enhancement_strength | 0.7 | 0.0 - 1.0 | Overall enhancement intensity |
| harmonic_intensity | 0.6 | 0.0 - 1.0 | Amount of harmonic generation |
| stereo_width | 1.3 | 0.5 - 2.5 | Stereo image width (1.0 = unchanged) |
| dynamic_enhancement | 1.2 | 0.8 - 2.5 | Dynamic range adjustment |
| bass_boost | 1.0 | 0.5 - 2.0 | Bass frequency control |
| presence_boost | 1.0 | 0.5 - 2.0 | Presence/clarity control |
| warmth | 0.5 | 0.0 - 1.0 | Analog-style tube saturation |
| target_sample_rate | keep_original | various | Resample to target rate |
| enable_noise_reduction | false | boolean | Enable spectral noise reduction |
| noise_reduction_level | 5 | 1 - 10 | Noise reduction intensity |
| start_time | 0.0 | 0.0+ | Start time for processing (seconds) |
| end_time | 0.0 | 0.0+ | End time for processing (0 = end of file) |
| apply_to | full_track | 3 options | Processing mode (see below) |

### Apply To Modes

**full_track** - Process the entire audio file
- Use for complete audio enhancement
- Applies settings to all audio

**selection_only** - Process only the specified time range
- Define start_time and end_time
- Only the selection is enhanced

**v2v_mode** - Reference-based enhancement for video-to-video workflows
- Analyzes 3 seconds before the selection as reference
- Enhances selection to match reference quality
- Automatically adjusts parameters based on comparison
- Ideal for maintaining consistency across video transitions

### Basic Usage

**Simple enhancement:**
```
enhancement_mode: manual
enhancement_strength: 0.7
apply_to: full_track
```

**Automatic enhancement:**
```
enhancement_mode: auto_enhance
apply_to: full_track
```

**Video-to-video mode:**
```
start_time: 10.0
end_time: 15.0
apply_to: v2v_mode
```

### Recommended Settings

**For Voice/Podcasts:**
- enhancement_strength: 0.6-0.8
- warmth: 0.6-0.8
- bass_boost: 1.2-1.4
- presence_boost: 0.9-1.1
- enable_noise_reduction: true

**For Music:**
- enhancement_strength: 0.5-0.7
- warmth: 0.3-0.5
- stereo_width: 1.3-1.5
- dynamic_enhancement: 1.3-1.5

**For Thin/Tinny Audio:**
- warmth: 0.7-1.0
- bass_boost: 1.3-1.5
- presence_boost: 0.8-0.9
- enable_noise_reduction: true

---

## Audio Normalize Node

### Features

- **LUFS-based normalization** - Industry-standard loudness measurement
- **Time-based selection** - Normalize specific segments or full tracks
- **Auto-balance mode** - Matches loudness to surrounding audio
- **Safe gain limiting** - Prevents clipping with intelligent safeguards
- **Multiple application modes** - Flexible targeting options

### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| target_lufs | -20.0 | -70.0 - 0.0 | Target loudness in LUFS |
| start_time | 0.0 | 0.0+ | Start time for analysis (seconds) |
| end_time | 0.0 | 0.0+ | End time for analysis (0 = end of file) |
| apply_to | full_track | 3 options | Normalization mode |

### Apply To Modes

**full_track** - Normalize entire audio to target LUFS

**selection_only** - Normalize only the specified time range

**auto_balance** - Match loudness to surrounding audio
- Analyzes 3 seconds before the selection
- Matches selection loudness to reference
- Limited to ±6dB change for safety
- Perfect for dialogue consistency

### Basic Usage

**Standard normalization:**
```
target_lufs: -16.0
apply_to: full_track
```

**Normalize selection:**
```
target_lufs: -16.0
start_time: 5.0
end_time: 10.0
apply_to: selection_only
```

**Auto-balance dialogue:**
```
start_time: 15.0
end_time: 20.0
apply_to: auto_balance
```

### Target LUFS Guidelines

- **-16 LUFS** - Streaming platforms (Spotify, YouTube)
- **-14 LUFS** - Louder streaming content
- **-20 LUFS** - Podcasts and audiobooks
- **-23 LUFS** - Broadcasting standard (EBU R128)

---

## Workflow Tips

### Recommended Order

For best results, process audio in this order:

```
Load Audio → Enhancement → Normalize → Save Audio
```

Enhancement changes dynamics and levels, so normalization should come last to ensure consistent output loudness.

### For Video-to-Video Projects

1. Use Enhancement node with `apply_to: v2v_mode` to maintain consistency
2. Set start_time/end_time to the regenerated segment
3. Follow with Normalize node using `apply_to: auto_balance`
4. This ensures both quality and loudness match the original

### Auto Enhance Mode

When `enhancement_mode` is set to `auto_enhance`, the node automatically:
- Analyzes frequency content (bass, mids, highs)
- Detects audio problems (thin, harsh, muddy, noisy)
- Estimates signal-to-noise ratio
- Sets optimal enhancement parameters
- Enables/adjusts noise reduction as needed

All manual parameters are ignored in auto mode.

---

## Technical Details

### Enhancement Algorithm

The enhancement node uses a multi-stage processing pipeline:

1. De-essing (4-10kHz sibilance reduction)
2. Low-pass filtering (removes >16kHz hiss)
3. Analog warmth (tube-style saturation)
4. Multi-band excitement (8 frequency bands)
5. Psychoacoustic enhancement (perceptually important frequencies)
6. Dynamic range enhancement (upward expansion)
7. Stereo width processing (M/S technique)
8. Final blending and limiting

### Normalization Algorithm

- RMS-based LUFS approximation
- 3-second sliding window for reference analysis
- ±6dB maximum gain change for safety
- Automatic peak limiting to -0.1dBFS

---

## Troubleshooting

**Enhancement makes audio distorted:**
- Lower enhancement_strength to 0.5-0.6
- Reduce harmonic_intensity
- Check input audio isn't already clipping

**No noticeable enhancement:**
- Increase enhancement_strength to 0.8-0.9
- Adjust bass_boost and presence_boost
- Try auto_enhance mode

**Normalization too quiet/loud:**
- Adjust target_lufs (more negative = quieter)
- Verify time range is correct
- Check input audio has actual content

**V2V mode inconsistent:**
- Ensure reference window (3 seconds before selection) has audio
- Check that start_time is at least 3 seconds into the file
- Try adjusting the time range

---

## Credits

**Audio Enhancement Algorithm:** Based on DSRE v2.0 Enhanced Audio Processing Suite

**Original DSRE:** Qu Le Fan

**ComfyUI Integration:** [Your Name]

---

## License

MIT License - See LICENSE file for details

---

## Support

For issues, questions, or feature requests:

**GitHub Issues:** [https://github.com/yourusername/ComfyUI-AudioTools/issues](https://github.com/yourusername/ComfyUI-AudioTools/issues)

---

## Changelog

### v1.0.0
- Initial release
- Audio Enhancement (DSRE) node with auto-enhance and V2V mode
- Audio Normalize (LUFS) node with auto-balance
- Time-based selection support
- Comprehensive voice and music optimization