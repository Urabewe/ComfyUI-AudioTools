# __init__.py
# Place this file in: ComfyUI/custom_nodes/ComfyUI-AudioTools/__init__.py

from .audio_normalize import NODE_CLASS_MAPPINGS as normalize_nodes
from .audio_normalize import NODE_DISPLAY_NAME_MAPPINGS as normalize_display_names

from .audio_enhance import NODE_CLASS_MAPPINGS as enhance_nodes
from .audio_enhance import NODE_DISPLAY_NAME_MAPPINGS as enhance_display_names

# Merge all nodes
NODE_CLASS_MAPPINGS = {
    **normalize_nodes,
    **enhance_nodes,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **normalize_display_names,
    **enhance_display_names,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']