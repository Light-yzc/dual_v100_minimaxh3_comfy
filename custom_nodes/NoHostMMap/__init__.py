"""Header-only model readers used by the MiniMax H3 low-RAM profile.

This package is copied into ComfyUI by ``scripts/setup_ubuntu.sh``.  The
actual GGUF loader imports :mod:`gguf_reader` from the same custom-node
directory, so no third-party package needs to be modified in-place.
"""

# This is a support package, not a UI node.  ComfyUI scans every directory
# below ``custom_nodes``; an empty mapping prevents a misleading
# "IMPORT FAILED" warning while preserving normal Python imports.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
