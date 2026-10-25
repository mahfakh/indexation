"""
Descriptor package — provides a unified factory for all descriptor extractors.

Available descriptor types:
  - 'color_rgb' : Per-channel RGB histogram (dims = bins * 3, default 192)
  - 'color_hsv' : Per-channel HSV histogram (dims = 38 with default bins)
  - 'lbp'       : Local Binary Patterns uniform histogram (dims = 26 by default)
  - 'hog'       : Histogram of Oriented Gradients (dims = 1764 at 128×128, 9 orient, 8×8 px/cell, 2×2 blk)
"""

from cbir.descriptors.color import ColorDescriptor
from cbir.descriptors.lbp import LBPDescriptor
from cbir.descriptors.hog import HOGDescriptor

_color = ColorDescriptor()
_lbp = LBPDescriptor()
_hog = HOGDescriptor()

DESCRIPTOR_TYPES = ["color_rgb", "color_hsv", "lbp", "hog"]


def get_extractor(descriptor_type: str):
    """Return a callable ``fn(image_path) -> np.ndarray`` for the given descriptor type.

    Args:
        descriptor_type: One of 'color_rgb', 'color_hsv', 'lbp', 'hog'.

    Returns:
        A zero-argument-free callable that takes an image path and returns a feature vector.

    Raises:
        ValueError: If the descriptor type is unknown.
    """
    if descriptor_type == "color_rgb":
        return lambda path: _color.extract(path, color_space="rgb")
    elif descriptor_type == "color_hsv":
        return lambda path: _color.extract(path, color_space="hsv")
    elif descriptor_type == "lbp":
        return lambda path: _lbp.extract(path)
    elif descriptor_type == "hog":
        return lambda path: _hog.extract(path)
    else:
        raise ValueError(f"Unknown descriptor type: '{descriptor_type}'. "
                         f"Available: {DESCRIPTOR_TYPES}")
