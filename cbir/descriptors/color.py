"""
Color descriptor extraction for CBIR.

Computes per-channel color histograms (concatenated) for RGB and HSV color spaces.
Uses per-channel approach (bins * num_channels dimensions) rather than joint histograms
to keep feature dimensionality practical for VP-Tree indexing.

Typical dimensions:
  - RGB with 64 bins: 64*3 = 192
  - HSV with (18, 16, 4) bins: 18+16+4 = 38
"""

import cv2
import numpy as np
from typing import Tuple, Union, Optional
import logging

logger = logging.getLogger(__name__)


class ColorDescriptor:
    """Extracts color histogram descriptors from images."""

    def _load_and_resize(self, image_path: str, size: Tuple[int, int] = (256, 256)) -> Optional[np.ndarray]:
        """Load image and resize to consistent dimensions.

        Args:
            image_path: Path to the image file.
            size: Target (width, height) for resizing.

        Returns:
            BGR image as numpy array, or None if loading failed.
        """
        image = cv2.imread(image_path)
        if image is None:
            return None
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.resize(image, size)
        return image

    def extract_rgb_histogram(self, image_path: str, bins: int = 64) -> np.ndarray:
        """Extract a normalized per-channel RGB histogram.

        Computes a histogram for each channel (R, G, B) separately, then
        concatenates them. Total dimensions = bins * 3.

        Args:
            image_path: Path to the image file.
            bins: Number of histogram bins per channel.

        Returns:
            Normalized histogram vector of shape (bins * 3,).

        Raises:
            ValueError: If the image cannot be loaded.
        """
        image = self._load_and_resize(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Compute per-channel histograms and concatenate
        histograms = []
        for channel in range(3):
            hist = cv2.calcHist([image_rgb], [channel], None, [bins], [0, 256])
            histograms.append(hist.flatten())

        combined = np.concatenate(histograms)

        # L1 normalize
        total = combined.sum()
        if total > 0:
            combined = combined / total

        return combined.astype(np.float32)

    def extract_hsv_histogram(self, image_path: str, bins: Tuple[int, int, int] = (18, 16, 4)) -> np.ndarray:
        """Extract a normalized per-channel HSV histogram.

        Computes separate histograms for H, S, V channels and concatenates.
        Total dimensions = bins[0] + bins[1] + bins[2].

        Args:
            image_path: Path to the image file.
            bins: Number of bins for (H, S, V) channels.
                  Defaults to (18, 16, 4) for a 38-dimensional descriptor.

        Returns:
            Normalized histogram vector of shape (sum(bins),).

        Raises:
            ValueError: If the image cannot be loaded.
        """
        image = self._load_and_resize(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        ranges = [(0, 180), (0, 256), (0, 256)]  # H range is 0-180 in OpenCV
        histograms = []
        for channel, (num_bins, (lo, hi)) in enumerate(zip(bins, ranges)):
            hist = cv2.calcHist([hsv], [channel], None, [num_bins], [lo, hi])
            histograms.append(hist.flatten())

        combined = np.concatenate(histograms)

        # L1 normalize
        total = combined.sum()
        if total > 0:
            combined = combined / total

        return combined.astype(np.float32)

    def extract(self, image_path: str, color_space: str = 'rgb',
                bins: Union[int, Tuple[int, int, int]] = 64) -> np.ndarray:
        """Extract color histogram using the specified color space.

        Args:
            image_path: Path to the image file.
            color_space: 'rgb' or 'hsv'.
            bins: Bins per channel. For RGB, an int. For HSV, a tuple of 3 ints.

        Returns:
            Normalized histogram feature vector.

        Raises:
            ValueError: If the color space is unsupported or image cannot be loaded.
        """
        if color_space.lower() == 'rgb':
            return self.extract_rgb_histogram(
                image_path,
                bins=bins if isinstance(bins, int) else bins[0]
            )
        elif color_space.lower() == 'hsv':
            if isinstance(bins, int):
                bins = (18, 16, 4)
            return self.extract_hsv_histogram(image_path, bins=bins)
        else:
            raise ValueError(f"Unsupported color space: {color_space}")
