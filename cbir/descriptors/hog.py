import cv2
import numpy as np
from skimage.feature import hog
from typing import Tuple

class HOGDescriptor:
    def extract(self, image_path: str, orientations: int = 9, pixels_per_cell: Tuple[int, int] = (8, 8), cells_per_block: Tuple[int, int] = (2, 2)) -> np.ndarray:
        """
        Extract Histogram of Oriented Gradients (HOG) features from an image.
        Returns L2-normalized feature vector.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))

        features = hog(
            gray,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm='L2-Hys',
            visualize=False,
            feature_vector=True
        )

        features = features / (np.linalg.norm(features) + 1e-7)
        return features
