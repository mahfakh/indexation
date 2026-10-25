import cv2
import numpy as np
from skimage.feature import local_binary_pattern

class LBPDescriptor:
    def extract(self, image_path: str, num_points: int = 24, radius: int = 3) -> np.ndarray:
        """
        Extract Local Binary Patterns (LBP) features from an image.
        Returns a normalized histogram.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (256, 256))

        lbp = local_binary_pattern(gray, num_points, radius, method='uniform')
        
        n_bins = num_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        return hist
