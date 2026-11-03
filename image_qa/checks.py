"""Individual image quality checks.

Each check takes an image (a numpy array as loaded by OpenCV) and returns a
CheckResult: did it pass, the measured value, and a short reason.
Keeping checks as small pure functions makes them easy to test and combine.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CheckResult:
    """Outcome of one quality check."""
    name: str
    passed: bool
    value: float
    reason: str


def load_image(path):
    """Load an image from disk as a BGR numpy array. Raises if it can't be read."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def blur_score(image):
    """Measure sharpness. Higher = sharper.

    Uses the variance of the Laplacian: the Laplacian filter highlights edges,
    and a sharp image has strong, varied edges while a blurry one is smooth.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def check_blur(image, threshold=100.0):
    """Fail the image if its sharpness score is below `threshold`.

    The default threshold is a common starting point; tune it against real
    samples from your pipeline.
    """
    score = blur_score(image)
    passed = score >= threshold
    reason = "sharp" if passed else f"blurry (score {score:.1f} < {threshold})"
    return CheckResult(name="blur", passed=passed, value=score, reason=reason)
