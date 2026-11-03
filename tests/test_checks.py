"""Tests for image_qa.checks.

Images are generated in code (no files needed), so the tests are self-contained.
"""

import cv2
import numpy as np
import pytest

from image_qa.checks import blur_score, check_blur


def make_checkerboard(size=200, square=20):
    """A black/white checkerboard: lots of hard edges, so it is very sharp."""
    rows = (np.arange(size) // square) % 2
    board = (rows[:, None] ^ rows[None, :]) * 255
    gray = board.astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def make_blurry(image, ksize=31):
    """Heavily blur an image with a Gaussian filter."""
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def test_sharp_image_has_higher_score_than_blurry():
    sharp = make_checkerboard()
    blurry = make_blurry(sharp)
    assert blur_score(sharp) > blur_score(blurry)


def test_sharp_image_passes_blur_check():
    result = check_blur(make_checkerboard())
    assert result.passed is True
    assert result.name == "blur"


def test_blurry_image_fails_blur_check():
    result = check_blur(make_blurry(make_checkerboard()))
    assert result.passed is False
    assert "blurry" in result.reason


def test_flat_image_is_flagged_as_blurry():
    # A solid gray image has no edges at all: score should be ~0.
    flat = np.full((100, 100, 3), 128, dtype=np.uint8)
    result = check_blur(flat)
    assert result.passed is False
    assert result.value == pytest.approx(0.0)
