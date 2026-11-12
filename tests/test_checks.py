import numpy as np
import pytest
import cv2
from image_qa.checks import check_blur,blur_score

def make_flat(size=100):
    return np.full((size,size,3),128,dtype=np.uint8)

def test_flat_image_is_blurry():
    result = check_blur(make_flat())
    assert result.passed is False
    assert result.value == pytest.approx(0.0)

def make_checkerboard(size=200,square=20):
    rows = (np.arange(size) // square) % 2
    board = (rows[:, None] ^ rows[None,:]) * 255
    gray = board.astype(np.uint8)
    return cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)

def test_sharp_image_passes():
    result = check_blur(make_checkerboard())
    assert result.passed is True
    assert result.name == 'blur'

def make_blurry(image,ksize=31):
    return cv2.GaussianBlur(image, (ksize,ksize),0)

def test_blurry_image_fails():
    result = check_blur(make_blurry(make_checkerboard()))
    assert result.passed is False
    assert 'blurry' in result.reason

def test_sharp_scores_higher_than_blurry():
    sharp = make_checkerboard()
    assert blur_score(sharp) > blur_score(make_blurry(sharp))