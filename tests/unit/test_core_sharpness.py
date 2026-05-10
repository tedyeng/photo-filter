import cv2
import numpy as np
import pytest
from pathlib import Path
from photofilter.core import compute_sharpness

def create_sharp_image(tmp_path: Path) -> Path:
    # Create a high‑frequency checkerboard pattern (256×256)
    size = 256
    img = np.zeros((size, size), dtype=np.uint8)
    # Fill squares
    square = 16
    for y in range(0, size, square):
        for x in range(0, size, square):
            if (x // square + y // square) % 2 == 0:
                img[y:y+square, x:x+square] = 255
    path = tmp_path / 'sharp.png'
    cv2.imwrite(str(path), img)
    return path

def create_blurred_image(sharp_path: Path) -> Path:
    img = cv2.imread(str(sharp_path), cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (9, 9), 0)
    blurred_path = sharp_path.parent / 'blurred.png'
    cv2.imwrite(str(blurred_path), blurred)
    return blurred_path

def test_compute_sharpness_relative(tmp_path: Path):
    sharp_path = create_sharp_image(tmp_path)
    blurred_path = create_blurred_image(sharp_path)
    sharp_score = compute_sharpness(sharp_path)
    blur_score = compute_sharpness(blurred_path)
    assert sharp_score > blur_score, "Sharp image should have higher sharpness score"
    assert sharp_score > 0, "Sharpness score should be positive"
