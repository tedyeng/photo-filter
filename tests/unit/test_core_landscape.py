import pytest
from pathlib import Path
from photofilter.core import score_landscape

def test_score_landscape_valid_image(tmp_path: Path):
    from PIL import Image
    import numpy as np
    
    img_path = tmp_path / "landscape.jpg"
    # Create a gradient image
    arr = np.linspace(0, 255, 100*100, dtype=np.uint8).reshape((100, 100))
    img = Image.fromarray(arr)
    img.save(img_path)
    
    score = score_landscape(img_path)
    assert isinstance(score, float)
    assert score >= 0.0

def test_score_landscape_invalid_image(tmp_path: Path):
    invalid_path = tmp_path / "missing.jpg"
    score = score_landscape(invalid_path)
    assert score == 0.0
