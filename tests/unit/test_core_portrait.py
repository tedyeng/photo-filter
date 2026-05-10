import pytest
import numpy as np
from photofilter.core import score_portrait

def test_score_portrait_valid_faces():
    # Mocking YuNet faces array
    # Format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
    h, w = 1000, 1000
    
    # Face exactly in center: 400, 400, 200, 200
    perfect_face = np.array([[400, 400, 200, 200, 450, 450, 550, 450, 500, 500, 450, 550, 550, 550, 0.99]], dtype=np.float32)
    score1 = score_portrait(perfect_face, h, w)
    
    # Face off-center: 0, 0, 200, 200
    offcenter_face = np.array([[0, 0, 200, 200, 50, 50, 150, 50, 100, 100, 50, 150, 150, 150, 0.99]], dtype=np.float32)
    score2 = score_portrait(offcenter_face, h, w)
    
    assert score1 > score2
    assert 0.0 <= score1 <= 1.2 # Including size bonus

def test_score_portrait_no_faces():
    score = score_portrait(None, 1000, 1000)
    assert score == 0.0
    
    score = score_portrait(np.array([]), 1000, 1000)
    assert score == 0.0
