import cv2
import numpy as np
import pytest
import time
from pathlib import Path
from photofilter.core import process_folder

def _create_image(path: Path, pattern: str):
    # create a simple image based on pattern name
    size = 64
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    if pattern == 'A':
        cv2.circle(img, (size//2, size//2), 10, (0,0,0), -1)
    elif pattern == 'B':
        cv2.rectangle(img, (10,10), (30,30), (0,0,0), -1)
    else:
        pass
    cv2.imwrite(str(path), img)

def test_process_folder_basic(tmp_path: Path):
    # Input folder with three images (two same, one different)
    input_dir = tmp_path / 'in'
    input_dir.mkdir()
    out_dir = tmp_path / 'out'
    # create images
    a1 = input_dir / 'a1.jpg'
    _create_image(a1, 'A')
    
    # Slight sleep so modification times are slightly different
    time.sleep(0.1)
    a2 = input_dir / 'a2.jpg'
    _create_image(a2, 'A')
    
    # Sleep > 2s to simulate a different burst
    time.sleep(2.1)
    b = input_dir / 'b.jpg'
    _create_image(b, 'B')
    
    # Run processing with default config (2s time threshold)
    process_folder(input_dir, out_dir, threshold=0.0, config={}, use_gpu=False)
    
    # Expect one image (best from group A) in out_dir, others in _Rejected
    rejected_dir = input_dir / '_Rejected'
    out_files = list(out_dir.glob('*.jpg'))
    rejected_files = list(rejected_dir.glob('*.jpg'))
    
    # We expect 2 winners (1 from A burst, 1 from B burst)
    assert len(out_files) == 2
    assert len(rejected_files) == 1
    
    # Ensure the output file is from group A (hashes equal to a1 or a2)
    out_names = {f.name for f in out_files}
    assert out_names >= {'b.jpg'}  # b is a separate burst, always kept
    assert len(out_names & {'a1.jpg', 'a2.jpg'}) == 1  # one from group A kept
