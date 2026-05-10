import pathlib
import pytest
from photofilter.core import process_folder

def _create_dummy_image(path: pathlib.Path):
    import numpy as np, cv2
    img = np.full((64, 64, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(path), img)

def test_process_folder_csv(tmp_path: pathlib.Path):
    in_dir = tmp_path / 'in'
    out_dir = tmp_path / 'out'
    in_dir.mkdir()
    # create three images (two similar, one different)
    a1 = in_dir / 'a1.jpg'
    a2 = in_dir / 'a2.jpg'
    b = in_dir / 'b.jpg'
    _create_dummy_image(a1)
    _create_dummy_image(a2)
    _create_dummy_image(b)
    # Run processing
    process_folder(in_dir, out_dir, threshold=0.0, config={}, use_gpu=False)
    csv_path = in_dir / 'defects.csv'
    assert csv_path.is_file(), "defects.csv should be generated"
    # Simple check: csv contains expected columns
    import pandas as pd
    df = pd.read_csv(csv_path)
    expected_cols = {'Filename', 'Burst ID', 'Status', 'Reason', 'Has Face', 'Portrait Score', 'Landscape Score', 'Composite', 'Hash', 'Has RAW'}
    assert expected_cols.issubset(set(df.columns))
