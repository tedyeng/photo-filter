import json
from pathlib import Path
import pytest
from datetime import datetime
import os
from photofilter.utils import load_config, get_exif_time

def test_load_config_missing_file(tmp_path: Path):
    missing = tmp_path / 'nonexistent.env'
    with pytest.raises(ValueError) as exc:
        load_config(missing)
    assert 'Config file not found' in str(exc.value)

def test_load_config_success(tmp_path: Path):
    cfg = tmp_path / '.env'
    cfg.write_text("BURST_TIME_THRESHOLD=3.5\nWEIGHT_COMPOSITION=0.8\nWEIGHT_SHARPNESS=0.2\nGPU_ENABLED=false", encoding='utf-8')
    result = load_config(cfg)
    assert result['burst_time_threshold'] == 3.5
    assert result['weights']['composition'] == 0.8
    assert result['weights']['sharpness'] == 0.2
    assert result['gpu_enabled'] is False

def test_load_config_defaults(tmp_path: Path):
    result = load_config(None)
    assert result['burst_time_threshold'] == 2.0
    assert result['weights']['composition'] == 0.6
    assert result['gpu_enabled'] is True

def test_get_exif_time_with_valid_exif(tmp_path: Path):
    # Creating a dummy image with EXIF using Pillow
    from PIL import Image
    import piexif
    img_path = tmp_path / "test_exif.jpg"
    img = Image.new('RGB', (10, 10))
    exif_dict = {"Exif": {piexif.ExifIFD.DateTimeOriginal: b"2023:05:10 15:30:45"}}
    exif_bytes = piexif.dump(exif_dict)
    img.save(img_path, "jpeg", exif=exif_bytes)
    
    dt = get_exif_time(img_path)
    assert dt is not None
    assert dt.year == 2023
    assert dt.month == 5
    assert dt.day == 10
    assert dt.hour == 15
    assert dt.minute == 30
    assert dt.second == 45

def test_get_exif_time_fallback_to_mtime(tmp_path: Path):
    # Image without EXIF
    from PIL import Image
    img_path = tmp_path / "test_no_exif.jpg"
    img = Image.new('RGB', (10, 10))
    img.save(img_path, "jpeg")
    
    dt = get_exif_time(img_path)
    assert dt is not None
    # Check if it's close to current time
    now = datetime.now()
    assert abs((dt - now).total_seconds()) < 5

def test_get_exif_time_invalid_file(tmp_path: Path):
    invalid_path = tmp_path / "nonexistent.jpg"
    dt = get_exif_time(invalid_path)
    assert dt is None
