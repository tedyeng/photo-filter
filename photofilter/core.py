import hashlib
import logging
import numpy as np
import cv2
import urllib.request
import subprocess
from pathlib import Path
from typing import Mapping

logger = logging.getLogger(__name__)

# Expected SHA256 of the YuNet ONNX model
_YUNET_MODEL_HASH = '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'

# Cached FaceDetectorYN instance to avoid re‑initialising per image
_face_detector = None  # type: cv2.FaceDetectorYN | None
_face_detector_size = None  # type: tuple[int, int] | None

# Optional rawpy import for RAW support
try:
    import rawpy
except ImportError:  # pragma: no cover
    rawpy = None

RAW_EXTS = {'.cr3', '.cr2', '.arw', '.dng', '.nef', '.orf', '.raf', '.raw', '.rw2'}
JPG_EXTS = {'.jpg', '.jpeg'}

def _load_image_grayscale(image_path: Path) -> np.ndarray:
    """Load image as grayscale, supporting JPEG and RAW formats.
    Returns a NumPy 2‑D array or None on failure.
    """
    suffix = image_path.suffix.lower()
    if suffix in RAW_EXTS:
        if rawpy is None:
            raise ImportError('rawpy is required to read RAW files')
        raw = rawpy.imread(str(image_path))
        rgb = raw.postprocess()
        raw.close()
        # Convert RGB to grayscale via OpenCV
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        return cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

def _load_image_color(image_path: Path) -> np.ndarray:
    """Load image as BGR (OpenCV format), supporting JPEG and RAW.
    Returns a NumPy 3‑D array or None on failure.
    """
    suffix = image_path.suffix.lower()
    if suffix in RAW_EXTS:
        if rawpy is None:
            raise ImportError('rawpy is required to read RAW files')
        raw = rawpy.imread(str(image_path))
        rgb = raw.postprocess()
        raw.close()
        # Convert RGB to BGR for OpenCV compatibility
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        return cv2.imread(str(image_path))

def compute_sharpness(image_path: Path) -> float:
    """Calculate Laplacian variance of the sharpest 10% of the image.
    The image is divided into 64×64 blocks; the variance of each block's Laplacian is computed.
    The mean of the top‑10% highest‑variance blocks is returned as the sharpness score.
    The raw score (mean variance) is returned – higher means sharper.
    """
    img = _load_image_grayscale(image_path)
    if img is None:
        return 0.0
    # Compute Laplacian
    lap = cv2.Laplacian(img, cv2.CV_64F)
    h, w = lap.shape
    block_size = 64
    # Trim to whole multiple of block size
    h_trim = h - (h % block_size)
    w_trim = w - (w % block_size)
    if h_trim == 0 or w_trim == 0:
        # Image too small for block division – fall back to global variance
        return float(lap.var())
    lap_trimmed = lap[:h_trim, :w_trim]
    # Reshape into blocks
    blocks = lap_trimmed.reshape(h_trim // block_size, block_size, w_trim // block_size, block_size)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, block_size, block_size)
    variances = np.var(blocks, axis=(1, 2))
    # Take top 10% blocks
    top_n = max(1, int(len(variances) * 0.10))
    top_variances = np.sort(variances)[-top_n:]
    return float(np.mean(top_variances))

# ---------------------------------------------------------------------------
# Composition scoring – uses YuNet face detection to evaluate subject centering
# ---------------------------------------------------------------------------

def _ensure_yunet_model() -> Path:
    """Download YuNet ONNX model if not present and return its Path.
    Mirrors the logic from `cull_defects.py`.
    """
    YUNET_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    model_path = Path.home() / ".cache" / "photofilter" / "face_detection_yunet_2023mar.onnx"
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(YUNET_MODEL_URL, str(model_path))
    except Exception:
        # Fallback to curl for macOS SSL issues
        subprocess.run(["curl", "-L", "-o", str(model_path), YUNET_MODEL_URL], check=True)
    return model_path


def _get_face_detector(w: int, h: int):
    """Return cached FaceDetectorYN instance, recreating if image size changes."""
    global _face_detector, _face_detector_size
    if _face_detector is None or _face_detector_size != (w, h):
        model_path = _ensure_yunet_model()
        _face_detector = cv2.FaceDetectorYN.create(
            model=str(model_path),
            config="",
            input_size=(w, h),
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=5000,
        )
        _face_detector_size = (w, h)
    return _face_detector

def detect_faces(image_path: Path):
    """Detect faces in an image using YuNet.
    Returns (faces, h, w) where faces is the numpy array from YuNet,
    and h, w are the image dimensions. If no faces, returns (None, h, w).
    """
    img = _load_image_color(image_path)
    if img is None:
        return None, 0, 0
    h, w = img.shape[:2]
    fd = _get_face_detector(w, h)
    _, faces = fd.detect(img)
    if faces is None or len(faces) == 0:
        return None, h, w
    return faces, h, w

def score_landscape(image_path: Path) -> float:
    """Compute context-aware landscape score. Currently uses Laplacian sharpness."""
    return compute_sharpness(image_path)

def score_portrait(faces, h: int, w: int) -> float:
    """Compute portrait score based on YuNet faces array.
    Returns a score from 0.0 to ~1.2 based on centering and face size.
    """
    if faces is None or len(faces) == 0:
        return 0.0
    
    total_area = h * w
    best_face = None
    best_area_pct = 0.0
    for face in faces:
        fx, fy, fw, fh = face[0:4]
        area_pct = (fw * fh) / total_area * 100
        if area_pct > best_area_pct:
            best_area_pct = area_pct
            best_face = face
            
    if best_face is None:
        return 0.0
        
    fx, fy, fw, fh = best_face[0:4]
    face_cx = fx + fw / 2.0
    face_cy = fy + fh / 2.0
    dx = abs(face_cx - w / 2) / w
    dy = abs(face_cy - h / 2) / h
    centering_score = 1.0 - (dx + dy) / 2.0
    size_bonus = min(0.2, best_area_pct / 100.0)
    
    return max(0.0, min(1.0, centering_score + size_bonus))



import imagehash
from datetime import datetime
from .utils import hash_image, move_to, get_exif_time

def group_bursts(image_infos: list[dict], time_thresh: float = 2.0, hash_thresh: int = 10) -> list[list[dict]]:
    """Group images into burst events based on time difference and perceptual hash similarity.
    `image_infos` is a list of dicts containing at least 'path', 'time', and 'phash' keys.
    Returns a list of lists, where each sublist is a group of image info dicts.
    """
    if not image_infos:
        return []
        
    # Sort images by time. If time is missing, use a fallback (e.g., maximum time) to put them at the end
    sorted_infos = sorted(image_infos, key=lambda x: x.get('time') or datetime.max)
    
    groups = []
    current_group = [sorted_infos[0]]
    
    for i in range(1, len(sorted_infos)):
        info = sorted_infos[i]
        prev_info = current_group[-1]
        
        # Check time diff
        time_present = info.get('time') is not None and prev_info.get('time') is not None
        time_diff_ok = False
        if time_present:
            diff = abs((info['time'] - prev_info['time']).total_seconds())
            if diff <= time_thresh:
                time_diff_ok = True
                
        # Check hash
        hash_present = info.get('phash') and prev_info.get('phash')
        hash_ok = False
        if hash_present:
            try:
                h1 = imagehash.hex_to_hash(info['phash'])
                h2 = imagehash.hex_to_hash(prev_info['phash'])
                if (h1 - h2) <= hash_thresh:
                    hash_ok = True
            except Exception:
                pass
                
        # To be in the same group, it must not violate time or hash thresholds
        match = True
        if time_present and not time_diff_ok:
            match = False
        if hash_present and not hash_ok:
            match = False
            
        if match:
            current_group.append(info)
        else:
            groups.append(current_group)
            current_group = [info]
            
    if current_group:
        groups.append(current_group)
        
    return groups

def _find_raw_pair(image_path: Path) -> Path | None:
    """Find a RAW file with the same stem as the given JPG.
    Checks extensions defined in core module's RAW_EXTS.
    """
    for ext in RAW_EXTS:
        candidate = image_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def _move_image_and_raw(src: Path, dest_dir: Path) -> None:
    """Move an image and its associated RAW pair (if any) to dest_dir."""
    move_to(src, dest_dir)
    raw = _find_raw_pair(src)
    if raw:
        move_to(raw, dest_dir)

def process_folder(input_path: Path, output_path: Path, threshold: float = 0.55, config: dict = None, use_gpu: bool = False) -> None:
    """Full implementation:
    1. Scan JPG files.
    2. Extract EXIF time and perceptual hash.
    3. Group by time and hash (burst detection).
    4. Within each burst group, detect faces and score accordingly.
    5. Move selected image (and its RAW pair) to output_path.
    6. Move all others (and their RAW pairs) to a `_Rejected` folder.
    """
    # Gather all image files (JPEG + RAW) case‑insensitively, ignoring macOS metadata files
    image_files = [p for p in input_path.iterdir() if p.is_file() and not p.name.startswith('._') and p.suffix.lower() in JPG_EXTS.union(RAW_EXTS)]
    if not image_files:
        raise ValueError('No supported image files found in input folder.')
    
    # Process JPGs, fallback to image_files if none
    jpg_files = [p for p in image_files if p.suffix.lower() in JPG_EXTS]
    if not jpg_files:
        jpg_files = image_files

    image_infos = []
    for f in jpg_files:
        ph = hash_image(f)
        t = get_exif_time(f)
        image_infos.append({
            'path': f,
            'phash': ph,
            'time': t,
        })
        
    time_thresh = config.get('burst_time_threshold', 2.0) if config else 2.0
    bursts = group_bursts(image_infos, time_thresh=time_thresh)
    
    rejected_dir = input_path / '_Rejected'
    report_rows = []
    
    for burst_id, burst in enumerate(bursts, start=1):
        for info in burst:
            faces, h, w = detect_faces(info['path'])
            has_face = faces is not None and len(faces) > 0
            if has_face:
                portrait_score = score_portrait(faces, h, w)
                landscape_score = 0.0
                composite = portrait_score
            else:
                portrait_score = 0.0
                landscape_score = score_landscape(info['path'])
                composite = landscape_score
                
            info.update({
                'has_face': has_face,
                'portrait_score': portrait_score,
                'landscape_score': landscape_score,
                'composite': composite,
                'burst_id': burst_id
            })
            
        # Select the single best image in this burst
        best = max(burst, key=lambda i: (i['has_face'], i['composite']))
        
        for info in burst:
            status = 'KEEP' if info is best else 'REJECT'
            
            if status == 'KEEP':
                _move_image_and_raw(info['path'], output_path)
            else:
                _move_image_and_raw(info['path'], rejected_dir)
                    
            report_rows.append({
                'Filename': info['path'].name,
                'Burst ID': info['burst_id'],
                'Status': status,
                'Reason': 'Not best' if status == 'REJECT' else '',
                'Has Face': 'Yes' if info['has_face'] else 'No',
                'Portrait Score': f"{info['portrait_score']:.2f}",
                'Landscape Score': f"{info['landscape_score']:.2f}",
                'Composite': f"{info['composite']:.2f}",
                'Hash': info['phash'],
                'Has RAW': 'Yes' if _find_raw_pair(info['path']) else 'No',
            })
            
    if report_rows:
        import pandas as pd
        csv_path = input_path / 'defects.csv'
        pd.DataFrame(report_rows).to_csv(csv_path, index=False)
        print(f"📑 Report saved to: {csv_path}")
        
    if use_gpu:
        print("GPU acceleration enabled (simulation).")
    return None
