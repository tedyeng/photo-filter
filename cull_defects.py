#!/usr/bin/env python3
import os
import cv2
import numpy as np
import json
import shutil
import hashlib
import argparse
import urllib.request
import pandas as pd
from pathlib import Path
from PIL import Image
import imagehash
from typing import List, Dict, Any, Tuple, Optional

# --- Face Detection Model ---
YUNET_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_MODEL_PATH = Path.home() / ".cache" / "cull_defects" / "face_detection_yunet_2023mar.onnx"

# Minimum face area as percentage of total image to count as "main subject"
# For a 60MP image (9504x6336), a face at 0.3% = ~500x360px = clearly a main subject
MIN_FACE_AREA_PCT = 0.3

# Supported Extensions
JPG_EXTS = {'.jpg', '.jpeg'}
# Expanded RAW list for multiple cameras (Canon, Sony, Nikon, DJI, etc.)
RAW_EXTS = {'.cr3', '.cr2', '.arw', '.dng', '.nef', '.orf', '.raf', '.raw', '.rw2'}

def get_sharpness_score(image_path: Path) -> float:
    """Calculates Laplacian variance of the sharpest 10% of the image.
    This handles shallow depth of field and empty spaces (e.g. skies) by
    only judging the parts of the image that are actually in focus."""
    # Load image in grayscale at full resolution (No resizing!)
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    
    # Calculate Laplacian
    lap = cv2.Laplacian(img, cv2.CV_64F)
    
    # Divide image into 64x64 blocks
    h, w = img.shape
    block_size = 64
    
    h_trim = h - (h % block_size)
    w_trim = w - (w % block_size)
    
    # Fallback to global variance if image is too small
    if h_trim == 0 or w_trim == 0:
        return float(lap.var())
        
    lap_trimmed = lap[:h_trim, :w_trim]
    
    # Reshape into blocks and calculate variance per block
    blocks = lap_trimmed.reshape(h_trim // block_size, block_size, w_trim // block_size, block_size)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, block_size, block_size)
    variances = np.var(blocks, axis=(1, 2))
    
    # Take the top 10% sharpest blocks (representing the focus point)
    num_top_blocks = max(1, int(len(variances) * 0.10))
    top_variances = np.sort(variances)[-num_top_blocks:]
    
    return float(np.mean(top_variances))

def get_phash(image_path: Path) -> str:
    """Calculates perceptual hash for burst grouping."""
    try:
        with Image.open(image_path) as img:
            return str(imagehash.phash(img))
    except Exception:
        return ""

def _ensure_yunet_model() -> Path:
    """Downloads YuNet face detection model if not already cached."""
    if YUNET_MODEL_PATH.exists():
        return YUNET_MODEL_PATH
    YUNET_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"📥 Downloading face detection model...")
    try:
        urllib.request.urlretrieve(YUNET_MODEL_URL, str(YUNET_MODEL_PATH))
    except Exception:
        # Fallback to curl (handles macOS SSL cert issues)
        import subprocess
        subprocess.run(
            ["curl", "-L", "-o", str(YUNET_MODEL_PATH), YUNET_MODEL_URL],
            check=True
        )
    print(f"✅ Model saved to {YUNET_MODEL_PATH}")
    return YUNET_MODEL_PATH

def _eye_openness(img_gray: np.ndarray, eye_center: np.ndarray, eye_dist: float) -> float:
    """Measures eye openness by analyzing the Laplacian variance of the eye region.
    Open eyes have more edge detail (iris, pupil, eyelashes) = higher Laplacian variance.
    Closed/half-closed eyes are smoother = lower variance."""
    ex, ey = int(eye_center[0]), int(eye_center[1])
    half_w = int(eye_dist * 0.20)
    half_h = int(eye_dist * 0.12)
    
    h, w = img_gray.shape
    y1, y2 = max(0, ey - half_h), min(h, ey + half_h)
    x1, x2 = max(0, ex - half_w), min(w, ex + half_w)
    
    eye_crop = img_gray[y1:y2, x1:x2]
    if eye_crop.size == 0:
        return 0.0
    
    return float(cv2.Laplacian(eye_crop, cv2.CV_64F).var())

def get_face_expression_score(image_path: Path) -> Tuple[float, bool]:
    """Scores facial expression quality using YuNet face detection.
    
    Eye openness is the most important factor (40%), measured by analyzing
    the Laplacian variance of the eye region crops.
    
    Returns:
        (expression_score, has_main_face)
        - expression_score: 0-100 composite score (higher = better expression)
        - has_main_face: True if a face occupies >MIN_FACE_AREA_PCT of the image
    """
    model_path = _ensure_yunet_model()
    
    img = cv2.imread(str(image_path))
    if img is None:
        return 0.0, False
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    total_area = h * w
    
    # Create face detector
    fd = cv2.FaceDetectorYN.create(
        model=str(model_path),
        config="",
        input_size=(w, h),
        score_threshold=0.5,
        nms_threshold=0.3,
        top_k=5000
    )
    
    _, faces = fd.detect(img)
    
    if faces is None or len(faces) == 0:
        return 0.0, False
    
    # Find the largest face (main subject)
    best_score = 0.0
    has_main_face = False
    
    for face in faces:
        fx, fy, fw, fh = face[0:4]
        face_area_pct = (fw * fh) / total_area * 100
        
        if face_area_pct < MIN_FACE_AREA_PCT:
            continue  # Skip tiny background faces
        
        has_main_face = True
        
        right_eye = np.array([face[4], face[5]])
        left_eye = np.array([face[6], face[7]])
        nose_tip = np.array([face[8], face[9]])
        right_mouth = np.array([face[10], face[11]])
        left_mouth = np.array([face[12], face[13]])
        confidence = face[14]
        
        # Inter-ocular distance (baseline for normalization)
        eye_dist = np.linalg.norm(left_eye - right_eye)
        if eye_dist < 1:
            continue
        
        # --- Expression metrics ---
        
        # 1. Eye Openness (MOST IMPORTANT): Laplacian variance of eye region
        #    Open eyes have iris/pupil detail -> high variance (30-70+)
        #    Closed/half-closed eyes are smooth -> low variance (10-25)
        r_eye_open = _eye_openness(img_gray, right_eye, eye_dist)
        l_eye_open = _eye_openness(img_gray, left_eye, eye_dist)
        avg_eye_open = (r_eye_open + l_eye_open) / 2
        # Normalize: values typically range 10-70, map to 0-1
        eye_open_score = min(1.0, max(0.0, (avg_eye_open - 10.0) / 50.0))
        
        # 2. Smile Score: mouth width relative to eye distance
        mouth_width = np.linalg.norm(left_mouth - right_mouth)
        smile_ratio = mouth_width / eye_dist
        smile_score = min(1.0, max(0.0, (smile_ratio - 0.5) / 0.5))
        
        # 3. Face Confidence: YuNet's confidence (face clarity, pose quality)
        conf_score = min(1.0, confidence)
        
        # 4. Symmetry Score: balanced face = good eye contact / pose
        nose_to_right_eye = np.linalg.norm(right_eye - nose_tip)
        nose_to_left_eye = np.linalg.norm(left_eye - nose_tip)
        nose_to_right_mouth = np.linalg.norm(right_mouth - nose_tip)
        nose_to_left_mouth = np.linalg.norm(left_mouth - nose_tip)
        eye_symmetry = 1.0 - abs(nose_to_right_eye - nose_to_left_eye) / eye_dist
        mouth_symmetry = 1.0 - abs(nose_to_right_mouth - nose_to_left_mouth) / eye_dist
        symmetry_score = max(0.0, (eye_symmetry + mouth_symmetry) / 2)
        
        # 5. Natural expression: mouth position relative to face proportions
        mouth_center = (left_mouth + right_mouth) / 2
        nose_to_mouth_y = mouth_center[1] - nose_tip[1]
        eye_center = (left_eye + right_eye) / 2
        eye_to_nose_y = nose_tip[1] - eye_center[1]
        mouth_ratio = nose_to_mouth_y / eye_to_nose_y if eye_to_nose_y > 0 else 0
        natural_score = min(1.0, max(0.0, 1.0 - abs(mouth_ratio - 0.65) / 0.65))
        
        # Composite score (weighted):
        #   Eye openness: 40% — most important (user's top priority)
        #   Smile:        25% — pleasant expression
        #   Confidence:   15% — face clarity and quality
        #   Symmetry:     10% — good pose, eye contact
        #   Natural:      10% — natural mouth position
        composite = (
            eye_open_score * 40 +
            smile_score * 25 +
            conf_score * 15 +
            symmetry_score * 10 +
            natural_score * 10
        )
        
        # Weight by face size (larger face = more important subject)
        size_weight = min(1.0, face_area_pct / 10.0)
        weighted_score = composite * (0.5 + 0.5 * size_weight)
        
        best_score = max(best_score, weighted_score)
    
    return best_score, has_main_face

def find_raw_pair(image_path: Path) -> Optional[Path]:
    """Finds a RAW file with the same stem but different extension."""
    # Check for same name with different extension
    for ext in RAW_EXTS:
        # Check case-insensitive
        for p in image_path.parent.glob(f"{image_path.stem}.*"):
            if p.suffix.lower() == ext:
                return p
    return None

def process_folder(folder_path: Path, blur_threshold: float = 50.0, hash_threshold: int = 4):
    """Processes a folder to cull blurry and duplicate photos."""
    folder_path = Path(folder_path).resolve()
    print(f"🔍 Analyzing folder: {folder_path}")
    
    jpg_files = [f for f in folder_path.iterdir() if f.suffix.lower() in JPG_EXTS and not f.name.startswith('._')]
    if not jpg_files:
        print("⚠️ No JPG files found.")
        return

    results = []
    
    # 1. First Pass: Scoring and Hashing
    print(f"📊 Calculating scores for {len(jpg_files)} images...")
    for i, jpg_path in enumerate(jpg_files, 1):
        print(f"  [{i}/{len(jpg_files)}] {jpg_path.name}", end="\r")
        score = get_sharpness_score(jpg_path)
        p_hash = get_phash(jpg_path)
        raw_path = find_raw_pair(jpg_path)
        
        results.append({
            "filename": jpg_path.name,
            "path": jpg_path,
            "raw_path": raw_path,
            "blur_score": score,
            "phash": p_hash,
            "is_rejected": score < blur_threshold,
            "reject_reason": "Blurry" if score < blur_threshold else "",
            "group_id": None
        })
    print("\n✅ Scoring complete.")

    df = pd.DataFrame(results)
    
    # 2. Second Pass: Grouping Duplicates (Bursts) by Hamming Distance
    print("🤝 Grouping bursts...")
    # Convert hex hashes to imagehash objects for distance calculation
    hashes = [imagehash.hex_to_hash(h) if h else None for h in df['phash']]
    
    # Assign group IDs based on hamming distance <= 10
    # (Threshold 10 catches burst shots with slight camera movement or exposure changes.
    #  Cross-group distances are typically 20+ so this is safe.)
    group_ids = [None] * len(df)
    current_group = 0
    
    for i, h1 in enumerate(hashes):
        if not h1: continue
        if group_ids[i] is None:
            group_ids[i] = str(current_group)
            
            # Find all similar images that haven't been grouped yet
            for j in range(i + 1, len(hashes)):
                h2 = hashes[j]
                if h2 and group_ids[j] is None:
                    if h1 - h2 <= 10:  # Hamming distance threshold
                        group_ids[j] = str(current_group)
            
            current_group += 1
            
    df['group_id'] = group_ids
    
    # 3. Third Pass: Pick the Winner in each group
    burst_count = 0
    for group_id, group in df.groupby('group_id'):
        if len(group) > 1:
            burst_count += 1
            # Sort by blur score descending to find the sharpest
            sorted_group = group.sort_values(by='blur_score', ascending=False)
            sharpest_idx = sorted_group.index[0]
            sharpest_score = df.at[sharpest_idx, 'blur_score']
            
            # Check if top candidates are within 15% sharpness of each other
            # If so, use facial expression to pick the winner
            close_candidates = sorted_group[
                sorted_group['blur_score'] >= sharpest_score * 0.85
            ]
            
            winner_idx = sharpest_idx
            selection_method = "sharpness"
            
            if len(close_candidates) > 1:
                # Multiple images with similar sharpness — check for faces
                face_scores = {}
                any_main_face = False
                for idx in close_candidates.index:
                    expr_score, has_face = get_face_expression_score(
                        df.at[idx, 'path']
                    )
                    face_scores[idx] = expr_score
                    if has_face:
                        any_main_face = True
                
                if any_main_face:
                    # Pick the best facial expression among close candidates
                    winner_idx = max(face_scores, key=face_scores.get)
                    selection_method = "expression"
                    
                    # Fill in face scores for the rest of the group (for report consistency) 
                    for idx in group.index:
                        if idx not in face_scores:
                            expr_score, _ = get_face_expression_score(df.at[idx, 'path'])
                            face_scores[idx] = expr_score
            
            winner_name = df.at[winner_idx, 'filename']
            others = [df.at[idx, 'filename'] for idx in group.index if idx != winner_idx]
            
            if selection_method == "expression":
                print(f"  Burst {burst_count}: KEEP {winner_name} "
                      f"(sharp={df.at[winner_idx, 'blur_score']:.1f}, "
                      f"expr={face_scores[winner_idx]:.1f}) 😊, "
                      f"reject {', '.join(others)}")
            else:
                print(f"  Burst {burst_count}: KEEP {winner_name} "
                      f"(score={df.at[winner_idx, 'blur_score']:.1f}), "
                      f"reject {', '.join(others)}")
            
            # Mark others as rejected (duplicates) if not already blurry
            for idx in group.index:
                if idx != winner_idx:
                    if not df.at[idx, 'is_rejected']:
                        df.at[idx, 'is_rejected'] = True
                        if selection_method == "expression":
                            df.at[idx, 'reject_reason'] = (
                                f"Burst (expr={face_scores.get(idx, 0):.1f} "
                                f"< winner {face_scores.get(winner_idx, 0):.1f})"
                            )
                        else:
                            df.at[idx, 'reject_reason'] = "Duplicate (Burst)"
    
    if burst_count > 0:
        print(f"✅ Found {burst_count} burst group(s).")
    else:
        print("  No burst groups detected.")

    # 4. Final Pass: Culling
    rejects_dir = folder_path / "_Rejected"
    if df['is_rejected'].any():
        rejects_dir.mkdir(exist_ok=True)
        print(f"📦 Moving {df['is_rejected'].sum()} rejected files to {rejects_dir}...")

    final_results = []
    for _, row in df.iterrows():
        status = "KEEP"
        if row['is_rejected']:
            status = "REJECT"
            try:
                # Move JPG
                shutil.move(str(row['path']), str(rejects_dir / row['path'].name))
                # Move RAW if exists
                if row['raw_path']:
                    shutil.move(str(row['raw_path']), str(rejects_dir / row['raw_path'].name))
            except Exception as e:
                print(f"❌ Failed to move {row['filename']}: {e}")
        
        final_results.append({
            "Filename": row['filename'],
            "Status": status,
            "Reason": row['reject_reason'],
            "Blur Score": f"{row['blur_score']:.2f}",
            "Hash": row['phash'],
            "Has RAW": "Yes" if row['raw_path'] else "No"
        })

    # Save CSV
    csv_path = folder_path / "defects.csv"
    pd.DataFrame(final_results).to_csv(csv_path, index=False)
    print(f"📑 Report saved to: {csv_path}")
    print(f"Done! Keep: {(~df['is_rejected']).sum()}, Reject: {df['is_rejected'].sum()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Photo Defect Detector (Blur + Duplicates)")
    parser.add_argument("folder", help="Path to the camera folder")
    parser.add_argument("--blur", type=float, default=50.0, help="Sharpness threshold (default: 50)")
    args = parser.parse_args()
    
    folder_to_process = Path(args.folder)
    if folder_to_process.exists() and folder_to_process.is_dir():
        process_folder(folder_to_process, blur_threshold=args.blur)
    else:
        print(f"❌ Folder not found: {args.folder}")
