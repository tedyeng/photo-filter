# Photo-Filter CLI – Idea Draft

## Problem Statement
Develop a cross-platform CLI tool that automatically selects the best-composed and sharpest photo from a burst of Sony ARW (and other RAW) files, and moves it to a user-specified directory. The main goal is to reduce the time amateur travelers spend manually culling large numbers of photos.

How can we accurately segment different burst events from a folder containing mixed burst scenes, and automatically apply different scoring rules based on the main subject (portraits vs. landscapes/action) to select the best photo from each burst sequence?

## Target User & Motivation
- **User**: Amateur travelers
- **Motivation**: Travel photography often results in a massive amount of burst shots. Manually filtering them is time-consuming and prone to missing well-composed photos. A fast, reliable, and cross-platform CLI tool is needed to automate this task.

## Success Criteria
- **Similarity > 90%**: The automatically selected photos should have over 90% similarity in composition and sharpness compared to the best photos selected manually (verifiable via visual inspection or structural similarity tests).
- **Processing Time < 5s**: Processing 100 20-MP RAW/JPG images on a standard laptop should take no more than 5 seconds in total.

## Constraints & Technical Preferences
- Must support cross-platform execution on **macOS, Windows, and Linux**.
- **Python** is the language of choice.
- Avoid relying on cloud services; all analysis must be done locally.

## Existing Solutions & Gap
- **No prior use** of similar tools. While commercial software (like Adobe Lightroom, Capture One) offers auto-culling, they are GUI-based, require licenses, and cannot be used in CLI or automation scripts.
- There is a lack of a lightweight, scriptable, cross-platform **pure CLI** solution for this specific need.

## Why Now
- The popularization of travel and camera technology has caused an explosion in the volume of burst photos produced per trip, making the pain point of manual culling increasingly obvious.
- Existing open-source image processing and face detection models (like OpenCV YuNet) are mature enough, lowering the development barrier.

## Recommended Direction (MVP)
**Smart Burst Grouping + Rules Engine (The Smart Pipeline)**
1. **Language & Tech Stack**: Python as the core, utilizing `opencv-python`, `pillow`, `imagehash`, `tqdm`, and `numpy` for image analysis.
2. **Smart Grouping**: Extract EXIF capture timestamps to calculate time gaps (< 2s) combined with perceptual hash (pHash) distances to accurately segment photos into multiple independent burst events.
3. **Context-Aware Scoring**:
   - **Portraits**: If a face is detected using **YuNet**, evaluate the eyes open status (via landmarks), subject centering, and face size to calculate the portrait score.
   - **Landscapes**: If no face is detected, fall back to landscape mode, scoring purely based on global sharpness (Laplacian variance) and rule-of-thirds/visual balance.
   - Select the single best image from each burst group based on these scores.
   - Move the highest-scoring photo and its paired RAW file to the output folder. Move the remaining photos in the burst to a `_Rejected` folder.
4. **CLI Interface**:
   ```bash
   photofilter -i <burst_folder> -o <best_folder> [--threshold 0.55] [--env-file .env]
   ```
   - `-i`: Input burst folder.
   - `-o`: Destination folder for the best photos.
   - `--env-file`: Configuration file for adjusting burst thresholds and weights.
5. **Reporting**: Generate a `defects.csv` report containing a `Burst ID` column to map each file to its respective burst.
6. **Testing**: Add unit and integration tests for time-delta logic, scoring calculations, and file moving logic to ensure cross-platform consistency.

## Key Assumptions to Validate
- [x] **EXIF Reliability**: Camera-written EXIF timestamps (down to the second) are accurate enough to distinguish different burst events (with pHash as a secondary check).
- [x] **Landmarks Accuracy**: The 5 facial landmarks output by the lightweight YuNet model are sufficient for basic heuristic blink detection or centering evaluation.
- [x] **Performance Impact**: Reading EXIF data and computing landmarks still allows us to meet the 5-second performance target for 100 photos.

## Not Doing (and Why)
- **No real-time video stream support**: Currently only processing static files to avoid extra I/O complexity.
- **No cloud uploads**: Fits the "no cloud dependency" constraint.
- **No automatic conversion for non-RAW files**: RAW files are only paired and moved; conversion is left to the user.
- **No Deep Learning Aesthetics Models**: Doing so would slow down processing speed and bloat the CLI package, violating the lightweight, cross-platform goal.
- **No Image Merging/Stacking**: Attempting to merge multiple faces to fix blinking often creates ghosting artifacts, confusing the user more.
- **No Advanced Action Recognition**: We will not attempt to semantically understand moments like "the bat hitting the ball", relying strictly on edge sharpness (lack of motion blur) to capture the peak of the action.

## Open Questions & Decisions
- (Resolved) Should it support multi-language messages? -> **English only**.
- (Resolved) Should we provide a config file to customize weights? -> **Yes**, using a `.env` file to adjust weights (`WEIGHT_COMPOSITION`, `WEIGHT_SHARPNESS`) and `BURST_TIME_THRESHOLD`.
- (Resolved) Consider GPU acceleration? -> **Yes**, optional GPU acceleration if CUDA is available.
- (Resolved) What should the default burst time interval be? -> Based on timestamp analysis of example files, the interval within bursts is < 1s, and between different bursts is ~8s. Thus, the default threshold is set to **2.0 seconds**.
