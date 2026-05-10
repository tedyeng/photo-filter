# Spec: Photo‑Filter CLI - Smart Burst Selection

## Objective
Develop a cross‑platform command‑line tool that automatically selects the best photo from multiple bursts of Sony ARW (and other RAW) files mixed in a single folder. The tool uses a "Smart Pipeline" to accurately group photos into distinct burst events based on EXIF timestamps (< 2s interval) and perceptual hashes. It then applies context-aware scoring—using facial landmarks for portraits and rule-of-thirds/sharpness for landscapes—to pick the single best photo per burst event, moving the winners to a specified output directory.

## Tech Stack
- **Language**: Python (≥3.9)
- **Key Libraries**:
  - `opencv-python` (image processing, Laplacian sharpness, YuNet face detection & landmarks)
  - `pillow` (image I/O, EXIF extraction)
  - `imagehash` (perceptual hashing for grouping validation)
  - `numpy`
  - `pandas` (CSV report generation)
- **Optional GPU**: CUDA‑enabled OpenCV for accelerated processing when available.

## Commands
```bash
# Main CLI
photofilter -i <burst_folder> -o <best_folder> [--threshold 0.55]

# Flags
-i, --input      Path to the folder containing the burst images (JPG + RAW)
-o, --output     Destination folder for the selected best photos
--threshold      Minimum composite score to accept a photo (default 0.55)
--config <file>   Path to JSON config to customise weights, burst time threshold, GPU usage, etc.

# Helper scripts (optional)
python -m photofilter.generate_report   # Generates defects.csv report
```

## Project Structure
```text
photo-filter/
├── photofilter/            # Python package
│   ├── __init__.py
│   ├── cli.py             # Argument parsing, entry point
│   ├── core.py            # Context-aware scoring, EXIF parsing, smart burst grouping
│   ├── utils.py           # Helper functions (hashing, file moves, time diff)
│   └── config.yaml        # Default configuration (weights, thresholds)
├── docs/
│   ├── burst-selection.md # Refined idea specification (source of truth)
│   ├── idea.md            # Original idea draft
│   ├── plan.md            # Implementation plan
│   └── spec.md            # This specification file (tracked in VCS)
├── tests/
│   ├── unit/              # Unit tests for core functions
│   └── integration/       # End‑to‑end tests using sample bursts
├── requirements.txt       # Fixed dependency versions
└── README.md
```

## Code Style
- **PEP 8** compliant, use `black` for formatting.
- Type hints on all public functions (`def func(arg: str) -> float:`).
- Logging via the standard library `logging` module, default level `INFO`.
- Example snippet (showing context-aware scoring):
```python
def compute_composition(image_path: Path) -> float:
    """Compute context-aware composition score (0-1)."""
    faces = detect_faces(image_path)
    if faces:
        return score_portrait(faces)
    else:
        return score_landscape(image_path)
```

## Testing Strategy
- **Framework**: `pytest` + `pytest-cov`.
- **Unit tests** cover:
  - EXIF time extraction and time-delta calculation.
  - Smart grouping logic (Time < 2s + pHash).
  - Context-aware composition scoring (mocked face landmarks vs landscape gradients).
- **Integration tests** run the full CLI on a sample folder with multiple mixed bursts and verify:
  - Correct number of winners are moved to output (one per burst group).
  - `_Rejected` folder contains expected duplicates.
  - `defects.csv` contains the new `Burst ID` column.
- **Coverage**: ≥ 90 % for core module.
- **CI**: GitHub Actions run `pytest` on macOS, Windows, Linux matrices.

## Boundaries
- **Always**:
  - Run the test suite before any commit (`pre‑commit` hook).
  - Group bursts based on EXIF time before applying image hash similarity.
  - Log each decision (why a photo was rejected and which burst it belongs to).
- **Ask first**:
  - Change default time interval (2 seconds) or weighting of portrait vs landscape.
  - Introduce new deep-learning models (e.g., for aesthetics).
- **Never**:
  - Perform image synthesis/merging to fix blinking eyes.
  - Delete user photos without moving them to `_Rejected`.
  - Process entire folders as a single burst.

## Success Criteria
- **Functional**: The CLI accurately separates a mixed folder of continuous shots into distinct burst events and selects the single best photo per event.
- **Performance**: Total runtime < 5 seconds on a typical laptop (no GPU) for 100 images.
- **Reporting**: The output `defects.csv` correctly attributes every photo to a specific `Burst ID`.
- **Cross‑platform**: Passes CI tests on macOS, Windows, and Linux.

## Open Questions
- None (Default burst interval is set to 2 seconds based on timestamp analysis, configurable in `config.json`).
