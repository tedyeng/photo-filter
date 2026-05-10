# Implementation Plan: Photo-Filter CLI (Smart Burst Selection)

## Overview
Update the existing Photo-Filter CLI to support a "Smart Pipeline". The tool will accurately group photos into distinct burst events based on EXIF timestamps (< 2s interval) and perceptual hashes. It will then apply context-aware scoring (facial landmarks for portraits, rule-of-thirds/sharpness for landscapes) to pick the single best photo per burst event.

## Architecture Decisions
- **Smart Grouping**: Use EXIF timestamps as the primary grouping mechanism (default < 2s interval), combined with perceptual hashing to ensure scenes match.
- **Context-Aware Scoring**: Instead of a single scoring mechanism, detect faces first. If faces exist, score based on portrait metrics (eyes open, centering). If no faces, score based on landscape metrics (sharpness, balance).
- **Reporting**: Attribute every photo to a specific `Burst ID` in the generated `defects.csv`.

## Task List

### Phase 1: Foundations & Utilities

- [ ] **Task 1: Implement EXIF Time Extraction & Config updates**
  - **Description:** Create a utility to extract the original capture time from EXIF data. Update the config schema to accept `burst_time_threshold` (default 2).
  - **Acceptance criteria:**
    - [ ] `utils.get_exif_time(path)` returns a `datetime` object.
    - [ ] Config loading supports `burst_time_threshold`.
  - **Verify:** Run `pytest tests/unit/test_utils.py` covering EXIF extraction.
  - **Files likely touched:** `photofilter/utils.py`, `photofilter/config.py`, `tests/unit/test_utils.py`
  - **Estimated scope:** S

- [ ] **Task 2: Refactor Smart Grouping Logic**
  - **Description:** Implement the logic to group a list of image paths into sub-lists (bursts) based on time delta (< 2s) and pHash similarity.
  - **Acceptance criteria:**
    - [ ] `core.group_bursts(images, time_thresh, hash_thresh)` returns a list of burst groups (each group has a unique Burst ID).
    - [ ] Images further apart in time than `time_thresh` are split into different groups.
  - **Verify:** Run unit tests with mocked times and hashes.
  - **Files likely touched:** `photofilter/core.py`, `tests/unit/test_core_grouping.py`
  - **Estimated scope:** M

### Checkpoint: Foundation
- [ ] Grouping logic correctly splits the `raw_test` sample folder into multiple events.
- [ ] Unit tests pass.

### Phase 2: Context-Aware Scoring

- [ ] **Task 3: Landscape Scoring (Sharpness & Balance)**
  - **Description:** Refactor existing sharpness to be part of a `score_landscape(image_path)` function that also evaluates basic gradient symmetry/rule-of-thirds if possible.
  - **Acceptance criteria:**
    - [ ] Returns a 0-1 score representing landscape quality.
  - **Verify:** Unit test on landscape images (no faces).
  - **Files likely touched:** `photofilter/core.py`, `tests/unit/test_core_landscape.py`
  - **Estimated scope:** S

- [ ] **Task 4: Portrait Scoring (YuNet Landmarks)**
  - **Description:** Update YuNet usage to extract the 5 facial landmarks. Calculate a portrait score based on face size, centering, and heuristic "eyes open" estimation if feasible from landmarks.
  - **Acceptance criteria:**
    - [ ] `score_portrait(image_path, faces)` returns a 0-1 score.
  - **Verify:** Unit test on portrait images with faces.
  - **Files likely touched:** `photofilter/core.py`, `tests/unit/test_core_portrait.py`
  - **Estimated scope:** M

### Checkpoint: Scoring
- [ ] Both portrait and landscape scoring functions run without crashing and return expected ranges.

### Phase 3: Core Pipeline

- [ ] **Task 5: End-to-End Folder Processing**
  - **Description:** Rewrite `process_folder` to use the new `group_bursts` logic. Iterate over each Burst ID, apply `score_portrait` or `score_landscape` based on face detection, pick the best photo per Burst ID, move files, and generate the updated `defects.csv`.
  - **Acceptance criteria:**
    - [ ] Multiple best photos are selected (one per burst group).
    - [ ] `defects.csv` includes a `Burst ID` column.
    - [ ] `_Rejected` folder populated correctly.
  - **Verify:** Run integration tests over the `raw_test` folder. Check CSV output.
  - **Files likely touched:** `photofilter/core.py`, `photofilter/cli.py`, `tests/integration/test_process_folder.py`
  - **Estimated scope:** L

### Checkpoint: Complete Pipeline
- [ ] Integration tests pass.
- [ ] End-to-end execution on `raw_test` finishes in < 5 seconds and perfectly segregates bursts.

### Phase 4: Polish

- [ ] **Task 6: Documentation & CI**
  - **Description:** Update README to explain the new smart pipeline, burst IDs, and context-aware scoring. Ensure CI matrix runs all new tests.
  - **Acceptance criteria:**
    - [ ] `README.md` reflects new behaviour.
    - [ ] GitHub Actions CI is green.
  - **Verify:** Manual review of docs.
  - **Files likely touched:** `README.md`, `.github/workflows/ci.yml`
  - **Estimated scope:** S

- [ ] **Task 7: Progress Bar & Execution Time**
  - **Description:** Add `tqdm` to dependencies. Wrap the main image processing loops in `core.process_folder` with a progress bar. Track and print the total execution time at the end.
  - **Acceptance criteria:**
    - [ ] A progress bar is visible during processing.
    - [ ] Total elapsed time is printed after completion.
  - **Verify:** Run the CLI manually on `raw_test` and observe standard output.
  - **Files likely touched:** `pyproject.toml`, `photofilter/core.py`, `photofilter/cli.py`
  - **Estimated scope:** S

## Risks and Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| EXIF time missing | Medium | Fallback to file modification time (`os.path.getmtime`). |
| YuNet landmarks insufficient for blink detection | Low | Use basic face centering and size as primary portrait score, falling back gracefully if blink detection heuristics fail. |
| Memory usage spikes with many bursts | Medium | Ensure images are unloaded after scoring, maintaining only metadata in memory. |

## Open Questions
- None.
