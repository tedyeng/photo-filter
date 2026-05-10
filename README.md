# 📷 Photo-Filter CLI

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A cross-platform, high-performance command-line tool designed to help amateur photographers and travelers quickly cull massive amounts of continuous burst photos. 

It automatically groups your photos into distinct burst events and uses AI-assisted, context-aware scoring to pick the absolute best shot from each sequence, saving you hours of manual culling.

---

## ✨ Key Features

- **🧠 Smart Burst Grouping:** Dump all your photos from different scenes into a single folder. The CLI automatically segments them into distinct "burst events" by analyzing EXIF capture timestamps (default `< 2s` apart) and perceptual hash (pHash) similarity.
- **🎭 Context-Aware Scoring:** 
  - **Portrait Mode:** Automatically detects faces (via OpenCV YuNet). If faces are present, it scores the photo based on subject centering, face size, and clarity.
  - **Landscape/Action Mode:** If no faces are detected, it falls back to evaluating global edge sharpness (Laplacian variance) and visual balance.
- **📁 Automatic RAW Pairing:** When the best JPEG is selected, its corresponding RAW file (e.g., `.ARW`, `.CR3`, `.NEF`) is automatically identified and moved with it.
- **📊 Detailed Reporting:** Generates a comprehensive `defects.csv` report mapping every processed file to its specific `Burst ID`, score, and the reason it was kept or rejected.
- **⚡ GPU Acceleration:** Optional CUDA support for lightning-fast face detection when processing thousands of photos.

---

## 🚀 Installation

This project uses **[uv](https://github.com/astral-sh/uv)** for fast, reliable dependency management.

### 1. Prerequisites
- Python 3.9 or higher.
- `uv` installed on your system.

### 2. Setup the Environment
Clone the repository and set up the virtual environment:

```bash
git clone https://github.com/yourusername/photo-filter.git
cd photo-filter

# Create and activate a virtual environment
uv venv .venv
source .venv/bin/activate   # macOS / Linux
# .\.venv\Scripts\Activate.ps1 # Windows PowerShell

# Install the project and its dependencies
uv pip install -e .
```

---

## 💻 Usage

Once installed, you can run the CLI tool from anywhere within your environment:

```bash
photofilter -i <input_burst_folder> -o <best_photos_folder> [--threshold 0.55] [--env-file .env] [--gpu]
```

### Arguments

| Argument | Short | Description | Default |
| :--- | :---: | :--- | :--- |
| `--input` | `-i` | **(Required)** Path to the folder containing your burst images (JPG + RAW). | None |
| `--output` | `-o` | **(Required)** Destination folder for the selected "best" photos. | None |
| `--threshold` | | Minimum composite score (0.0 to 1.0) to accept a photo. | `0.55` |
| `--env-file` | | Path to a `.env` configuration file for overriding weights and thresholds. | None |
| `--gpu` | | Enable GPU acceleration if CUDA is available on your system. | `False` |

### What Happens When You Run It?

1. The tool scans your `--input` directory.
2. It groups photos into bursts (Burst 1, Burst 2, etc.).
3. It selects the single highest-scoring photo per burst.
4. The winning JPG (and its RAW pair) is moved to the `--output` folder.
5. All losing photos (and their RAW pairs) are moved to `<input_folder>/_Rejected/`.
6. A `defects.csv` is created in the input folder.

---

## ⚙️ Configuration (`.env`)

You can customize the engine's behavior by creating a `.env` file. We provide a `.env.example` to get you started.

```bash
cp .env.example .env
```

**Available Settings:**

```env
# Time gap (in seconds) that separates one burst event from another.
# If two photos are taken 2.5 seconds apart, they belong to different bursts.
BURST_TIME_THRESHOLD=2.0

# Scoring Weights (must sum up to ~1.0 theoretically, though algorithm normalizes internally)
# How much importance to place on Face detection vs Sharpness.
WEIGHT_COMPOSITION=0.6
WEIGHT_SHARPNESS=0.4

# Hardware Acceleration flag (true/false)
GPU_ENABLED=true
```

---

## 📋 Example Report (`defects.csv`)

After processing, you can review the AI's decisions in the generated CSV file:

| Filename | Burst ID | Status | Reason | Has Face | Portrait Score | Landscape Score | Composite | Hash | Has RAW |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| A7C08850.jpg | 1 | REJECT | Not best | Yes | 0.85 | 0.00 | 0.85 | f87e... | Yes |
| A7C08851.jpg | 1 | KEEP | | Yes | 0.92 | 0.00 | 0.92 | f87e... | Yes |
| A7C08863.jpg | 2 | KEEP | | No | 0.00 | 0.78 | 0.78 | a12b... | Yes |
| A7C08864.jpg | 2 | REJECT | Not best | No | 0.00 | 0.65 | 0.65 | a12b... | Yes |

---

## 🛠️ Development & Testing

To run the test suite (Unit + Integration tests):

```bash
# Ensure you have installed the testing dependencies
uv pip install pytest pytest-cov

# Run all tests
uv run pytest tests/
```

---
*Built to bring your best memories into focus.*