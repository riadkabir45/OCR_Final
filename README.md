# OCR Dataset Processing Pipeline

Complete workflow for generating damaged OCR datasets and preprocessing them.

## Overview

This pipeline takes raw OCR dataset images with JSON annotations and:
1. Generates damaged versions with realistic visual artifacts
2. Preprocesses the damaged images using background subtraction and enhancement

## Directory Structure

```
OCR_Final/
├── dataset/
│   ├── raw/              # Original images + JSON annotations
│   ├── damaged/          # Generated damaged images + updated JSONs
│   └── preprocessed/     # Preprocessed damaged images + JSONs
├── scripts/
│   ├── generate_damaged_dataset.py
│   └── preprocess_damaged.py
├── processor.ref         # Reference preprocessing implementation
└── README.md            # This file
```

## Step 1: Generate Damaged Dataset

Creates damaged versions of raw images with two types of realistic artifacts:

- **Irregular spot**: Near-round dark gradient blob (~5% of image area, near-black center fading to brownish edges)
- **Distortion area**: Localized horizontal distortion (3-10% of image area)

The number of damages per image is random (0 to `--max-damages`), and damage types are randomly selected.

### Usage

```bash
# Generate damaged dataset with up to 10 damage instances per image
python3 scripts/generate_damaged_dataset.py \
  --input-dir dataset/raw \
  --output-dir dataset/damaged \
  --max-damages 10

# Quick test on 3 files with reproducible results
python3 scripts/generate_damaged_dataset.py \
  --input-dir dataset/raw \
  --output-dir dataset/damaged \
  --max-damages 10 \
  --limit 3 \
  --seed 42

# Overwrite existing outputs
python3 scripts/generate_damaged_dataset.py \
  --input-dir dataset/raw \
  --output-dir dataset/damaged \
  --max-damages 10 \
  --overwrite
```

### Options

- `--input-dir`: Source folder with raw images and JSON files (default: `dataset/raw`)
- `--output-dir`: Output folder for damaged images and JSONs (default: `dataset/damaged`)
- `--max-damages`: Maximum number of damage instances per image (default: 3)
- `--limit`: Process only first N files (default: 0 = all files)
- `--seed`: Random seed for reproducibility (optional)
- `--overwrite`: Overwrite existing output files

### Output

Each processed image pair produces:
- Damaged image with same filename
- JSON file with:
  - Original annotation data (shapes, imageHeight, imageWidth)
  - New `damage` field containing metadata about applied damages
  - Updated `imagePath`

Example damage metadata:
```json
{
  "damage": {
    "applied": [
      {
        "type": "spot",
        "center": [640, 480],
        "radius": 85,
        "intensity": 0.92
      },
      {
        "type": "distortion",
        "box": [120, 200, 380, 350],
        "max_shift": 15
      }
    ]
  }
}
```

### Dependencies

```bash
pip install pillow numpy
```

## Step 2: Preprocess Damaged Dataset

Applies the preprocessing pipeline from `processor.ref` to damaged images:

1. **Background subtraction**: Per-channel dilation → median blur → difference → normalization
2. **Enhancement adjustments**:
   - Brightness: 0.7
   - Sharpness: 1.0
   - Contrast: 255
   - Color: 1.0

### Usage

```bash
# Preprocess all damaged images
python3 scripts/preprocess_damaged.py \
  --input-dir dataset/damaged \
  --output-dir dataset/preprocessed

# Test on 5 files
python3 scripts/preprocess_damaged.py \
  --input-dir dataset/damaged \
  --output-dir dataset/preprocessed \
  --limit 5

# Overwrite existing outputs
python3 scripts/preprocess_damaged.py \
  --input-dir dataset/damaged \
  --output-dir dataset/preprocessed \
  --overwrite
```

### Options

- `--input-dir`: Source folder with damaged images and JSONs (default: `dataset/damaged`)
- `--output-dir`: Output folder for preprocessed images and JSONs (default: `dataset/preprocessed`)
- `--limit`: Process only first N files (default: 0 = all files)
- `--overwrite`: Overwrite existing output files

### Output

- Preprocessed images with same filenames
- JSON files copied with updated `imagePath`

### Dependencies

```bash
pip install opencv-python pillow numpy
```

## Complete Workflow Example

```bash
# 1. Install dependencies
pip install pillow numpy opencv-python

# 2. Generate damaged dataset (up to 10 damages per image)
python3 scripts/generate_damaged_dataset.py \
  --input-dir dataset/raw \
  --output-dir dataset/damaged \
  --max-damages 10

# 3. Preprocess damaged images
python3 scripts/preprocess_damaged.py \
  --input-dir dataset/damaged \
  --output-dir dataset/preprocessed

# Final output: dataset/preprocessed/ contains preprocessed damaged images ready for OCR training
```

## Quick Test Workflow

```bash
# Test with 3 files
python3 scripts/generate_damaged_dataset.py \
  --input-dir dataset/raw \
  --output-dir dataset/test_damaged \
  --max-damages 5 \
  --limit 3 \
  --seed 42

python3 scripts/preprocess_damaged.py \
  --input-dir dataset/test_damaged \
  --output-dir dataset/test_preprocessed \
  --limit 3
```

## Notes

- All scripts preserve the original annotation coordinates (no geometric transforms like rotation)
- JSON files maintain all original annotation data including bounding boxes and labels
- The damage generator uses gradient blobs instead of solid rectangles for natural appearance
- Preprocessing enhances text visibility against damaged backgrounds
