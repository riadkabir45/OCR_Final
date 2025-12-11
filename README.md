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

## Step 3: Organize Dataset into Splits

Structures the preprocessed dataset into train/val/test splits with a manifest file.

### Usage

```bash
# Create organized dataset with default splits (70% train, 15% val, 15% test)
python3 scripts/organize_dataset.py \
  --input-dir dataset/preprocessed \
  --output-dir dataset/organized

# Custom splits (60% train, 20% val, 20% test)
python3 scripts/organize_dataset.py \
  --input-dir dataset/preprocessed \
  --output-dir dataset/organized \
  --train-split 0.6 \
  --val-split 0.2
```

### Options

- `--input-dir`: Source folder with preprocessed images and JSONs (default: `dataset/preprocessed`)
- `--output-dir`: Output folder for organized dataset (default: `dataset/organized`)
- `--train-split`: Fraction for training set (default: 0.7)
- `--val-split`: Fraction for validation set (default: 0.15, remainder is test)
- `--seed`: Random seed for reproducibility (default: 42)

### Output

Creates directory structure:
```
dataset/organized/
├── train/           # Training samples
│   ├── 0_21_0.jpg
│   ├── 0_21_0.json
│   └── ...
├── val/             # Validation samples
│   └── ...
├── test/            # Test samples
│   └── ...
└── manifest.json    # Metadata about all samples
```

The manifest file contains split information and metadata for each sample.

## Step 5: Clean Dataset

Removes text shapes (bounding boxes) that overlap with damaged areas, while **keeping the damage metadata**.
This allows models to:
- Learn to detect damage regions (using damage boxes)
- Not train on indistinguishable text in damaged areas (text boxes removed)

### Usage

```bash
# Remove text shapes in damaged areas, keep damage metadata
python3 scripts/clean_dataset.py \
  --input-dir dataset/organized \
  --output-dir dataset/cleaned

# Only mark text shapes as damaged (keep them with 'damaged' flag)
python3 scripts/clean_dataset.py \
  --input-dir dataset/organized \
  --output-dir dataset/cleaned \
  --mode mark

# Process only training set with custom threshold (40% overlap = remove)
python3 scripts/clean_dataset.py \
  --input-dir dataset/organized \
  --output-dir dataset/cleaned \
  --split train \
  --threshold 0.4
```

### Options

- `--input-dir`: Path to organized dataset (default: `dataset/organized`)
- `--output-dir`: Path to write cleaned dataset (default: `dataset/cleaned`)
- `--mode`: `remove` to delete text shapes in damage areas (default), `mark` to add `"damaged"` flag
- `--threshold`: Intersection threshold 0.0-1.0 (default: 0.25, meaning ≥25% overlap = remove text)
- `--split`: Process specific split: train/val/test/all (default: all)

### Output

Creates cleaned dataset preserving damage metadata:
```
dataset/cleaned/
├── train/
│   ├── image.jpg
│   ├── image.json  # Text boxes in damage areas removed, damage info kept
│   └── ...
├── val/
└── test/
```

Each JSON includes:
- All original **damage** metadata (boxes for spots/distortions)
- Filtered **shapes** (text boxes not overlapping damage removed)
- **cleaning** metadata showing what was removed

Example:
```json
{
  "damage": {
    "applied": [
      {"type": "spot", "center": [...], "radius": 85},
      {"type": "distortion", "box": [...]}
    ]
  },
  "shapes": [
    {"label": "text1", "points": [...]},
    {"label": "text3", "points": [...]}
  ],
  "cleaning": {
    "mode": "remove",
    "threshold": 0.25,
    "original_shapes": 5,
    "cleaned_shapes": 3,
    "removed": 2
  }
}
```

## Step 6: Visualize Cleaned Dataset

Visualize the cleaned dataset to verify damaged shapes were removed:

```bash
python3 scripts/visualize_dataset.py \
  --dataset-dir dataset/cleaned \
  --split train \
  --num-samples 5 \
  --output-dir visualizations_cleaned/
```

Now the visualization shows no boxes for shapes that were in damaged areas.

### Usage

```bash
# Visualize a random sample from training set
python3 scripts/visualize_dataset.py \
  --dataset-dir dataset/organized \
  --split train

# Visualize specific sample
python3 scripts/visualize_dataset.py \
  --dataset-dir dataset/organized \
  --split train \
  --sample-id 0_21_0

# Visualize multiple random samples and save to folder
python3 scripts/visualize_dataset.py \
  --dataset-dir dataset/organized \
  --split train \
  --num-samples 5 \
  --output-dir visualizations/

# Visualize from validation set
python3 scripts/visualize_dataset.py \
  --dataset-dir dataset/organized \
  --split val
```

### Options

- `--dataset-dir`: Path to organized dataset (default: `dataset/organized`)
- `--split`: Dataset split to visualize: train/val/test (default: train)
- `--sample-id`: Specific sample ID (random if not specified)
- `--num-samples`: Number of random samples to visualize (default: 1)
- `--output-dir`: Save visualizations to this directory (display if not specified)

### Output

Visualizations show:
- Image with color-coded bounding boxes for each shape
- Text labels for each shape
- Damage type information (spot, distortion) if present
- Image dimensions and number of shapes
- Occlusion flags if applicable

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

# 4. Organize into train/val/test splits
python3 scripts/organize_dataset.py \
  --input-dir dataset/preprocessed \
  --output-dir dataset/organized

# 5. Visualize samples (optional)
python3 scripts/visualize_dataset.py \
  --dataset-dir dataset/organized \
  --split train \
  --num-samples 5 \
  --output-dir visualizations/

# Final output: dataset/organized/ contains structured dataset ready for OCR training
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
