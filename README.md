# OCR Dataset Processing Pipeline

Complete workflow for generating damaged OCR datasets and preprocessing them.

## Overview

This pipeline takes raw OCR dataset images with JSON annotations and:

1. Generates damaged versions with realistic visual artifacts

2. Preprocesses the damaged images using background subtraction and enhancement

## Directory Structure

```bash
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
- `--reduce-damage-boxes`: Reduce damage boxes to visible area using grayscale difference
- `--diff-threshold`: Grayscale abs-diff threshold (0-255) to consider a pixel damaged (default: 20)
- `--min-damage-pixels`: Minimum count of above-threshold pixels within a box to shrink it (default: 32)

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

## Step 2: Visualize Dataset

Visualize dataset samples with bounding boxes and damage overlays to inspect the data.

### Color Modes

- **labels** (default): Text/shape boxes in red, damage regions in orange
- **classic**: Text/shape boxes in green, damage regions in yellow

### Usage Examples

```bash
# Visualize a specific sample
python3 scripts/visualize_dataset.py \
  --dataset-dir dataset/damaged \
  --sample-id 101_21_0

# Visualize all samples and save to folder
python3 scripts/visualize_dataset.py \
  --dataset-dir dataset/damaged \
  --output-dir out_vis \
  --color-mode labels \
  --thickness 3

# Use classic colors with thicker lines
python3 scripts/visualize_dataset.py \
  --dataset-dir dataset/damaged \
  --output-dir out_vis \
  --color-mode classic \
  --thickness 3

# Draw only damage boxes
python3 scripts/visualize_dataset.py \
  --dataset-dir dataset/damaged \
  --output-dir out_vis \
  --boxes damage

# Draw only text boxes
python3 scripts/visualize_dataset.py \
  --dataset-dir dataset/damaged \
  --output-dir out_vis \
  --boxes text

# Draw no boxes (metadata overlays only)
python3 scripts/visualize_dataset.py \
  --dataset-dir dataset/damaged \
  --output-dir out_vis \
  --boxes none
```

### Options

- `--dataset-dir`: Path to dataset (default: `dataset/cleaned`)
- `--sample-id`: Specific sample ID to visualize (if not specified, visualizes all samples)
- `--output-dir`: Save visualizations to this directory (display if not specified)
- `--color-mode`: Color scheme: `labels` (default) or `classic`
- `--thickness`: Outline thickness for boxes (default: 2)
- `--boxes`: Which boxes to draw: `both` (default), `text`, `damage`, or `none`

### Output

Visualizations show:
- Image with color-coded bounding boxes for each shape
- Text labels for each shape
- Damage type information (spot, distortion) if present
- Image dimensions and number of shapes
- Occlusion flags if applicable

## Step 3: Preprocess Damaged Dataset


## Step 3: Clean Dataset

Removes text shapes (bounding boxes) that overlap with damaged areas, while **keeping the damage metadata**.
This allows models to:
- Learn to detect damage regions (using damage boxes)
- Not train on indistinguishable text in damaged areas (text boxes removed)

### Usage

```bash
# Remove text shapes in damaged areas, keep damage metadata
python3 scripts/clean_dataset.py \
  --input-dir dataset/preprocessed \
  --output-dir dataset/cleaned

# Only mark text shapes as damaged (keep them with 'damaged' flag)
python3 scripts/clean_dataset.py \
  --input-dir dataset/preprocessed \
  --output-dir dataset/cleaned \
  --mode mark

# Custom threshold (40% overlap = remove)
python3 scripts/clean_dataset.py \
  --input-dir dataset/preprocessed \
  --output-dir dataset/cleaned \
  --threshold 0.4
```

### Options

- `--input-dir`: Path to preprocessed dataset (default: `dataset/preprocessed`)
- `--output-dir`: Path to write cleaned dataset (default: `dataset/cleaned`)
- `--mode`: `remove` to delete text shapes in damage areas (default), `mark` to add `"damaged"` flag
- `--threshold`: Intersection threshold 0.0-1.0 (default: 0.25, meaning ≥25% overlap = remove text)

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

## Step 4: Organize Dataset


Removes text shapes (bounding boxes) that overlap with damaged areas, while **keeping the damage metadata**.
This allows models to:
- Learn to detect damage regions (using damage boxes)
- Not train on indistinguishable text in damaged areas (text boxes removed)

### Usage

```bash
# Remove text shapes in damaged areas, keep damage metadata
python3 scripts/clean_dataset.py \
  --input-dir dataset/preprocessed \
  --output-dir dataset/cleaned

# Only mark text shapes as damaged (keep them with 'damaged' flag)
python3 scripts/clean_dataset.py \
  --input-dir dataset/preprocessed \
  --output-dir dataset/cleaned \
  --mode mark

# Custom threshold (40% overlap = remove)
python3 scripts/clean_dataset.py \
  --input-dir dataset/preprocessed \
  --output-dir dataset/cleaned \
  --threshold 0.4
```

### Options

- `--input-dir`: Path to preprocessed dataset (default: `dataset/preprocessed`)
- `--output-dir`: Path to write cleaned dataset (default: `dataset/cleaned`)
- `--mode`: `remove` to delete text shapes in damage areas (default), `mark` to add `"damaged"` flag
- `--threshold`: Intersection threshold 0.0-1.0 (default: 0.25, meaning ≥25% overlap = remove text)

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

## Step 4: Visualize Cleaned Dataset

Visualize the cleaned dataset to verify damaged shapes were removed:

```bash
python3 scripts/visualize_dataset.py \
  --dataset-dir dataset/cleaned \
  --num-samples 5 \
  --output-dir visualizations_cleaned/
```

### Usage

```bash
# Visualize a random sample
python3 scripts/visualize_dataset.py \
  --dataset-dir dataset/cleaned

# Visualize specific sample
python3 scripts/visualize_dataset.py \
  --dataset-dir dataset/cleaned \
  --sample-id 0_21_0

# Visualize multiple random samples and save to folder
python3 scripts/visualize_dataset.py \
  --dataset-dir dataset/cleaned \
  --num-samples 5 \
  --output-dir visualizations/
```

### Options

- `--dataset-dir`: Path to dataset (default: `dataset/cleaned`)
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

# 4. Clean dataset (remove text boxes in damaged areas)
python3 scripts/clean_dataset.py \
  --input-dir dataset/preprocessed \
  --output-dir dataset/cleaned

# 5. Visualize samples (optional)
python3 scripts/visualize_dataset.py \
  --dataset-dir dataset/cleaned \
  --num-samples 5 \
  --output-dir visualizations/

# Note: Use organize_dataset.py at the end of your project when ready to split into train/val/test

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
