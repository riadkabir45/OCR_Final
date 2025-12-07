#!/usr/bin/env python3
"""
Preprocess damaged dataset using the reference processor from processor.ref.

Applies the improveImage + adjust pipeline to all images in the damaged dataset
and writes the preprocessed images to an output directory.
"""
import os
import sys
import json
import argparse
import cv2
import numpy as np
from PIL import Image, ImageEnhance


def change_contrast(img, level):
    """Change contrast of a PIL Image."""
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)


def change_sharpness(img, level):
    """Change sharpness of a PIL Image."""
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(level)
    return img


def adjust(img, blevel, slevel, clevel, colevel):
    """Apply brightness, sharpness, contrast, and color adjustments."""
    # brightness
    benhance = ImageEnhance.Brightness(img)
    img = benhance.enhance(blevel)
    # sharpness
    img = change_sharpness(img, slevel)
    # contrast
    img = change_contrast(img, clevel)
    # color
    cenhance = ImageEnhance.Color(img)
    img = cenhance.enhance(colevel)
    return img


def improveImage(img_dir):
    """Background subtraction and normalization using OpenCV."""
    img = cv2.imread(img_dir, -1)
    if img is None:
        raise ValueError(f"Cannot read image: {img_dir}")
    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0,
                                 beta=255, norm_type=cv2.NORM_MINMAX,
                                 dtype=cv2.CV_8UC1)

        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    return result_norm


def convert_image(image_dir):
    """Full preprocessing pipeline from processor.ref."""
    img = improveImage(image_dir)
    img = adjust(Image.fromarray(img),
                 blevel=0.7, slevel=1, clevel=255,
                 colevel=1)
    return img


def process_file(json_path: str, input_dir: str, output_dir: str, overwrite: bool = False) -> None:
    """Process one JSON+image pair: preprocess image and copy JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    img_name = meta.get("imagePath")
    if not img_name:
        print(f"No imagePath in {json_path}, skipping")
        return

    img_path = os.path.join(input_dir, img_name)
    if not os.path.exists(img_path):
        alt = os.path.join(os.path.dirname(json_path), img_name)
        if os.path.exists(alt):
            img_path = alt
        else:
            print(f"Image not found for {json_path}: {img_path}")
            return

    out_img_path = os.path.join(output_dir, img_name)
    out_json_path = os.path.join(output_dir, os.path.basename(json_path))

    if os.path.exists(out_img_path) and not overwrite:
        print(f"Skipping existing: {out_img_path}")
        return

    # Apply preprocessing
    preprocessed = convert_image(img_path)
    preprocessed.save(out_img_path)

    # Copy JSON with updated imagePath
    meta_out = dict(meta)
    meta_out["imagePath"] = os.path.basename(out_img_path)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)

    print(f"Preprocessed: {out_img_path}, {out_json_path}")


def main():
    p = argparse.ArgumentParser(description="Preprocess damaged dataset using processor.ref pipeline")
    p.add_argument("--input-dir", default="dataset/damaged", help="Path to damaged dataset folder")
    p.add_argument("--output-dir", default="dataset/preprocessed", help="Path to write preprocessed images and JSONs")
    p.add_argument("--limit", type=int, default=0, help="Limit number of files processed (0 = all)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = p.parse_args()

    # collect json files
    files = []
    for fname in os.listdir(args.input_dir):
        if fname.lower().endswith(".json"):
            files.append(os.path.join(args.input_dir, fname))
    files.sort()
    if args.limit and args.limit > 0:
        files = files[:args.limit]

    if not files:
        print("No JSON files found in", args.input_dir)
        return

    os.makedirs(args.output_dir, exist_ok=True)

    for j in files:
        try:
            process_file(j, args.input_dir, args.output_dir, overwrite=args.overwrite)
        except Exception as e:
            print(f"Error processing {j}: {e}")


if __name__ == "__main__":
    main()
