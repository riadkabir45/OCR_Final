#!/usr/bin/env python3
"""
Clean dataset by removing text shapes that overlap with damaged areas.

Removes shapes (text bounding boxes) that are significantly damaged/occluded,
but KEEPS the damage metadata so models can learn to detect damage regions.
The model will see damage boxes but not text boxes within those regions,
making text in damaged areas indistinguishable for training purposes.
"""
import os
import sys
import json
import argparse
import shutil
from pathlib import Path


def bbox_intersection_area(b1, b2):
    """Calculate intersection area between two bounding boxes."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def is_shape_damaged(shape, damage_boxes, threshold=0.25):
    """Check if a shape is significantly damaged/occluded."""
    pts = shape.get("points", [])
    if len(pts) < 2:
        return False
    
    x1 = min(pts[0][0], pts[1][0])
    y1 = min(pts[0][1], pts[1][1])
    x2 = max(pts[0][0], pts[1][0])
    y2 = max(pts[0][1], pts[1][1])
    
    shape_bbox = (x1, y1, x2, y2)
    shape_area = (x2 - x1) * (y2 - y1)
    
    if shape_area <= 0:
        return False
    
    # Check intersection with all damage areas
    for dmg_box in damage_boxes:
        inter = bbox_intersection_area(shape_bbox, dmg_box)
        if inter / shape_area >= threshold:
            return True
    
    return False


def clean_json(json_path: str, removal_mode: str = "remove", threshold: float = 0.25) -> dict:
    """
    Clean a JSON file by handling damaged shapes.
    
    Args:
        json_path: Path to JSON annotation file
        removal_mode: "remove" to delete damaged shapes, "mark" to add "damaged" flag
        threshold: Intersection threshold (0.0-1.0) to consider shape damaged
    
    Returns:
        Updated metadata dict
    """
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    # Extract damage boxes
    damage = meta.get("damage", {})
    applied_damages = damage.get("applied", [])
    damage_boxes = []
    
    for dmg in applied_damages:
        dmg_type = dmg.get("type")
        
        if dmg_type == "spot":
            center = dmg.get("center", [0, 0])
            radius = dmg.get("radius", 0)
            cx, cy = center[0], center[1]
            damage_boxes.append((cx - radius, cy - radius, cx + radius, cy + radius))
            
        elif dmg_type == "distortion":
            box = dmg.get("box", [0, 0, 0, 0])
            damage_boxes.append(tuple(box))
    
    # Process shapes
    shapes = meta.get("shapes", [])
    original_count = len(shapes)
    
    if removal_mode == "remove":
        # Filter out damaged shapes
        cleaned_shapes = [
            s for s in shapes
            if not is_shape_damaged(s, damage_boxes, threshold)
        ]
    elif removal_mode == "mark":
        # Mark damaged shapes but keep them
        cleaned_shapes = []
        for s in shapes:
            if is_shape_damaged(s, damage_boxes, threshold):
                s["damaged"] = True
            cleaned_shapes.append(s)
    else:
        raise ValueError(f"Unknown removal_mode: {removal_mode}")
    
    meta["shapes"] = cleaned_shapes
    
    # Add cleaning metadata
    if "cleaning" not in meta:
        meta["cleaning"] = {}
    meta["cleaning"]["mode"] = removal_mode
    meta["cleaning"]["threshold"] = threshold
    meta["cleaning"]["original_shapes"] = original_count
    meta["cleaning"]["cleaned_shapes"] = len(cleaned_shapes)
    meta["cleaning"]["removed"] = original_count - len(cleaned_shapes)
    
    return meta


def process_split(input_split_dir: str, output_split_dir: str, removal_mode: str = "remove", 
                  threshold: float = 0.25) -> None:
    """Process all samples in a split directory."""
    os.makedirs(output_split_dir, exist_ok=True)
    
    json_files = [f for f in os.listdir(input_split_dir) if f.endswith(".json")]
    
    total_removed = 0
    total_kept = 0
    
    for json_fname in sorted(json_files):
        json_path = os.path.join(input_split_dir, json_fname)
        img_fname = json_fname.replace(".json", ".jpg")
        img_path = os.path.join(input_split_dir, img_fname)
        
        # Clean JSON
        cleaned_meta = clean_json(json_path, removal_mode, threshold)
        
        # Copy image
        dst_img = os.path.join(output_split_dir, img_fname)
        if os.path.exists(img_path):
            shutil.copy2(img_path, dst_img)
        
        # Write cleaned JSON
        dst_json = os.path.join(output_split_dir, json_fname)
        with open(dst_json, "w", encoding="utf-8") as f:
            json.dump(cleaned_meta, f, ensure_ascii=False, indent=2)
        
        removed = cleaned_meta["cleaning"]["removed"]
        kept = cleaned_meta["cleaning"]["cleaned_shapes"]
        total_removed += removed
        total_kept += kept
        
        status = f"Removed: {removed}, Kept: {kept}"
        print(f"{json_fname}: {status}")
    
    print(f"\n{os.path.basename(output_split_dir)} summary: Total removed: {total_removed}, Total kept: {total_kept}")


def main():
    p = argparse.ArgumentParser(description="Clean dataset by removing damaged shapes from annotations")
    p.add_argument("--input-dir", default="dataset/organized", 
                   help="Path to organized dataset directory")
    p.add_argument("--output-dir", default="dataset/cleaned", 
                   help="Path to write cleaned dataset")
    p.add_argument("--mode", choices=["remove", "mark"], default="remove",
                   help="remove: delete damaged shapes, mark: add 'damaged' flag")
    p.add_argument("--threshold", type=float, default=0.25,
                   help="Intersection threshold to consider shape damaged (0.0-1.0)")
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="all",
                   help="Which split(s) to process")
    args = p.parse_args()
    
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    
    for split in splits:
        input_split = os.path.join(args.input_dir, split)
        output_split = os.path.join(args.output_dir, split)
        
        if not os.path.exists(input_split):
            print(f"Split not found: {input_split}, skipping")
            continue
        
        print(f"\nProcessing {split}...")
        process_split(input_split, output_split, args.mode, args.threshold)
    
    print(f"\nCleaned dataset saved to {args.output_dir}")


if __name__ == "__main__":
    main()
