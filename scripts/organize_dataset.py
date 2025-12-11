#!/usr/bin/env python3
"""
Organize preprocessed dataset into a structured format.

Creates a dataset directory with train/val/test splits and a manifest file
listing all samples with their metadata.
"""
import os
import sys
import json
import argparse
import shutil
from pathlib import Path


def create_dataset(input_dir: str, output_dir: str, train_split: float = 0.7, 
                   val_split: float = 0.15, seed: int = 42) -> None:
    """Organize preprocessed dataset into structured format with splits."""
    import random
    if seed is not None:
        random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all JSON files
    json_files = []
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(".json"):
            json_files.append(fname)
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    json_files.sort()
    random.shuffle(json_files)
    
    # Calculate split indices
    n = len(json_files)
    train_n = int(n * train_split)
    val_n = int(n * val_split)
    
    train_files = json_files[:train_n]
    val_files = json_files[train_n:train_n + val_n]
    test_files = json_files[train_n + val_n:]
    
    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }
    
    manifest = {
        "dataset": output_dir,
        "source": input_dir,
        "splits": {
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
            "total": n
        },
        "samples": []
    }
    
    # Process each split
    for split_name, files in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        for json_fname in files:
            json_path = os.path.join(input_dir, json_fname)
            
            # Read JSON metadata
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            
            img_name = meta.get("imagePath", "")
            img_path = os.path.join(input_dir, img_name)
            
            # Copy image and JSON
            dst_img = os.path.join(split_dir, img_name)
            dst_json = os.path.join(split_dir, json_fname)
            
            if os.path.exists(img_path):
                shutil.copy2(img_path, dst_img)
            shutil.copy2(json_path, dst_json)
            
            # Add to manifest
            manifest["samples"].append({
                "id": json_fname.replace(".json", ""),
                "split": split_name,
                "image": img_name,
                "json": json_fname,
                "width": meta.get("imageWidth"),
                "height": meta.get("imageHeight"),
                "shapes": len(meta.get("shapes", [])),
                "damage": meta.get("damage")
            })
            
            print(f"Organized {split_name}/{json_fname}")
    
    # Write manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    print(f"\nDataset created at {output_dir}")
    print(f"  Train: {len(train_files)} samples")
    print(f"  Val: {len(val_files)} samples")
    print(f"  Test: {len(test_files)} samples")
    print(f"  Manifest: {manifest_path}")


def main():
    p = argparse.ArgumentParser(description="Organize preprocessed dataset into structured format")
    p.add_argument("--input-dir", default="dataset/preprocessed", 
                   help="Path to preprocessed images and JSONs")
    p.add_argument("--output-dir", default="dataset/organized", 
                   help="Path to write organized dataset")
    p.add_argument("--train-split", type=float, default=0.7, 
                   help="Fraction for training set (default: 0.7)")
    p.add_argument("--val-split", type=float, default=0.15, 
                   help="Fraction for validation set (default: 0.15)")
    p.add_argument("--seed", type=int, default=42, 
                   help="Random seed for reproducibility")
    args = p.parse_args()
    
    create_dataset(args.input_dir, args.output_dir, args.train_split, args.val_split, args.seed)


if __name__ == "__main__":
    main()
