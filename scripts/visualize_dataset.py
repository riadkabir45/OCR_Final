#!/usr/bin/env python3
"""
Visualize preprocessed dataset samples with bounding boxes and damage info.

Displays random samples or specific sample IDs with:
- Original image with bounding boxes and text labels
- Damage information overlay
- Image metadata (dimensions, number of shapes, etc.)
"""
import os
import sys
import json
import argparse
import random
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install pillow numpy")
    sys.exit(1)


def load_sample(sample_dir: str, sample_id: str):
    """Load image and JSON for a sample."""
    json_path = os.path.join(sample_dir, f"{sample_id}.json")
    
    if not os.path.exists(json_path):
        print(f"Sample not found: {json_path}")
        return None, None
    
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    img_name = meta.get("imagePath")
    img_path = os.path.join(sample_dir, img_name)
    
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return meta, None
    
    img = Image.open(img_path).convert("RGB")
    return meta, img


def bbox_intersection_area(b1, b2):
    """Calculate intersection area between two bounding boxes."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def draw_sample(meta: dict, img: Image.Image, output_path: str = None) -> Image.Image:
    """Draw image with bounding boxes, labels, damage areas, and damage info."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    
    # Try to load font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    # First, draw damage areas (if present)
    damage = meta.get("damage", {})
    applied_damages = damage.get("applied", [])
    damage_boxes = []
    
    for dmg in applied_damages:
        dmg_type = dmg.get("type")
        
        if dmg_type == "spot":
            # Draw spot as circle outline
            center = dmg.get("center", [0, 0])
            radius = dmg.get("radius", 0)
            cx, cy = center[0], center[1]
            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                outline=(255, 100, 0),  # orange
                width=3
            )
            damage_boxes.append((cx - radius, cy - radius, cx + radius, cy + radius))
            
        elif dmg_type == "distortion":
            # Draw distortion box
            box = dmg.get("box", [0, 0, 0, 0])
            draw.rectangle(box, outline=(0, 255, 100), width=3)  # cyan-green
            damage_boxes.append(tuple(box))
    
    # Draw bounding boxes and labels
    shapes = meta.get("shapes", [])
    colors = [
        (255, 0, 0),      # red
        (0, 255, 0),      # green
        (0, 0, 255),      # blue
        (255, 255, 0),    # yellow
        (255, 0, 255),    # magenta
        (0, 255, 255),    # cyan
    ]
    
    for i, shape in enumerate(shapes):
        pts = shape.get("points", [])
        if len(pts) >= 2:
            x1 = min(pts[0][0], pts[1][0])
            y1 = min(pts[0][1], pts[1][1])
            x2 = max(pts[0][0], pts[1][0])
            y2 = max(pts[0][1], pts[1][1])
            
            shape_bbox = (x1, y1, x2, y2)
            
            color = colors[i % len(colors)]
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Check if shape is occluded or damaged
            occluded = shape.get("occluded", False)
            is_damaged = False
            
            # Check intersection with damage areas
            shape_area = (x2 - x1) * (y2 - y1)
            if shape_area > 0:
                for dmg_box in damage_boxes:
                    inter = bbox_intersection_area(shape_bbox, dmg_box)
                    if inter / shape_area >= 0.25:  # 25% threshold
                        is_damaged = True
                        break
            
            # Draw label only if not occluded/damaged
            if not occluded and not is_damaged:
                label = shape.get("label", f"Shape {i}")
                
                # Draw text background
                bbox = draw.textbbox((x1, y1 - 15), label, font=font)
                draw.rectangle(bbox, fill=color)
                draw.text((x1, y1 - 15), label, fill=(255, 255, 255), font=font)
            else:
                # Draw indicator for hidden label
                indicator = "[HIDDEN]" if is_damaged else "[OCCLUDED]"
                draw.text((x1, y1 - 15), indicator, fill=(200, 0, 0), font=font)
    
    # Draw damage info overlay (top-left)
    if applied_damages:
        text_y = 10
        for dmg in applied_damages:
            dmg_type = dmg.get("type", "unknown")
            text = f"Damage: {dmg_type}"
            draw.text((10, text_y), text, fill=(255, 200, 0), font=font)
            text_y += 20
    
    # Draw metadata (top-right)
    img_h = meta.get("imageHeight")
    img_w = meta.get("imageWidth")
    n_shapes = len(shapes)
    
    metadata_text = [
        f"Size: {img_w}x{img_h}",
        f"Shapes: {n_shapes}",
    ]
    
    text_y = 10
    for txt in metadata_text:
        bbox = draw.textbbox((w - 200, text_y), txt, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0))
        draw.text((w - 200, text_y), txt, fill=(200, 200, 200), font=font)
        text_y += 20
    
    if output_path:
        img.save(output_path)
        print(f"Saved visualization: {output_path}")
    
    return img


def visualize_sample(dataset_dir: str, split: str, sample_id: str = None, output_dir: str = None) -> None:
    """Visualize a single sample or random sample from a split."""
    split_dir = os.path.join(dataset_dir, split)
    
    if not os.path.exists(split_dir):
        print(f"Split directory not found: {split_dir}")
        return
    
    # If no sample_id, pick a random one
    if not sample_id:
        json_files = [f for f in os.listdir(split_dir) if f.endswith(".json")]
        if not json_files:
            print(f"No samples in {split}")
            return
        sample_id = random.choice(json_files).replace(".json", "")
    
    print(f"\nVisualizing {split}/{sample_id}...")
    meta, img = load_sample(split_dir, sample_id)
    
    if meta is None:
        print("Failed to load sample")
        return
    
    if img is None:
        print("Failed to load image")
        return
    
    # Draw
    visualized = draw_sample(meta, img.copy())
    
    # Display or save
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{sample_id}_visualized.jpg")
        visualized.save(output_path)
        print(f"Saved: {output_path}")
    else:
        visualized.show()
    
    # Print metadata
    print(f"  Size: {meta.get('imageWidth')}x{meta.get('imageHeight')}")
    print(f"  Shapes: {len(meta.get('shapes', []))}")
    damage = meta.get("damage", {})
    if damage:
        applied = damage.get("applied", [])
        print(f"  Damage instances: {len(applied)}")
        for dmg in applied:
            print(f"    - {dmg.get('type')}")


def main():
    p = argparse.ArgumentParser(description="Visualize preprocessed dataset samples")
    p.add_argument("--dataset-dir", default="dataset/organized", 
                   help="Path to organized dataset directory")
    p.add_argument("--split", default="train", choices=["train", "val", "test"],
                   help="Dataset split to visualize")
    p.add_argument("--sample-id", default=None,
                   help="Specific sample ID to visualize (random if not specified)")
    p.add_argument("--num-samples", type=int, default=1,
                   help="Number of random samples to visualize")
    p.add_argument("--output-dir", default=None,
                   help="Save visualizations to this directory (show if not specified)")
    args = p.parse_args()
    
    # Visualize samples
    for i in range(args.num_samples):
        visualize_sample(args.dataset_dir, args.split, args.sample_id, args.output_dir)
        if args.sample_id:
            break  # Only visualize specified sample once


if __name__ == "__main__":
    main()
