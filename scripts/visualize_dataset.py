#!/usr/bin/env python3
"""
Visualize preprocessed dataset samples with bounding boxes and damage info.

Displays random samples or specific sample IDs with:
- Original image with bounding boxes and text labels
- Damage information overlay
- Image metadata (dimensions, number of shapes, etc.)

Color modes (select with --color-mode):
- labels (default): text/shape boxes in red, damage regions in orange
- classic: text/shape boxes in green, damage regions in yellow
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


def draw_sample(meta: dict, img: Image.Image, output_path: str = None,
                color_mode: str = "labels", thickness: int = 2,
                show_text: bool = True, show_damage: bool = True) -> Image.Image:
    """Draw image with bounding boxes, labels, damage areas, and damage info.

    color_mode:
      - "labels": text boxes red, damage orange
      - "classic": text boxes green, damage yellow
    thickness: outline thickness for boxes
    """
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Try to load font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()

    # Select color palette
    if color_mode == "classic":
        COLOR_TEXT = (0, 255, 0)
        COLOR_TEXT_LABEL_BG = (0, 255, 0)
        COLOR_TEXT_LABEL_FG = (0, 0, 0)
        COLOR_DAMAGE = (255, 255, 0)
        COLOR_DAMAGE_TEXT = (255, 255, 0)
    else:  # labels (default)
        COLOR_TEXT = (255, 0, 0)
        COLOR_TEXT_LABEL_BG = (255, 0, 0)
        COLOR_TEXT_LABEL_FG = (255, 255, 255)
        COLOR_DAMAGE = (255, 100, 0)
        COLOR_DAMAGE_TEXT = (255, 200, 0)

    # First, compute and optionally draw damage areas (if present)
    damage = meta.get("damage", {})
    applied_damages = damage.get("applied", [])
    damage_boxes = []

    for dmg in applied_damages:
        dmg_type = dmg.get("type")

        if dmg_type == "spot":
            # Remove circular overlay; draw only rectangular box
            center = dmg.get("center", [0, 0])
            radius = dmg.get("radius", 0)
            cx, cy = center[0], center[1]
            box = dmg.get("box")
            if isinstance(box, (list, tuple)) and len(box) == 4:
                if show_damage:
                    draw.rectangle(box, outline=COLOR_DAMAGE, width=max(1, thickness))
                damage_boxes.append(tuple(box))
            else:
                rect = (cx - radius, cy - radius, cx + radius, cy + radius)
                if show_damage:
                    draw.rectangle(rect, outline=COLOR_DAMAGE, width=max(1, thickness))
                damage_boxes.append(rect)

        elif dmg_type == "distortion":
            box = dmg.get("box", [0, 0, 0, 0])
            if show_damage:
                draw.rectangle(box, outline=COLOR_DAMAGE, width=max(1, thickness))
            damage_boxes.append(tuple(box))
    
    # Draw bounding boxes and labels
    shapes = meta.get("shapes", [])
    
    for i, shape in enumerate(shapes):
        pts = shape.get("points", [])
        if len(pts) >= 2:
            x1 = min(pts[0][0], pts[1][0])
            y1 = min(pts[0][1], pts[1][1])
            x2 = max(pts[0][0], pts[1][0])
            y2 = max(pts[0][1], pts[1][1])
            
            shape_bbox = (x1, y1, x2, y2)
            
            # Draw rectangle for text/shape
            if show_text:
                draw.rectangle([x1, y1, x2, y2], outline=COLOR_TEXT, width=max(1, thickness))
            
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
            if show_text and (not occluded and not is_damaged):
                label = shape.get("label", f"Shape {i}")
                
                # Draw text background
                bbox = draw.textbbox((x1, y1 - 15), label, font=font)
                draw.rectangle(bbox, fill=COLOR_TEXT_LABEL_BG)
                draw.text((x1, y1 - 15), label, fill=COLOR_TEXT_LABEL_FG, font=font)
            elif show_text:
                # Draw indicator for hidden label
                indicator = "[HIDDEN]" if is_damaged else "[OCCLUDED]"
                draw.text((x1, y1 - 15), indicator, fill=(200, 0, 0), font=font)
    
    # Draw damage info overlay (top-left)
    if applied_damages:
        text_y = 10
        for dmg in applied_damages:
            dmg_type = dmg.get("type", "unknown")
            text = f"Damage: {dmg_type}"
            draw.text((10, text_y), text, fill=COLOR_DAMAGE_TEXT, font=font)
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


def visualize_sample(dataset_dir: str, sample_id: str = None, output_dir: str = None,
                     color_mode: str = "labels", thickness: int = 2,
                     boxes: str = "both") -> None:
    """Visualize a single sample from dataset."""
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
        return
    
    if not sample_id:
        print("Error: sample_id must be provided to visualize_sample().")
        return
    
    print(f"\nVisualizing {sample_id}...")
    meta, img = load_sample(dataset_dir, sample_id)
    
    if meta is None:
        print("Failed to load sample")
        return
    
    if img is None:
        print("Failed to load image")
        return
    
    # Resolve box visibility
    show_text = boxes in ("both", "text")
    show_damage = boxes in ("both", "damage")

    # Draw
    visualized = draw_sample(meta, img.copy(), color_mode=color_mode, thickness=thickness,
                             show_text=show_text, show_damage=show_damage)
    
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
    p = argparse.ArgumentParser(description="Visualize dataset samples")
    p.add_argument("--dataset-dir", default="dataset/cleaned", 
                   help="Path to dataset directory")
    p.add_argument("--sample-id", default=None,
                   help="Specific sample ID to visualize (all if not specified)")
    p.add_argument("--output-dir", default=None,
                   help="Save visualizations to this directory (show if not specified)")
    p.add_argument("--color-mode", choices=["labels", "classic"], default="labels",
                   help="Color scheme: 'labels' = text red & damage orange (default), 'classic' = text green & damage yellow")
    p.add_argument("--thickness", type=int, default=2,
                   help="Outline thickness for boxes")
    p.add_argument("--boxes", choices=["both", "text", "damage", "none"], default="both",
                   help="Which boxes to draw: both (default), text, damage, or none")
    args = p.parse_args()
    
    # If sample_id provided, visualize just that one
    if args.sample_id:
        visualize_sample(args.dataset_dir, args.sample_id, args.output_dir,
                         color_mode=args.color_mode, thickness=args.thickness,
                         boxes=args.boxes)
    else:
        # Visualize all samples
        if not os.path.exists(args.dataset_dir):
            print(f"Dataset directory not found: {args.dataset_dir}")
            return
        
        json_files = [f for f in os.listdir(args.dataset_dir) if f.endswith(".json")]
        if not json_files:
            print(f"No samples in {args.dataset_dir}")
            return
        
        json_files.sort()
        print(f"Visualizing {len(json_files)} samples from {args.dataset_dir}")
        for json_file in json_files:
            sample_id = json_file.replace(".json", "")
            visualize_sample(args.dataset_dir, sample_id, args.output_dir,
                             color_mode=args.color_mode, thickness=args.thickness,
                             boxes=args.boxes)


if __name__ == "__main__":
    main()
