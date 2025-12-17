#!/usr/bin/env python3
"""
Generate visually damaged copies of dataset images and update annotations.

Creates `output_dir` with damaged images and JSONs. Damage types are applied
without geometric transforms so annotation coordinates remain valid. Shapes
that are overlapped by occlusions receive an `"occluded": true` flag.

Dependencies: Pillow, numpy
If missing, the script prints install instructions.
"""
from __future__ import annotations
import os
import sys
import json
import random
import argparse
from io import BytesIO
try:
    from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
    import numpy as np
except Exception as e:
    print("Missing dependency:", e)
    print("Install required packages: pip install pillow numpy")
    sys.exit(1)


def gaussian_blur(img: Image.Image, radius: float = 2.0) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius))


def add_gaussian_noise(img: Image.Image, sigma: float = 10.0) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr += noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def adjust_brightness_contrast(img: Image.Image, brightness: float = 1.0, contrast: float = 1.0) -> Image.Image:
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    return img


def add_stain(img: Image.Image, intensity: float = 0.6, max_radius: int | None = None) -> Image.Image:
    # Create a soft gradient elliptical blob (brown -> black) and composite it.
    w, h = img.size
    if max_radius is None:
        max_radius = max(w, h) // 8
    # choose random bbox for the stain
    rx = random.randint(max_radius // 2, max_radius)
    ry = random.randint(max_radius // 2, max_radius)
    cx = random.randint(0, w)
    cy = random.randint(0, h)
    x1 = max(0, cx - rx)
    y1 = max(0, cy - ry)
    x2 = min(w, cx + rx)
    y2 = min(h, cy + ry)
    bbox = (x1, y1, x2, y2)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return img

    # Colors: center darker (near black) -> edge brown
    inner_col = np.array([0, 0, 0], dtype=np.float32)
    outer_col = np.array([120, 60, 20], dtype=np.float32)
    max_alpha = int(255 * float(intensity))

    yy, xx = np.mgrid[0:bh, 0:bw]
    cx_loc = bw / 2.0
    cy_loc = bh / 2.0
    dx = (xx - cx_loc) / (bw / 2.0 + 1e-6)
    dy = (yy - cy_loc) / (bh / 2.0 + 1e-6)
    dist = np.sqrt(dx * dx + dy * dy)
    dist = np.clip(dist, 0.0, 1.0)

    # radial falloff mask (stronger near center)
    mask = (1.0 - dist) ** 1.8

    # color blends from black (center) to brown (edge)
    cols = (inner_col[None, None, :] * (1.0 - dist[..., None]) + outer_col[None, None, :] * (dist[..., None])).astype(np.uint8)
    alphas = (mask * max_alpha).astype(np.uint8)

    rgba = np.zeros((bh, bw, 4), dtype=np.uint8)
    rgba[..., :3] = cols
    rgba[..., 3] = alphas

    blob = Image.fromarray(rgba, mode="RGBA")
    overlay.paste(blob, (x1, y1), blob)
    result = Image.alpha_composite(img.convert("RGBA"), overlay)
    return result.convert(img.mode)


def jpeg_compress(img: Image.Image, quality: int = 30) -> Image.Image:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert(img.mode)


def add_occlusion(img: Image.Image, occl_box: tuple[int, int, int, int], color=(20, 20, 20), alpha=255) -> Image.Image:
    # Create an irregular blob occlusion made of multiple overlapping gradient
    # elliptical blobs to produce a natural, rounded shape that gradients
    # between brown and black.
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    x1, y1, x2, y2 = occl_box
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return img

    # make 2-4 overlapping blobs inside the box
    blobs = random.randint(2, 4)
    for i in range(blobs):
        # random smaller ellipse inside the occlusion bbox
        bw = int(w * random.uniform(0.4, 1.0))
        bh = int(h * random.uniform(0.4, 1.0))
        bx1 = int(x1 + random.uniform(0, max(0, w - bw)))
        by1 = int(y1 + random.uniform(0, max(0, h - bh)))
        bx2 = bx1 + bw
        by2 = by1 + bh
        # inner black, outer brown
        inner_col = np.array([0, 0, 0], dtype=np.float32)
        outer_col = np.array([100 + random.randint(-20, 40), 45 + random.randint(-10, 20), 20 + random.randint(-5, 10)], dtype=np.float32)
        max_alpha = int(alpha * random.uniform(0.6, 1.0))

        bh_local = by2 - by1
        bw_local = bx2 - bx1
        if bw_local <= 0 or bh_local <= 0:
            continue

        yy, xx = np.mgrid[0:bh_local, 0:bw_local]
        cx_loc = bw_local / 2.0 + random.uniform(-bw_local * 0.15, bw_local * 0.15)
        cy_loc = bh_local / 2.0 + random.uniform(-bh_local * 0.15, bh_local * 0.15)
        dx = (xx - cx_loc) / (bw_local / 2.0 + 1e-6)
        dy = (yy - cy_loc) / (bh_local / 2.0 + 1e-6)
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.clip(dist, 0.0, 1.0)
        mask = (1.0 - dist) ** (1.6 + random.random())

        cols = (inner_col[None, None, :] * (1.0 - dist[..., None]) + outer_col[None, None, :] * (dist[..., None]))
        cols = np.clip(cols, 0, 255).astype(np.uint8)
        alphas = (mask * max_alpha).astype(np.uint8)

        rgba = np.zeros((bh_local, bw_local, 4), dtype=np.uint8)
        rgba[..., :3] = cols
        rgba[..., 3] = alphas

        blob = Image.fromarray(rgba, mode="RGBA")
        overlay.paste(blob, (bx1, by1), blob)

    out = Image.alpha_composite(img.convert("RGBA"), overlay)
    return out.convert(img.mode)


def random_occlusion_box(w: int, h: int, min_area_frac=0.02, max_area_frac=0.15) -> tuple[int, int, int, int]:
    area = w * h
    target = random.uniform(min_area_frac, max_area_frac) * area
    # choose random aspect
    aspect = random.uniform(0.3, 3.0)
    ow = int((target * aspect) ** 0.5)
    oh = int((target / aspect) ** 0.5)
    ox = random.randint(0, max(0, w - ow))
    oy = random.randint(0, max(0, h - oh))
    return (ox, oy, min(w, ox + ow), min(h, oy + oh))


def bbox_intersection_area(b1: tuple[float, float, float, float], b2: tuple[float, float, float, float]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def _create_irregular_spot_overlay(size: tuple[int, int], center: tuple[int, int], radius: float, intensity: float = 1.0) -> Image.Image:
    """Create an RGBA overlay with an irregular near-round dark spot.

    - `size`: (w,h) of overlay
    - `center`: (cx,cy)
    - `radius`: base radius in pixels
    - `intensity`: overall alpha multiplier 0..1
    """
    w, h = size
    cx, cy = int(center[0]), int(center[1])
    # compute bbox for processing to save work
    x1 = max(0, int(cx - radius * 1.6))
    y1 = max(0, int(cy - radius * 1.6))
    x2 = min(w, int(cx + radius * 1.6))
    y2 = min(h, int(cy + radius * 1.6))
    bw, bh = x2 - x1, y2 - y1

    # coordinate grids
    yy, xx = np.mgrid[0:bh, 0:bw]
    xx = xx + x1
    yy = yy + y1
    dx = xx - cx
    dy = yy - cy
    d = np.sqrt(dx * dx + dy * dy)

    # base radial falloff (smoothstep S-curve for tighter core and softer edge)
    r = radius
    t = np.clip(1.0 - (d / r), 0.0, 1.0)
    # smootherstep for steeper center and softer edge
    smooth = t * t * t * (t * (6 * t - 15) + 10)
    # bias toward darker core; exponent <1 sharpens center falloff
    base = smooth ** 0.5  # sharper at center
    # irregularity: smooth noise
    noise = (np.random.rand(bh, bw).astype(np.float32) - 0.5) * 0.6
    # smooth noise using PIL GaussianBlur
    noise_img = Image.fromarray(np.uint8((noise - noise.min()) / (noise.max() - noise.min() + 1e-9) * 255))
    noise_img = noise_img.filter(ImageFilter.GaussianBlur(radius=max(1, int(radius / 8))))
    noise = np.array(noise_img).astype(np.float32) / 255.0
    noise = (noise - 0.5) * 0.6

    irregular = base * (1.0 + noise)
    irregular = np.clip(irregular, 0.0, 1.0)
    # S-shaped opacity falloff; smaller exponent keeps dark area closer to target radius
    alpha = (irregular ** 0.55) * 255.0 * intensity

    # color: center near-black, edge slightly brownish
    center_col = np.array([10, 10, 10], dtype=np.uint8)
    edge_col = np.array([40, 25, 10], dtype=np.uint8)
    mix = np.expand_dims(base, axis=2)
    rgb = (center_col * mix + edge_col * (1 - mix)).astype(np.uint8)

    overlay_arr = np.zeros((bh, bw, 4), dtype=np.uint8)
    overlay_arr[:, :, 0:3] = rgb
    overlay_arr[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)

    overlay = Image.fromarray(overlay_arr, mode="RGBA")
    full = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    full.paste(overlay, (x1, y1), overlay)
    return full


def _apply_spot(img: Image.Image, target_frac: float = 0.05) -> tuple[Image.Image, dict]:
    w, h = img.size
    area = w * h
    target_area = area * target_frac
    # approximate circle radius from area
    radius = int(((target_area) / np.pi) ** 0.5)
    # jitter radius a bit
    radius = max(10, int(radius * random.uniform(0.85, 1.15)))
    cx = random.randint(radius, max(radius, w - radius))
    cy = random.randint(radius, max(radius, h - radius))
    intensity = random.uniform(0.8, 1.0)
    overlay = _create_irregular_spot_overlay((w, h), (cx, cy), radius, intensity=intensity)
    out = Image.alpha_composite(img.convert("RGBA"), overlay).convert(img.mode)
    box = (max(0, cx - radius), max(0, cy - radius), min(w, cx + radius), min(h, cy + radius))
    return out, {"type": "spot", "box": box, "center": (cx, cy), "radius": radius, "intensity": intensity}


def _apply_distortion(img: Image.Image, frac_min=0.03, frac_max=0.10) -> tuple[Image.Image, dict]:
    """Apply a localized horizontal-shear-like distortion to a random box region."""
    w, h = img.size
    area = w * h
    target_frac = random.uniform(frac_min, frac_max)
    target_area = area * target_frac
    # choose box aspect randomly
    aspect = random.uniform(0.4, 2.5)
    bw = int((target_area * aspect) ** 0.5)
    bh = int((target_area / aspect) ** 0.5)
    bw = min(max(20, bw), w)
    bh = min(max(20, bh), h)
    ox = random.randint(0, max(0, w - bw))
    oy = random.randint(0, max(0, h - bh))
    box = (ox, oy, ox + bw, oy + bh)

    region = img.crop(box)
    arr = np.array(region)
    # row-wise horizontal jitter
    max_shift = max(1, int(bw * 0.08))
    out_arr = np.zeros_like(arr)
    for i in range(bh):
        shift = int((np.sin((i / bh) * np.pi * random.uniform(1.0, 3.0)) * max_shift) + random.randint(-max_shift, max_shift))
        out_arr[i] = np.roll(arr[i], shift, axis=0)
        # fill vacated cols with nearby pixels to avoid black bars
        if shift > 0:
            out_arr[i, :shift] = arr[i, :shift]
        elif shift < 0:
            out_arr[i, shift:] = arr[i, shift:]

    distorted = Image.fromarray(out_arr)
    out = img.copy()
    out.paste(distorted, box)
    return out, {"type": "distortion", "box": box, "max_shift": max_shift}


def apply_random_damages(img: Image.Image, w: int, h: int) -> tuple[Image.Image, dict]:
    """Apply only two damages: one irregular near-round spot (~10% area) and
    one localized distortion area. Returns the modified image and metadata.
    """
    out = img.copy()
    applied = []

    # Spot (near 5% area)
    out, spot_meta = _apply_spot(out, target_frac=0.05)
    applied.append(spot_meta)

    # Distortion (random area)
    out, dist_meta = _apply_distortion(out)
    applied.append(dist_meta)

    return out, {"applied": applied}


def _reduce_damage_boxes_by_diff(orig_img: Image.Image, damaged_img: Image.Image,
                                 applied: list[dict], diff_threshold: int = 20,
                                 min_pixels: int = 32,
                                 shrink_only: bool = True) -> list[dict]:
    """Shrink damage boxes to the visible changed area using grayscale abs-diff.

    - For each damage with a 'box', compute a tight bbox around pixels where
      |damaged - original| > diff_threshold within that box.
    - If not enough pixels exceed threshold (min_pixels), keep the original box.
    - If shrink_only, never expand beyond the original.
    """
    if not applied:
        return applied

    orig_gray = np.array(orig_img.convert("L"), dtype=np.int16)
    dmg_gray = np.array(damaged_img.convert("L"), dtype=np.int16)
    diff = np.abs(dmg_gray - orig_gray)

    w, h = orig_img.size
    new_applied = []
    for a in applied:
        if not isinstance(a, dict) or "box" not in a:
            new_applied.append(a)
            continue
        x1, y1, x2, y2 = map(int, a["box"])
        x1 = max(0, min(w, x1)); x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1)); y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            new_applied.append(a)
            continue
        sub = diff[y1:y2, x1:x2]
        mask = sub > int(diff_threshold)
        if int(mask.sum()) < int(min_pixels):
            new_applied.append(a)
            continue
        ys, xs = np.where(mask)
        if ys.size == 0 or xs.size == 0:
            new_applied.append(a)
            continue
        nx1 = x1 + int(xs.min())
        ny1 = y1 + int(ys.min())
        nx2 = x1 + int(xs.max()) + 1
        ny2 = y1 + int(ys.max()) + 1
        if shrink_only:
            nx1 = max(x1, nx1); ny1 = max(y1, ny1)
            nx2 = min(x2, nx2); ny2 = min(y2, ny2)
        if nx2 <= nx1 or ny2 <= ny1:
            new_applied.append(a)
            continue
        a2 = dict(a)
        a2["box"] = (int(nx1), int(ny1), int(nx2), int(ny2))
        new_applied.append(a2)

    return new_applied


def process_file(json_path: str, input_dir: str, output_dir: str, overwrite: bool = False) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    img_name = meta.get("imagePath")
    if not img_name:
        print(f"No imagePath in {json_path}, skipping")
        return
    img_path = os.path.join(input_dir, img_name)
    if not os.path.exists(img_path):
        # try same directory as json
        alt = os.path.join(os.path.dirname(json_path), img_name)
        if os.path.exists(alt):
            img_path = alt
        else:
            print(f"Image not found for {json_path}: {img_path}")
            return
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    damaged_img, damage_meta = apply_random_damages(img, w, h)

    # Update shapes with occlusion flags when occlusion overlaps bounding boxes
    occlusions = damage_meta.get("occlusions", [])
    shapes = meta.get("shapes", [])
    for s in shapes:
        pts = s.get("points", [])
        if len(pts) >= 2:
            # assume bbox [x1,y1],[x2,y2]
            x1 = min(pts[0][0], pts[1][0])
            y1 = min(pts[0][1], pts[1][1])
            x2 = max(pts[0][0], pts[1][0])
            y2 = max(pts[0][1], pts[1][1])
            bbox = (x1, y1, x2, y2)
            occluded = False
            for ob in occlusions:
                # ob is (ox1,oy1,ox2,oy2)
                inter = bbox_intersection_area(bbox, ob)
                area = (x2 - x1) * (y2 - y1)
                if area > 0 and (inter / area) >= 0.25:
                    occluded = True
                    break
            if occluded:
                s["occluded"] = True

    # prepare output paths
    os.makedirs(output_dir, exist_ok=True)
    out_img_path = os.path.join(output_dir, img_name)
    out_json_path = os.path.join(output_dir, os.path.basename(json_path))

    if os.path.exists(out_img_path) and not overwrite:
        print(f"Skipping existing: {out_img_path}")
        return

    damaged_img.save(out_img_path)
    # augment meta
    meta_out = dict(meta)
    meta_out["damage"] = damage_meta
    meta_out["imagePath"] = os.path.basename(out_img_path)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)
    print(f"Wrote: {out_img_path}, {out_json_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="dataset/raw", help="Path to raw dataset folder containing images and JSONs")
    p.add_argument("--output-dir", default="dataset/damaged", help="Path to write damaged images and JSONs")
    p.add_argument("--limit", type=int, default=0, help="Limit number of files processed (0 = all)")
    p.add_argument("--max-damages", type=int, default=2, help="Maximum number of damages to apply per image (0..n)")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    p.add_argument("--reduce-damage-boxes", action="store_true",
                   help="Reduce damage boxes to visible area using grayscale difference")
    p.add_argument("--diff-threshold", type=int, default=20,
                   help="Grayscale abs-diff threshold (0-255) to consider a pixel damaged")
    p.add_argument("--min-damage-pixels", type=int, default=32,
                   help="Minimum count of above-threshold pixels within a box to shrink it")
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # collect json files
    files = []
    for fname in os.listdir(args.input_dir):
        if fname.lower().endswith(".json"):
            files.append(os.path.join(args.input_dir, fname))
    files.sort()
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    if not files:
        print("No JSON files found in", args.input_dir)
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # configure damage selection
    allowed = ["spot", "distortion"]

    for j in files:
        try:
            # decide how many damages to apply (0..max_damages)
            max_d = max(0, int(args.max_damages))
            if max_d == 0:
                # simply copy without damage
                process_file(j, args.input_dir, args.output_dir, overwrite=args.overwrite)
            else:
                k = random.randint(0, max_d)
                # choose k damage types at random (with replacement to allow repeats)
                chosen = random.choices(allowed, k=k)
                # process_file will call apply_random_damages, but we need a version
                # that accepts chosen types. To avoid large refactor, open and
                # modify process inline here.
                with open(j, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                img_name = meta.get("imagePath")
                if not img_name:
                    print(f"No imagePath in {j}, skipping")
                    continue
                img_path = os.path.join(args.input_dir, img_name)
                if not os.path.exists(img_path):
                    alt = os.path.join(os.path.dirname(j), img_name)
                    if os.path.exists(alt):
                        img_path = alt
                    else:
                        print(f"Image not found for {j}: {img_path}")
                        continue
                img = Image.open(img_path).convert("RGB")
                w, h = img.size

                out_img = img.copy()
                applied = []
                for typ in chosen:
                    if typ == "spot":
                        out_img, meta_d = _apply_spot(out_img, target_frac=0.10)
                        applied.append(meta_d)
                    elif typ == "distortion":
                        out_img, meta_d = _apply_distortion(out_img)
                        applied.append(meta_d)

                # Reduce damage boxes by visible diff if enabled
                if getattr(args, "reduce_damage_boxes", False):
                    applied = _reduce_damage_boxes_by_diff(
                        img, out_img, applied,
                        diff_threshold=getattr(args, "diff_threshold", 20),
                        min_pixels=getattr(args, "min_damage_pixels", 32),
                        shrink_only=True,
                    )

                # Update shapes occlusion flag: we mark shapes occluded if any
                # applied damage has a box overlapping >=25% (only spot/distortion)
                shapes = meta.get("shapes", [])
                damage_boxes = []
                for a in applied:
                    if "box" in a:
                        damage_boxes.append(tuple(a["box"]))
                for s in shapes:
                    pts = s.get("points", [])
                    if len(pts) >= 2:
                        x1 = min(pts[0][0], pts[1][0])
                        y1 = min(pts[0][1], pts[1][1])
                        x2 = max(pts[0][0], pts[1][0])
                        y2 = max(pts[0][1], pts[1][1])
                        bbox = (x1, y1, x2, y2)
                        occluded = False
                        for ob in damage_boxes:
                            inter = bbox_intersection_area(bbox, ob)
                            area = (x2 - x1) * (y2 - y1)
                            if area > 0 and (inter / area) >= 0.25:
                                occluded = True
                                break
                        if occluded:
                            s["occluded"] = True

                os.makedirs(args.output_dir, exist_ok=True)
                out_img_path = os.path.join(args.output_dir, img_name)
                out_json_path = os.path.join(args.output_dir, os.path.basename(j))
                if os.path.exists(out_img_path) and not args.overwrite:
                    print(f"Skipping existing: {out_img_path}")
                    continue
                out_img.save(out_img_path)
                meta_out = dict(meta)
                meta_out["damage"] = {"applied": applied}
                meta_out["imagePath"] = os.path.basename(out_img_path)
                with open(out_json_path, "w", encoding="utf-8") as f:
                    json.dump(meta_out, f, ensure_ascii=False, indent=2)
                print(f"Wrote: {out_img_path}, {out_json_path}")
        except Exception as e:
            print(f"Error processing {j}: {e}")


if __name__ == "__main__":
    main()
