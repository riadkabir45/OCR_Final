# generate_damaged_dataset.py

Creates visually damaged copies of the images in your raw dataset and writes
augmented JSON annotation files that include a `damage` object describing the
applied effects. Shapes that are overlapped by occlusion rectangles receive an
`"occluded": true` flag.

Usage examples:

Process the whole `dataset/raw` folder and write to `dataset/damaged`:

```bash
python3 scripts/generate_damaged_dataset.py \
  --input-dir dataset/raw \
  --output-dir dataset/damaged
```

Run a quick smoke test on 3 files:

```bash
python3 scripts/generate_damaged_dataset.py --input-dir dataset/raw --output-dir dataset/damaged --limit 3 --seed 42
```

Notes:
 The script now only applies two damage types:

- **Irregular spot**: a not-perfect-round dark spot (near-black with slightly
  brownish edges) sized to roughly 5% of the image area. It's drawn as an
  irregular gradient blob so it looks natural.
 - **Distortion area**: a localized distorted region where rows are horizontally
   jittered to simulate a smear/warp.

 The script no longer applies blur, noise, rectangular occlusions,
 brightness/contrast, or JPEG artifacting.

New options:

- `--max-damages N`: choose a random number of damages between 0 and N (inclusive)
  to apply for each image. If `N=0` the image is copied unchanged. Default: 2.

Examples:

```bash
# Apply 0..2 random damages per image (default behavior)
python3 scripts/generate_damaged_dataset.py --input-dir dataset/raw --output-dir dataset/damaged

# Apply 0..4 random damages per image
python3 scripts/generate_damaged_dataset.py --input-dir dataset/raw --output-dir dataset/damaged --max-damages 4
```
