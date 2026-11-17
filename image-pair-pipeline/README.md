# Image Pair Pipeline

Processes before/after drone image pairs with a YOLO26 segmentation model to produce structured training data for a Siamese classification model.

## Installation

```bash
cd image-pair-pipeline
pip install -e .
```

## Usage

```bash
python -m pipeline \
  --weights               yolo26n-seg.pt \
  --before-dir            ./before/ \
  --after-dir             ./after/ \
  --intrinsics-csv        camera_intrinsics.csv \
  --gps-csv               camera_gps.csv \
  --building-csv          buildings.csv \
  --frame-buildings-json  frame_buildings.json \
  --output-crops          ./output_crops/ \
  --output-masks          ./output_masks/
```

## Input file formats

### `--frame-buildings-json` (optional per-frame pixel locations)
```json
{
  "frame_0001": [
    {
      "handle": "13765050175",
      "address": "1010 N KINGSHIGHWAY BLVD",
      "damage_rating": "Moderate",
      "pixel_x": 768.32,
      "pixel_y": 496.01,
      "distance_meters": 110.36
    }
  ]
}
```
When provided, this ground-truth JSON is used directly:
1. For each YOLO detection on the before image, candidate points `(pixel_x, pixel_y)` are tested against the segmentation polygon mask bounds.
2. If multiple points fit within the mask, **preference is given to the closest point using `distance_meters`** (with ties broken by centrality to the mask).
3. If `--frame-buildings-json` is provided, `--building-csv` is completely optional and the raycast distance algorithm is bypassed.

### `--intrinsics-csv` (per-frame camera parameters)
```
#name	x	y	alt	heading	pitch	roll	f	px	py	k1	k2	k3
frame_0001.png	8.262	156.609	125.589	170.108	78.012	-1.107	22.227	0.00159	-0.00778	...
```

### `--gps-csv` (per-frame GPS + rotation)
```
#name	lon	lat	alt	omega	phi	kappa
frame_0001.png	-90.2621	38.6572	153.068	173.966	-26.258	-9.432
```

### `--building-csv` (one row per building with damage rating)
```
HANDLE	Lon	Lat	Rating	SITEADDR	...
13613000310	-90.234183	38.67055581	Substantial to Heavy	4300 SAN FRANCISCO AV	...
```

**Rating → class mapping:**
| Rating text | Class |
|---|---|
| Negligible to Slight / Slight | 1 |
| Moderate | 2 |
| Substantial to Heavy / Heavy | 3 |
| Very Heavy | 4 |
| Destruction / Complete | 5 |

## Output structure

```
output_crops/
  approvals/        ← side-by-side preview images (for human review)
  class1/
    0042_00_before.jpg
    0042_00_after.jpg
  class2/  ...

output_masks/
  class1/
    0042_00_before.png   ← segmented object pixels on black background, padded to square
    0042_00_after.png
  class2/  ...
```

## Filename conventions

| Image type | Pattern | Example |
|---|---|---|
| Before | `frame_####.png` | `frame_0042.png` |
| After | `cesiumcamerasequence.####` | `cesiumcamerasequence.0042.jpg` |

Pairs are matched on the 4-digit suffix. Extra after-frames with no matching before-frame are silently ignored.

## Key options

| Flag | Default | Description |
|---|---|---|
| `--frame-step` | 5 | Sample every N-th frame pair |
| `--start-pair` | None | Starting frame pair suffix or filename (e.g. `0036` or `frame_0036`) |
| `--distance-algorithm` | `raycast` | Algorithm to match detections with buildings: `fov`, `hybrid`, `raycast`, or `drone` |
| `--min-bbox-px` | 50 | Discard detections narrower/shorter than this |
| `--iou-threshold` | 0.10 | Min IoU score to keep a pair |
| `--size-penalty-weight` | 1.0 | Exponent for size similarity penalty: $\text{score} = \text{IoU} \times (\min(\text{area}) / \max(\text{area}))^{\text{weight}}$ |
| `--building-radius-m` | 50.0 | Maximum search distance (m) to associate a building |
| `--default-class` | None | Fallback class (1-5) if building lookup exceeds radius |
| `--force-class` | None | Force all approved pairs to this class, bypassing lookup |
| `--class-map` | None | CSV mapping `suffix,class` or `frame,class` for manual overrides |
| `--max-per-building` | 3 | Max pairs retained per unique building |
| `--crop-pad` | 0.20 | Fractional bbox padding (20%) |
| `--sensor-width-mm` | 13.2 | Camera sensor width (DJI 1" default) |
| `--preview-height` | 1080 | Height of the approval preview image in pixels |
| `--no-approve` | off | Skip interactive approval step |

## Interactive Approval Controls

When a frame pair is inspected, the preview opens and prompts you in the terminal:
- `y` / `yes` / `all` : Approve all proposed matching pairs using their lookup classes.
- `y <class>` (e.g. `y 3`) : Approve all pairs, forcing them all to class 3.
- `1, 3` or `1 3` : Approve **only** specific pairs by their index numbers.
- `1:3, 2:4` : Approve pair #1 as class 3 and pair #2 as class 4.
- `1-3:2` : Approve pairs #1 through #3 as class 2.
- `n` / `no` / `skip` : Skip all pairs for this frame.
- `q` / `quit` : Abort the pipeline immediately.
- `1-3, 5` : Ranges are supported (approves pairs 1, 2, 3, and 5).
- `n` / `no` / `skip` : Skip this frame entirely (saves zero pairs).
- `q` / `quit` : Save progress and exit the pipeline immediately.

## Geo-projection notes

Pixel-to-lat/lon projection uses:
- Drone GPS position from `--gps-csv` (`lon`, `lat`, `alt`)
- Camera orientation from `--intrinsics-csv` (`heading`, `pitch`, `roll`)
- Focal length `f` from intrinsics CSV (treated as mm)
- Sensor width from `--sensor-width-mm` (default 13.2 mm for DJI 1" sensor)
- Ground reference altitude = altitude of the first (ground-level) frame

The projection assumes a pinhole camera model with nadir or near-nadir imagery.
Accuracy is sufficient for the 30 m building-match radius.

## Running tests

```bash
pip install pytest
pytest pipeline/tests/ -v
```
