# Nasal Deformation Pipeline

The nasal deformation pipeline applies clinically-realistic 2D morphological transforms to generated face images, creating paired datasets of "before" (normal) and "after" (pathological) images for rhinoplasty AI training.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Supported Morphologies](#supported-morphologies)
- [Clinical Viewpoints](#clinical-viewpoints)
- [Algorithm Routing](#algorithm-routing)
- [Mask Cascade](#mask-cascade)
- [Parameters Reference](#parameters-reference)
- [Storage Layout](#storage-layout)
- [Automatic vs. Manual Deformation](#automatic-vs-manual-deformation)
- [Adding a New Morphology](#adding-a-new-morphology)

---

## Overview

For each generated patient dataset, the deformation pipeline:

1. Loads the generated face image for a given viewpoint.
2. Acquires a nose segmentation mask (via a 4-tier cascade).
3. Routes to the view-calibrated deformation algorithm.
4. Saves the deformed image alongside the original.

All 9 clinical viewpoints are processed independently, with algorithms calibrated per viewpoint.

---

## Architecture

```
DeformationService (backend/app/services/deformation_service.py)
  │
  ├── DeformationRequest / BatchDeformationRequest (schemas)
  │
  └── NasalDeformationEngine (deformation/engine.py)
        │
        ├── validate_parameters()   → normalize type, strength, direction
        ├── Mask cascade            → acquire nose segmentation mask
        ├── apply_deformation()     → run algorithm for one image
        └── _route_algorithm()      → dispatch to view+type-specific function
              │
              ├── frontal_saddle_nose(), saddle_nose_deformation(), basal_saddle_nose(), helicopter_saddle_nose()
              ├── frontal_hump_nose(), hump_nose_deformation(), basal_hump_nose(), helicopter_hump_nose()
              ├── frontal_broad_nose(), profile_broad_nose(), basal_broad_nose(), helicopter_broad_nose()
              └── deviated_nose_deformation(), basal_deviated_nose_deformation(), helicopter_deviated_nose_deformation()
```

---

## Supported Morphologies

### `saddle` — Saddle Nose Deformity

| Property | Value |
|---|---|
| Clinical description | Depression of the nasal dorsum (bridge), creating a concave profile. Caused by cartilage/bone loss. |
| Default strength | `0.5` |
| Strength range | `0.1` – `0.9` |
| Supports direction | No |
| Clinical notes | Highest visual impact on profile views (left/right 60°). Frontal view shows narrowed bridge. |

### `hump` — Dorsal Hump

| Property | Value |
|---|---|
| Clinical description | Convex prominence of the nasal bridge formed by excess bone or cartilage. |
| Default strength | `0.5` |
| Strength range | `0.1` – `0.9` |
| Supports direction | No |
| Clinical notes | Most visible in profile views. Basal/helicopter views show increased nasal width. Strength capped at 0.45 for basal and helicopter views to avoid unrealistic results. |

### `broad` — Broad/Wide Nose

| Property | Value |
|---|---|
| Clinical description | Lateral widening of the nasal tip, alar base, and bridge beyond aesthetic norms. |
| Default strength | `0.5` |
| Strength range | `0.1` – `0.8` |
| Supports direction | No |
| Clinical notes | Alar flaring and tip widening visible in all views. Profile views show increased nasal depth. |

### `deviated` — Deviated Nose

| Property | Value |
|---|---|
| Clinical description | Asymmetric lateral displacement (C-shaped or S-shaped curvature) of the septum and nasal tip. |
| Default strength | `0.35` |
| Strength range | `0.1` – `0.7` |
| Supports direction | **Yes** (`left` or `right`) |
| Clinical notes | Simulates septal deviation with consistent directional shift across all clinical angles. Direction specifies which side the tip deviates toward. |

---

## Clinical Viewpoints

All 9 canonical viewpoints and their deformation routing:

| Key | Description | Angle | Deformation function group |
|---|---|---|---|
| `frontal` | Direct face-on view | 0° | `frontal_*` functions |
| `left_20` | Slight left oblique | 20° left | `*_deformation(..., contour_side="left")` |
| `left_40` | Medium left oblique | 40° left | `*_deformation(..., contour_side="left")` |
| `left_60` | Strong left profile | 60° left | `*_deformation(..., contour_side="left")` |
| `right_20` | Slight right oblique | 20° right | `*_deformation(..., contour_side="right")` |
| `right_40` | Medium right oblique | 40° right | `*_deformation(..., contour_side="right")` |
| `right_60` | Strong right profile | 60° right | `*_deformation(..., contour_side="right")` |
| `basal` | Upward worm's-eye view | Chin up | `basal_*` functions |
| `helicopter` | Downward bird's-eye view | Top down | `helicopter_*` functions |

> [!NOTE]
> The engine normalizes any filename stem or raw view string to one of these 9 canonical keys using `normalize_viewpoint_key()`. For example, `"left40"`, `"left_40"`, and `"LEFT40"` all resolve to `"left_40"`.

---

## Algorithm Routing

The `_route_algorithm()` method dispatches to exactly one calibrated function per (morphology, viewpoint) pair:

```
saddle
  frontal             → frontal_saddle_nose()
  left_*, right_*     → saddle_nose_deformation(contour_side=…)
  basal               → basal_saddle_nose()
  helicopter          → helicopter_saddle_nose()

hump
  frontal             → frontal_hump_nose()
  left_*, right_*     → hump_nose_deformation(contour_side=…)
  basal               → basal_hump_nose()       [strength capped at 0.45]
  helicopter          → helicopter_hump_nose()  [strength capped at 0.45]

broad
  frontal             → frontal_broad_nose()
  left_*, right_*     → profile_broad_nose()
  basal               → basal_broad_nose()
  helicopter          → helicopter_broad_nose()

deviated
  frontal             → deviated_nose_deformation(direction=…)
  left_*, right_*     → deviated_nose_deformation(direction=…)
  basal               → basal_deviated_nose_deformation(direction=…)
  helicopter          → helicopter_deviated_nose_deformation(direction=…)
```

If no route is found (e.g. an unregistered view + type combination), the engine raises `ValueError: Unmapped deformation routine`.

---

## Mask Cascade

The nose mask determines which pixels the deformation algorithm operates on. The engine uses a **4-tier cascade**, trying each source in order until a usable mask is obtained:

```
Tier 1: BiSeNet real-time segmentation
         → Uses the BiSeNet face parsing model to segment the nose region.
         → Most accurate. Requires the BiSeNet model weights to be available.

Tier 2: Patient pre-stored mask
         → If the patient has a manually-created or previously-generated mask
           saved in storage, load it.

Tier 3: Reference mask
         → Use a generic reference nose mask scaled to the image size.

Tier 4: Geometric ellipse fallback
         → Construct an elliptical mask centred in the lower-middle third of the image.
         → Always available. Least accurate.
```

After acquisition, the mask is:
1. **Resized** to match the image dimensions.
2. **Cleaned** via `clean_nose_mask(mask, viewpoint)` — applies morphological operations (erosion, dilation) calibrated for each viewpoint to remove stray pixels and smooth edges.

The final binary mask is stored in `DeformationExecutionResult.mask_used` for diagnostics.

---

## Parameters Reference

### `deformation_type`

One of `"saddle"`, `"hump"`, `"broad"`, `"deviated"`. Case-insensitive; leading/trailing whitespace stripped.

### `strength`

Float in `[0.0, 1.0]`. Controls the magnitude of the deformation.

| Value | Effect |
|---|---|
| `0.1–0.3` | Subtle, clinically mild presentation |
| `0.4–0.6` | Moderate, clearly visible (recommended for training data) |
| `0.7–0.9` | Severe presentation (saddle/hump only) |

Defaults per morphology:

| Morphology | Default |
|---|---|
| `saddle` | `0.5` |
| `hump` | `0.5` |
| `broad` | `0.5` |
| `deviated` | `0.35` |

### `direction`

Only applicable to `deviated`. One of `"left"` or `"right"` (default: `"right"`). Ignored for all other morphology types.

---

## Storage Layout

Deformed images are stored alongside the originals:

```
storage/
└── datasets/
    └── <dataset_id>/
        └── <patient_id>/
            └── <viewpoint>/
                ├── original.png          ← Generated image (pre-deformation)
                └── deformed_<type>.png   ← Post-deformation image
```

A `DeformationRecord` is written to the database for each processed image, tracking:
- `deformation_type`, `strength`, `direction`
- `mask_source` (which tier of the cascade was used)
- `mask_pixels`, `modified_pixels` (from `DeformationExecutionResult.metrics`)
- `processing_time_ms`
- `status`: `COMPLETED` or `FAILED`

---

## Automatic vs. Manual Deformation

### Automatic (triggered by background worker)

When a generation job reaches the `PACKAGING` stage, the worker automatically applies nasal deformation using the parameters stored on the dataset:

- `dataset.morphology_type` → `deformation_type`
- `dataset.deformation_strength` → `strength` (default: `0.6`)
- `dataset.deviated_direction` → `direction` (default: `"right"`)

If `morphology_type` is `"none"` or unset, deformation is skipped.

### Manual (via API)

You can trigger deformation for a specific image or all viewpoints of a dataset:

```bash
# Single image
POST /api/v1/deformation/apply
{
  "image_id": 42,
  "deformation_type": "hump",
  "strength": 0.6
}

# All viewpoints for a dataset
POST /api/v1/deformation/batch
{
  "dataset_id": 7,
  "deformation_type": "broad",
  "strength": 0.5
}
```

See [api-reference.md](api-reference.md) for full request/response schemas.

---

## Adding a New Morphology

1. **Add metadata** to `MORPHOLOGY_METADATA` in `deformation/engine.py`:

```python
"my_type": {
    "name": "My Morphology",
    "description": "...",
    "default_strength": 0.5,
    "min_strength": 0.1,
    "max_strength": 0.9,
    "supports_direction": False,
    "clinical_notes": "...",
}
```

2. **Implement algorithm functions** (at minimum one per viewpoint group):

```python
# In deformation/algorithms/my_type.py
def frontal_my_type(image_bgr, mask, strength=0.5): ...
def profile_my_type(image_bgr, mask, strength=0.5, contour_side="left"): ...
def basal_my_type(image_bgr, mask, strength=0.5): ...
def helicopter_my_type(image_bgr, mask, strength=0.5): ...
```

3. **Add routing** in `NasalDeformationEngine._route_algorithm()`:

```python
elif dtype == "my_type":
    if view == "frontal":
        return frontal_my_type(image_bgr, mask, strength=strength)
    elif view.startswith("left") or view.startswith("right"):
        return profile_my_type(image_bgr, mask, strength=strength)
    elif view == "basal":
        return basal_my_type(image_bgr, mask, strength=strength)
    elif view == "helicopter":
        return helicopter_my_type(image_bgr, mask, strength=strength)
```

4. Update the `morphology_type` enum in the dataset schema if you want it exposed via the API.
