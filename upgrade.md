# Upgrade Roadmap — Spot-the-Ball Model Improvements

> **Audience:** Developer / ML practitioner already familiar with the repo.  
> **Goal:** Incrementally improve the ball-localisation accuracy of the model pipeline, starting from the existing U-Net baseline (`Unet - Final.py`) and progressing toward a production-quality keypoint detector.

---

## Background

The current pipeline:

1. Trains a **U-Net** (ResNeXt-50 encoder, ImageNet weights) to produce a binary segmentation mask of the removed-ball region.
2. Post-processes the mask (centroid / largest-blob) to get an `(x, y)` ball coordinate.
3. Optionally synthesises the ball back into the image via a GAN (`Gan_training.py`).

The upgrades below address the model in layers — objective formulation first, then architecture, losses, augmentation, training, and inference.

---

## Recommended Formulation: Heatmap / Keypoint

**Recommendation: use a Gaussian heatmap target rather than direct coordinate regression.**

| Approach | Pros | Cons |
|---|---|---|
| Binary mask (current) | Simple; easy to label | Loses soft spatial signal; post-processing fragile |
| **Gaussian heatmap** (recommended) | Soft spatial signal; smooth gradients; scale-invariant | Requires heatmap generation step |
| Direct `(x, y)` regression | Tiny output head; fast | Sensitive to outliers; hard to debug |

A Gaussian heatmap places a 2-D Gaussian blob centred at the ball location. The model learns to predict a probability density surface; the predicted coordinate is recovered with **soft-argmax** (differentiable) or plain argmax at inference. This is the dominant approach in human-pose estimation and works well for small, single-keypoint tasks.

---

## Phase 0 — Baseline Health Check *(current state)*

Before any new work, establish a clean measurement baseline.

### Tasks
- [ ] Pin all dependency versions in `requirements.txt` (add `.txt` extension and exact versions).
- [ ] Split dataset reproducibly (fixed random seed) into train / val / test; log split sizes.
- [ ] Compute and record baseline metrics on the held-out test set: **IoU**, **Dice**, **Mean Distance Error (MDE)** in pixels, **PCK@k** (percentage of predictions within `k` pixels of ground truth, e.g. `k = 20, 40`).
- [ ] Add a minimal experiment log (CSV or `wandb`/`mlflow` run) capturing: model name, encoder, loss, LR, epoch of best val score, val Dice, test MDE.

### Done when
- [ ] Baseline metrics are written down and reproducible (re-running training gets within ~1% of recorded numbers).
- [ ] Experiment log exists with at least one entry.

---

## Phase 1 — Better Objective: Gaussian Heatmap Target

### 1.1 Heatmap Generation

Replace binary masks with soft Gaussian targets.

```python
import numpy as np

def make_gaussian_heatmap(height: int, width: int,
                           cx: float, cy: float,
                           sigma: float = 10.0) -> np.ndarray:
    """
    Create a 2-D Gaussian heatmap (values in [0, 1]).

    Args:
        height, width: output spatial dimensions.
        cx, cy: ball centre in pixel coordinates.
        sigma: Gaussian standard deviation in pixels.
    """
    xs = np.arange(width,  dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    heatmap = np.exp(-((xv - cx) ** 2 + (yv - cy) ** 2) / (2 * sigma ** 2))
    return heatmap  # shape (H, W), peak = 1.0 at (cx, cy)
```

- **sigma** controls the target spread. Start with `sigma ≈ 10 px` (roughly ball radius) and tune per image resolution.
- Store heatmaps alongside images (e.g. as `.npy` files) at dataset-creation time to avoid re-computation.

### 1.2 Coordinate Extraction

```python
import torch
import torch.nn.functional as F

def soft_argmax_2d(heatmap: torch.Tensor) -> torch.Tensor:
    """
    Differentiable soft-argmax over a spatial heatmap.

    Args:
        heatmap: (B, 1, H, W) tensor, values > 0.
    Returns:
        coords: (B, 2) tensor of (x, y) in [0, 1] normalised coordinates.
    """
    B, _, H, W = heatmap.shape
    flat = heatmap.view(B, -1)
    weights = F.softmax(flat * 100, dim=-1)          # temperature = 100
    ys = torch.arange(H, dtype=torch.float32, device=heatmap.device) / (H - 1)
    xs = torch.arange(W, dtype=torch.float32, device=heatmap.device) / (W - 1)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0)  # (2, H*W)
    coords = weights @ grid.T  # (B, 2)
    return coords

# At inference (no gradient needed):
def argmax_2d(heatmap: torch.Tensor):
    B, _, H, W = heatmap.shape
    idx = heatmap.view(B, -1).argmax(dim=-1)
    y = (idx // W).float()
    x = (idx %  W).float()
    return torch.stack([x, y], dim=-1)   # pixel coords
```

### 1.3 Loss Options

| Loss | When to use |
|---|---|
| **MSE on heatmap** | Simple baseline; works well with Gaussian targets |
| **Focal MSE** (`α·(1-p)^γ·MSE`) | Downweights easy background pixels; useful when ball region is tiny |
| **Dice / Tversky** | Good for unbalanced foreground/background (current DiceLoss is reasonable) |
| **Wing Loss / SmoothL1 (Huber)** on extracted coordinates | Direct supervision on the final `(x, y)` output; combine with heatmap loss |

Recommended combined loss:

```python
total_loss = λ_heatmap * mse_loss(pred_heatmap, gt_heatmap) \
           + λ_coord   * F.smooth_l1_loss(pred_coords, gt_coords)
# Start with λ_heatmap = 1.0, λ_coord = 10.0
```

### Done when
- [ ] Dataset pipeline generates Gaussian heatmap targets.
- [ ] Model output is a single-channel heatmap; soft-argmax/argmax extracts `(x, y)`.
- [ ] Combined heatmap + coordinate loss replaces pure Dice loss.
- [ ] Val MDE improves over Phase 0 baseline; logged.

---

## Phase 2 — Architecture Upgrades

Apply in order of expected impact vs. engineering effort.

### 2.1 Drop-in Encoder Upgrades (low effort)

The existing `smp.Unet` already supports swapping encoders. Try, in order:

```python
# In Unet - Final.py, change ENCODER:
ENCODER = 'efficientnet-b4'        # good accuracy/speed trade-off
ENCODER = 'convnext_base'          # state-of-the-art CNN; smp >= 0.3.3
ENCODER = 'timm-efficientnetv2-m'  # via timm integration in smp
```

All accept `ENCODER_WEIGHTS = 'imagenet'`.

### 2.2 Attention U-Net (moderate effort)

Attention gates suppress irrelevant activations and focus the decoder on the ball region.

```python
model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    decoder_attention_type='scse',   # concurrent spatial & channel squeeze-excitation
    activation=ACTIVATION,
)
```

Available natively in `segmentation_models_pytorch`.

### 2.3 UNet++ (nested dense skip connections)

```python
model = smp.UnetPlusPlus(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    activation=ACTIVATION,
)
```

UNet++ bridges the semantic gap between encoder and decoder more densely than plain U-Net, often improving small-object localisation.

### 2.4 HRNet-style High-Resolution Decoder (higher effort)

For very small ball regions, consider an HRNet backbone that maintains high-resolution feature maps throughout:

```python
# Requires timm HRNet weights:
ENCODER = 'tu-hrnet_w32'
model = smp.Unet(encoder_name=ENCODER, encoder_weights='imagenet', ...)
```

HRNet avoids the information bottleneck of standard encoder-decoder downsampling.

### 2.5 Pretrained / Self-Supervised Features

- **ImageNet pretrained encoders** (already in use) — continue leveraging these.
- **DINO / DINOv2 features (optional):** Vision Transformer features from DINOv2 (Meta) are powerful zero-shot spatial descriptors. They can be used as a frozen feature extractor with a lightweight heatmap head on top, especially when labelled data is scarce.

```python
# Pseudo-code — DINOv2 as frozen backbone + heatmap head
import torch
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
# Freeze backbone, train only the heatmap projection head
for param in dinov2.parameters():
    param.requires_grad = False
```

### Done when
- [ ] At least one encoder upgrade tested and logged.
- [ ] Attention U-Net or UNet++ tested and logged.
- [ ] Best architecture on val MDE identified and committed as the new baseline.

---

## Phase 3 — Training, Augmentation & Inference Improvements

### 3.1 Augmentation Hardening

The current augmentation set (`Unet - Final.py`, lines 119–158) is reasonable. Add:

- **`albu.CoarseDropout`** (cutout) — simulates occlusion.
- **`albu.GridDistortion`** / **`albu.ElasticTransform`** — mimics print/scan artefacts common in newspaper images.
- **`albu.RandomSunFlare`** / **`albu.RandomFog`** — lighting variations.
- **`albu.JPEG compression noise`** (`albu.ImageCompression`) — important for web-sourced images.
- Maintain existing flips, colour jitter, and perspective transforms.

> **Note:** For heatmap targets, ensure spatial transforms are applied identically to both image and heatmap (albumentations `additional_targets` supports this).

### 3.2 LR Scheduling

Replace the manual LR step (line 319) with a proper scheduler:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=1e-6
)
# Call scheduler.step() after each epoch
```

Or use a warmup + cosine schedule via `timm.scheduler`.

### 3.3 Multi-Stage Coarse-to-Fine Pipeline

For higher precision, add a crop-refinement stage:

1. **Stage 1 (coarse):** Run the full-resolution model; get a coarse `(x, y)` estimate.
2. **Stage 2 (refine):** Crop a `256×256` patch centred on the Stage 1 prediction; run a second (possibly smaller) model on the crop; get a sub-pixel refined coordinate.

```python
# Pseudocode
coarse_xy = predict(model_coarse, full_image)         # Stage 1
crop = extract_crop(full_image, coarse_xy, size=256)
refined_xy_local = predict(model_refine, crop)         # Stage 2
refined_xy = coarse_xy + (refined_xy_local - 128)      # map back to full image
```

### 3.4 Test-Time Augmentation (TTA)

Average predictions over several augmented versions of the test image:

```python
import ttach  # pip install ttach
model_tta = ttach.SegmentationTTAWrapper(
    model,
    ttach.aliases.hflip_transform(),   # horizontal flip
    merge_mode='mean',
)
pred = model_tta(x_tensor)
```

Or implement manually: predict on original + horizontally flipped; average heatmaps; extract argmax.

### 3.5 Uncertainty / Confidence Estimation

- **MC Dropout:** Enable `dropout` layers at inference time (`model.train()` with `torch.no_grad()`) and sample N forward passes. The variance across predictions is a proxy for uncertainty.
- **Ensemble:** Train 3–5 models with different seeds; average their heatmaps. The standard deviation of predicted coordinates gives a confidence interval.
- Log confidence alongside predictions to flag low-confidence entries for manual review.

### Done when
- [ ] Augmentation pipeline extended; no regression in val MDE.
- [ ] Cosine LR scheduler in use; LR curve logged.
- [ ] TTA implemented; test MDE with TTA vs. without logged.
- [ ] Optional: coarse-to-fine pipeline implemented and evaluated.
- [ ] Optional: uncertainty estimates available per prediction.

---

## Experiment Tracking (Minimal)

Add lightweight tracking so every run is reproducible and comparable. Recommended tools (pick one):

| Tool | Setup effort | Notes |
|---|---|---|
| **CSV log** | Minimal | Append one row per run to `experiments.csv` |
| **Weights & Biases (`wandb`)** | Low | `pip install wandb`; free tier sufficient |
| **MLflow** | Low–medium | Self-hosted option |

Minimum fields to log per run:

```
run_id, date, encoder, architecture, loss, augmentation_set,
epochs, best_val_dice, best_val_mde_px, test_mde_px, pck20, pck40,
notes
```

---

## Summary Metrics

Track these on the **held-out test set** after each phase:

| Metric | Description | Target direction |
|---|---|---|
| **Dice** | Overlap of predicted vs. ground-truth mask | ↑ higher |
| **IoU** | Intersection over Union | ↑ higher |
| **MDE** (Mean Distance Error, px) | Euclidean distance between predicted and true ball centre | ↓ lower |
| **PCK@20** | % predictions within 20 px of true centre | ↑ higher |
| **PCK@40** | % predictions within 40 px of true centre | ↑ higher |

---

## Risks & Ethics

This project is intended as a game-playing aid for the "Spot the Ball" newspaper competition (e.g., BOTB — Best of the Best).

- **Game integrity:** An automated solver could give an unfair advantage in competitions with real prizes. Check the terms and conditions of any competition before using this tool to submit entries.
- **Commercial use:** The newspaper images used as training data may be subject to copyright. Ensure you have appropriate rights or licences before scraping, storing, or distributing them.
- **Model bias:** A model trained on a narrow dataset (one competition's image style) may fail or behave unpredictably on out-of-distribution images. Always validate on a diverse test set before deployment.
- **Responsible disclosure:** If you discover a systematic weakness in a competition's image-removal process, consider disclosing it responsibly to the competition operator rather than exploiting it.

---

## Quick-Start Checklist

- [ ] Phase 0: Baseline metrics recorded
- [ ] Phase 1: Gaussian heatmap pipeline in place; MDE baseline set
- [ ] Phase 2: Best architecture variant identified and logged
- [ ] Phase 3: TTA and LR scheduler active; final test metrics logged
- [ ] Experiment log up to date with all runs
