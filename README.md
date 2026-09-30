# Stage 1: Autoencoder training (`train_autoencoder.py` + `dataset.py`)

Stage 1 of the 3D latent diffusion pipeline. Trains a 3D `AutoencoderKL` (VAE)
with an adversarial + perceptual loss to compress T1 MRI volumes into a small
latent space. Stage 2 (`train_diffusion.py`) trains a diffusion model in that
latent space, using this stage's frozen encoder/decoder.

Converted from `3d_ldm_fedNeuro-ADNI.ipynb`.

## Data pipeline (`dataset.py`)

### `MonaiMRIDatasetADNI` / `MonaiMRIDatasetNACC`
Both take a path to a CSV (not a directory — despite the `root_dir` parameter
name) and load one T1 volume per row via SimpleITK.

| | ADNI | NACC |
|---|---|---|
| Image path column | `FED_Path` | `FED_Path` |
| Diagnosis column | `DX` (`"CN"` / `"Dementia"`) | `DX_ADSP` (`1` / `3`) |
| Returns `dx`/`sex` strings | yes | `sex` only |

Both now return a `label` tensor (long, 0 or 1) via `_adni_label` /
`_nacc_label`, mapped to the shared scheme:

```python
CN_LABEL, AD_LABEL, NULL_LABEL = 0, 1, 2   # NULL_LABEL is for classifier-free guidance (train_diffusion.py)
```

**NACC label mapping is an assumption**, not verified against the NACC data
dictionary: `DX_ADSP` 1 → CN, 3 → AD. These are the only two values present in
the ADvsCN split CSVs. Confirm before trusting NACC-derived diagnosis labels.

Per-item processing (`_load_t1_tensor`): load with SimpleITK → pad to
`output_size` if smaller (`Padding`) → convert to numpy `(D, H, W)` → min-max
normalize to `[0, 1]` → tensor with a channel dim, `(1, D, H, W)`.

### `Padding`
Only pads — it resamples (via B-spline) up to the target shape when the
volume is *smaller* than `output_size` on any axis, and otherwise passes the
volume through unchanged (it does not crop down oversized volumes).

### `Resample`
A voxel-spacing resampler. Defined but not used by either ADNI/NACC dataset
class or by the training scripts.

### Dead / unused code
- `MonaiMRIDataset` (the generic, non-ADNI/NACC class) references an
  undefined `row` variable in `__getitem__` and will raise `NameError` if
  instantiated and indexed. Not used by any training script.
- `resize_3d` (scipy zoom-based resize) is defined but unused.

### Fixed bugs (as of this doc)
- `__len__` used to return the CSV *line count* (including the header row),
  one more than `len(self.df)`. This caused an `IndexError` on the last
  batch of every epoch. Now returns `len(self.df)`.
- `MonaiMRIDatasetNACC` used to read the image path from `item[1]`
  (the `AGE` column after a comma-split), not `FED_Path`. Fixed to read the
  dataframe like the ADNI class.

## Training script (`train_autoencoder.py`)

### Model
- **`AutoencoderKL`**: 3D, `in/out_channels=1`, `num_channels=(64,128,256)`
  (two downsampling stages), `latent_channels=8`, attention only at the
  deepest level (`attention_levels=(False, False, True)`). ~36.7M params.
- **`PatchDiscriminator`**: 3D, 4 layers, 64 base channels. ~44.6M params.
  Scores local patches as real/fake (PatchGAN), not the whole volume.
- **`PerceptualLoss`**: frozen 2D SqueezeNet applied to slices
  (`is_fake_3d=True`, `fake_3d_ratio=0.2` samples 20% of slices per call).
  Downloads pretrained weights via `torchvision`/`torch.hub` on first use —
  this fails/hangs on compute nodes with no outbound internet.

### Losses (generator side, `loss_g`)
| Term | Weight | Purpose |
|---|---|---|
| Weighted L1 (`weighted_l1_loss`) | 1.0 | Foreground voxels (intensity > 0.05) weighted 10x vs background, so the model can't win by just reproducing black background. |
| KL (`KL_loss`) | `1e-6` | Regularizes the latent toward a standard normal so Stage 2 has a well-behaved space to model. Deliberately tiny so it doesn't blur reconstructions. |
| Perceptual | `0.01` | SqueezeNet feature-space similarity, for perceptual/structural sharpness beyond pixelwise L1. |
| Adversarial (generator) | `0.01` | Pushes reconstructions to fool the discriminator. Only added after `--warmup-epochs`. |

### Discriminator loss
`discriminator_loss = 0.5 * (loss_d_fake + loss_d_real)`, trained **unscaled**
(previously multiplied by `adv_weight=0.01`, which had no clear purpose since
the discriminator has no competing loss term to balance against — removed).
Only trains after `--warmup-epochs`. Use `--disc-lr` to give the
discriminator its own learning rate if you want it to learn faster/slower
than the generator; defaults to `--lr`.

### Warmup
`if epoch > args.warmup_epochs` gates both the generator's adversarial term
and all discriminator training. Default 5 epochs: the autoencoder first
learns plain reconstruction, then the GAN loss kicks in to sharpen it.

### Per-step flow
1. Forward pass → `reconstruction, z_mu, z_sigma`.
2. Compute `loss_g` (L1 + KL + perceptual [+ adversarial after warmup]) →
   backward → `optimizer_g.step()`.
3. After warmup: score `reconstruction.detach()` as fake and `images` as
   real → `discriminator_loss` → backward → `optimizer_d.step()`.
4. Print a status line every `--log-every` steps.

### Checkpointing
- Every 100 epochs (`epoch % 100 == 0`): full training-state checkpoint
  (`autoencoder_state_epoch_{N}.pt`) with both model state dicts, both
  optimizer state dicts, and the epoch's losses. Resumable, but the script
  currently has no `--resume` flag to load one back in.
- Every 80 epochs: logs a middle-slice image (original vs. reconstruction)
  to wandb.
- At the end: saves **only the autoencoder weights** (`--ckpt-name`,
  ~147 MB) — this is the file `train_diffusion.py --autoencoder-ckpt` loads.
  The discriminator and its optimizer are not part of this final artifact.
- No per-epoch validation loop and no "best" checkpoint — `val_loader` is
  only used once, at the very end, for a single-batch MSE/MAE sanity check.

### Known gaps
- No mixed precision (unlike `train_diffusion.py`, which uses `autocast`),
  which is part of why this stage is memory-hungry (~46GB observed on an
  A6000 at batch size 2, 96×112×96 volumes).
- `l1_loss = L1Loss()` is constructed but never called — `weighted_l1_loss`
  is used instead.
- No early stopping / no best-checkpoint tracking.

### Key CLI args
| Flag | Default | Notes |
|---|---|---|
| `--train-csv` / `--val-csv` | ADNI paths | Must have `FED_Path`, `DX`, `SEX` columns |
| `--output-size` | `96 112 96` | Volumes smaller than this get padded up |
| `--batch-size` | 2 | Memory-bound at this volume size |
| `--epochs` | 500 | |
| `--warmup-epochs` | 5 | Epochs before adversarial loss/discriminator training starts |
| `--lr` / `--disc-lr` | `1e-4` / `--lr` | Separate discriminator LR (see above) |
| `--out-dir` | `/data/ashaji/FedNeuro/abhi` | CARC-specific default; override on other clusters |
| `--wandb-mode` | `online` | Use `disabled` for quick tests, `offline` on no-internet compute nodes |

### Usage
```bash
python train_autoencoder.py \
    --train-csv /path/ADNI_ADvsCN_train_fedpath.csv \
    --val-csv /path/ADNI_ADvsCN_test_fedpath.csv \
    --epochs 500 --batch-size 2 \
    --out-dir /path/to/output --ckpt-name autoencoder_exp1_adni.pth
```

## Relationship to Stage 2 and federation
- Stage 2 only needs the final autoencoder weights file — it is frozen
  (`requires_grad_(False)`) and never updated during diffusion training.
- The autoencoder is intentionally **unconditional** (no diagnosis input) —
  see the discussion in-session: conditioning belongs in Stage 2's UNet, not
  here, since the decoder should turn any latent into an image regardless of
  diagnosis.
- For a federated Stage 1: the autoencoder weights (147MB) are the natural
  thing to aggregate (FedAvg). The discriminator and perceptual loss are
  training aids; keeping the discriminator local to each site (not
  aggregated) is the simpler starting design. GroupNorm (`norm_num_groups`)
  is used throughout, so there's no BatchNorm running-stats mismatch issue
  across non-IID sites.
