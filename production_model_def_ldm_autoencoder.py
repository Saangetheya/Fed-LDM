"""
Federated Stage 1 of the 3D LDM: AutoencoderKL trained with L1 + KL + perceptual
+ adversarial losses, in the FLINT PyTorchDef format.

Converted from train_autoencoder.py (same model, losses and weights). What is
different for federation:

- Only the autoencoder is a registered submodule, so only its weights are in
  state_dict() and only it is aggregated. FLINT aggregates everything in
  state_dict() (see PyTorchModelOps.set_model_weights).
- The PatchDiscriminator stays LOCAL to each learner. FLINT rebuilds the model
  from model_def.pkl for every training task, so the discriminator and its
  optimizer are saved to the learner's disk at the end of fit() and reloaded at
  the start of the next fit(). The same file keeps a count of local epochs
  completed, which drives the adversarial warmup across rounds.
- The perceptual loss (frozen SqueezeNet/LPIPS) is created inside fit(), so it
  is never pickled, sent or aggregated.
- Mixed precision (bfloat16 autocast) is on by default: the centralized run used
  ~46 GB at batch size 2 in fp32, which is the whole memory of an L40S.
- The generator optimizer is recreated every round (standard FedAvg practice).
- checkpoint_state_dict() returns a plain AutoencoderKL state dict, so the saved
  community checkpoint loads directly in train_diffusion.py.
"""

import os
import socket

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from core.models.model_def import PyTorchDef
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator

# ---------------------------------------------------------------------------
# Settings (same values as train_autoencoder.py unless noted)
# ---------------------------------------------------------------------------
AE_KWARGS = dict(
    spatial_dims=3, in_channels=1, out_channels=1,
    num_channels=(64, 128, 256), latent_channels=8, num_res_blocks=2,
    norm_num_groups=16, attention_levels=(False, False, True),
)
DISC_KWARGS = dict(spatial_dims=3, num_layers_d=4, num_channels=64, in_channels=1, out_channels=1)

ADV_WEIGHT = 0.01
PERCEPTUAL_WEIGHT = 0.01
KL_WEIGHT = 1e-6
WARMUP_EPOCHS = 5          # counted in LOCAL epochs completed on this learner, across rounds
DEFAULT_LR = 1e-4          # used only if the yaml's optimizer config has no learning rate
DISC_LR = None             # None = same as the generator learning rate
USE_AMP = True             # bfloat16 autocast; set False to match the centralized fp32 run exactly
DATALOADER_WORKERS = 0     # FLINT runs training in a worker process; keep 0 unless you've verified >0 works
# Evaluation runs on every requested split (train/val/test) every round. Cap the
# number of volumes per evaluation to keep rounds fast; None = evaluate all.
EVAL_MAX_SAMPLES = 64
# Where each learner keeps its local discriminator state between rounds. The
# hostname keeps learners separate if several ever share a filesystem.
LOCAL_STATE_DIR = "/tmp/FLINT/ldm_local_state"


def weighted_l1_loss(pred, target, thr=0.05, fg_weight=10.0, bg_weight=1.0):
    weights = torch.where(
        target > thr,
        torch.tensor(fg_weight, device=target.device, dtype=target.dtype),
        torch.tensor(bg_weight, device=target.device, dtype=target.dtype),
    )
    return (weights * torch.abs(pred - target)).mean()


def kl_loss(z_mu, z_sigma):
    kl = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
    return torch.sum(kl) / kl.shape[0]


class ProductionModel(nn.Module, PyTorchDef):

    def __init__(self, batch_size=2, optimizer_config=None, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.optimizer_config = optimizer_config
        # The ONLY registered submodule: this is what FLINT aggregates.
        self.autoencoder = AutoencoderKL(**AE_KWARGS)

    # ----- FLINT interface ---------------------------------------------------
    def forward(self, x):
        return self.autoencoder(x)

    def get_model(self):
        return self

    def checkpoint_state_dict(self):
        """Plain AutoencoderKL state dict (no 'autoencoder.' prefix), for Stage 2."""
        return self.autoencoder.state_dict()

    # ----- helpers -------------------------------------------------------------
    def _optimizer_kwargs(self):
        kw = {}
        try:
            kw = dict(self.optimizer_config.optimizer_pb_kwargs)
        except Exception:
            pass
        lr = float(kw.get("learning_rate", DEFAULT_LR))
        betas = (float(kw.get("beta_1", 0.9)), float(kw.get("beta_2", 0.999)))
        eps = float(kw.get("epsilon", 1e-8))
        return lr, betas, eps

    @staticmethod
    def _local_state_path():
        return os.path.join(LOCAL_STATE_DIR, socket.gethostname(), "local_state.pt")

    @staticmethod
    def _as_torch_dataset(dataset):
        return dataset.x if hasattr(dataset, "x") else dataset

    @staticmethod
    def _autocast(device):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                              enabled=USE_AMP and device.type == "cuda")

    # ----- training ------------------------------------------------------------
    def fit(self, dataset, val_dataset=None, epochs=1, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.autoencoder.train()

        lr, betas, eps = self._optimizer_kwargs()
        optimizer_g = torch.optim.Adam(self.autoencoder.parameters(), lr=lr, betas=betas, eps=eps)

        # Local (non-aggregated) discriminator, restored from the previous round.
        discriminator = PatchDiscriminator(**DISC_KWARGS).to(device)
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=DISC_LR or lr)
        epochs_done = 0
        state_path = self._local_state_path()
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=device)
            discriminator.load_state_dict(state["discriminator"])
            optimizer_d.load_state_dict(state["optimizer_d"])
            epochs_done = int(state.get("epochs_done", 0))
            print(f"Restored local discriminator state ({epochs_done} local epochs done)", flush=True)

        adv_loss = PatchAdversarialLoss(criterion="least_squares")
        perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze",
                                    is_fake_3d=True, fake_3d_ratio=0.2).to(device)
        perceptual.eval()

        loader = DataLoader(self._as_torch_dataset(dataset), batch_size=self.batch_size,
                            shuffle=True, num_workers=DATALOADER_WORKERS)
        history = {"loss": [], "recons_loss": [], "gen_loss": [], "disc_loss": []}

        for local_epoch in range(epochs):
            global_epoch = epochs_done + local_epoch
            adversarial = global_epoch > WARMUP_EPOCHS   # same gate as the centralized script
            sums = {"loss": 0.0, "recons_loss": 0.0, "gen_loss": 0.0, "disc_loss": 0.0}
            n_steps = 0

            for batch in loader:
                images = batch["t1_image"].to(device, non_blocking=True)

                optimizer_g.zero_grad(set_to_none=True)
                with self._autocast(device):
                    reconstruction, z_mu, z_sigma = self.autoencoder(images)
                reconstruction = reconstruction.float()
                recons = weighted_l1_loss(reconstruction, images.float())
                p_loss = perceptual(reconstruction, images.float())
                loss_g = recons + KL_WEIGHT * kl_loss(z_mu.float(), z_sigma.float()) + PERCEPTUAL_WEIGHT * p_loss

                gen = torch.zeros((), device=device)
                if adversarial:
                    with self._autocast(device):
                        logits_fake = discriminator(reconstruction.contiguous())[-1]
                    gen = adv_loss(logits_fake.float(), target_is_real=True, for_discriminator=False)
                    loss_g = loss_g + ADV_WEIGHT * gen

                loss_g.backward()
                optimizer_g.step()

                disc = torch.zeros((), device=device)
                if adversarial:
                    optimizer_d.zero_grad(set_to_none=True)
                    with self._autocast(device):
                        logits_fake = discriminator(reconstruction.detach().contiguous())[-1]
                        logits_real = discriminator(images.contiguous())[-1]
                    disc = 0.5 * (adv_loss(logits_fake.float(), target_is_real=False, for_discriminator=True)
                                  + adv_loss(logits_real.float(), target_is_real=True, for_discriminator=True))
                    disc.backward()
                    optimizer_d.step()

                sums["loss"] += loss_g.item()
                sums["recons_loss"] += recons.item()
                sums["gen_loss"] += gen.item()
                sums["disc_loss"] += disc.item()
                n_steps += 1

            for k in history:
                history[k].append(sums[k] / max(n_steps, 1))
            print(f"Local epoch {local_epoch + 1}/{epochs} (global {global_epoch}, "
                  f"adversarial={'on' if adversarial else 'off'}) - "
                  + " - ".join(f"{k}: {history[k][-1]:.4f}" for k in history), flush=True)

        # Persist the local discriminator for the next round.
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        torch.save({
            "discriminator": {k: v.detach().cpu() for k, v in discriminator.state_dict().items()},
            "optimizer_d": optimizer_d.state_dict(),
            "epochs_done": epochs_done + epochs,
        }, state_path)

        del discriminator, optimizer_d, perceptual
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if val_dataset is not None:
            val_metrics = self.evaluate(val_dataset)
            for k, v in val_metrics.items():
                history[f"val_{k}"] = [float(v)]
        return history

    # ----- evaluation ----------------------------------------------------------
    def evaluate(self, dataset, **kwargs):
        """Reconstruction metrics on up to EVAL_MAX_SAMPLES volumes. Returns
        strings (FLINT protobuf requirement). Images are in [0, 1], so PSNR
        uses a peak value of 1."""
        if dataset is None:
            return {}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.autoencoder.eval()

        data = self._as_torch_dataset(dataset)
        if EVAL_MAX_SAMPLES is not None and len(data) > EVAL_MAX_SAMPLES:
            data = Subset(data, range(EVAL_MAX_SAMPLES))
        loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=DATALOADER_WORKERS)

        mses, maes, wl1s, psnrs = [], [], [], []
        with torch.no_grad():
            for batch in loader:
                images = batch["t1_image"].to(device).float()
                with self._autocast(device):
                    reconstruction, _, _ = self.autoencoder(images)
                reconstruction = reconstruction.float()
                mse = torch.mean((images - reconstruction) ** 2).item()
                mses.append(mse)
                maes.append(torch.mean(torch.abs(images - reconstruction)).item())
                wl1s.append(weighted_l1_loss(reconstruction, images).item())
                psnrs.append(10.0 * np.log10(1.0 / max(mse, 1e-12)))

        if not mses:
            return {}
        return {
            "loss": str(float(np.mean(mses))),
            "mse": str(float(np.mean(mses))),
            "mae": str(float(np.mean(maes))),
            "weighted_l1": str(float(np.mean(wl1s))),
            "psnr": str(float(np.mean(psnrs))),
        }
