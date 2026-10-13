#!/usr/bin/env python
"""
Stage 2: Train the DiffusionModelUNet (LDM) on top of a frozen, pre-trained
AutoencoderKL (ADNI).

Converted from 3d_ldm_fedNeuro-ADNI.ipynb so it can run as a SLURM batch job
on CARC (see train_diffusion.slurm). Run train_autoencoder.py first and pass
its checkpoint via --autoencoder-ckpt.

Usage:
    python train_diffusion.py --autoencoder-ckpt PATH [--epochs N] ...
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # headless plotting for batch jobs
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from monai.utils import set_determinism
from torch.cuda.amp import GradScaler, autocast

from generative.inferers import LatentDiffusionInferer
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

from dataset import NULL_LABEL, MonaiMRIDataset
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser(description="Train diffusion UNet (stage 2 of 3D LDM)")
    p.add_argument("--train-csv", default="/data/ashaji/FedNeuro/modified_csvs/ADNI_ADvsCN_new_train.csv")
    p.add_argument("--val-csv", default="/data/ashaji/FedNeuro/modified_csvs/ADNI_ADvsCN_new_test.csv")
    p.add_argument("--output-size", type=int, nargs=3, default=(96, 112, 96))
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--autoencoder-ckpt",
        default="/data/ashaji/FedNeuro/autoencoder_exp1_adni.pth",
        help="path to a checkpoint produced by train_autoencoder.py",
    )
    p.add_argument("--out-dir", default="/data/ashaji/FedNeuro")
    p.add_argument("--ckpt-name", default="ldm_checkpoint_modified_final.pt")
    p.add_argument(
        "--scale-factor",
        type=float,
        required=True,
        help="latent scale factor = 1 / std of the autoencoder's SAMPLED latents (encode_stage_2_inputs), "
        "precomputed over the training scans with the same frozen autoencoder",
    )
    p.add_argument("--cross-attention-dim", type=int, default=64, help="size of the diagnosis embedding fed to the UNet cross-attention")
    p.add_argument(
        "--cond-drop-prob",
        type=float,
        default=0.15,
        help="probability of replacing the diagnosis with the null label during training (classifier-free guidance)",
    )
    p.add_argument(
        "--init-ckpt",
        default=None,
        help="optional unconditional UNet checkpoint to warm-start from (conditioning layers stay newly initialised)",
    )
    p.add_argument("--log-every", type=int, default=10, help="print a status line every N training steps")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    set_determinism(args.seed)
    print(os.getpid())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_dataset = MonaiMRIDataset(args.train_csv, output_size=tuple(args.output_size), apply_padding=True)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    # ── Frozen autoencoder ────────────────────────────────────────────────
    autoencoder = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=(64, 128, 256),
        latent_channels=8,
        num_res_blocks=2,
        norm_num_groups=16,
        attention_levels=(False, False, True),
    ).to(device)
    autoencoder.load_state_dict(torch.load(args.autoencoder_ckpt, map_location=device))
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad_(False)

    # ── Diffusion UNet + scheduler ───────────────────────────────────────
    unet = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=8,
        out_channels=8,
        num_channels=(128, 256, 512),
        num_res_blocks=2,
        attention_levels=(False, True, True),
        num_head_channels=(0, 32, 64),
        with_conditioning=True,
        cross_attention_dim=args.cross_attention_dim,
    ).to(device)
    # 0 = CN, 1 = AD, 2 = null label (classifier-free guidance)
    label_embed = torch.nn.Embedding(NULL_LABEL + 1, args.cross_attention_dim).to(device)

    if args.init_ckpt:
        state = torch.load(args.init_ckpt, map_location=device)["unet_state_dict"]
        missing, unexpected = unet.load_state_dict(state, strict=False)
        print(f"Warm-started UNet from {args.init_ckpt}: {len(missing)} new params, {len(unexpected)} unexpected")

    scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195)

    # ── Scaling factor ────────────────────────────────────────────────────
    # Precomputed over the training scans (not estimated from one batch), so the latents fed to the
    # diffusion model have std ~1, matching the noise. Must match the frozen autoencoder above.
    scale_factor = args.scale_factor

    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)
    optimizer_diff = torch.optim.Adam(params=list(unet.parameters()) + list(label_embed.parameters()), lr=args.lr)

    # ── Training loop ─────────────────────────────────────────────────────
    epoch_loss_list = []
    scaler = GradScaler()
    best_loss = float("inf")

    for epoch in range(args.epochs):
        unet.train()
        label_embed.train()
        epoch_loss = 0
        n_steps = len(train_loader)
        for step, batch in enumerate(train_loader):
            images = batch["t1_image"].to(device)
            labels = batch["label"].to(device)
            drop = torch.rand(labels.shape[0], device=device) < args.cond_drop_prob
            labels = torch.where(drop, torch.full_like(labels, NULL_LABEL), labels)
            optimizer_diff.zero_grad(set_to_none=True)

            with autocast(enabled=True):
                z = autoencoder.encode_stage_2_inputs(images)
                noise = torch.randn_like(z).to(device)

                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                ).long()

                condition = label_embed(labels).unsqueeze(1)  # (B, 1, cross_attention_dim)
                noise_pred = inferer(
                    inputs=images,
                    autoencoder_model=autoencoder,
                    diffusion_model=unet,
                    noise=noise,
                    timesteps=timesteps,
                    condition=condition,
                )

                loss = F.mse_loss(noise_pred.float(), noise.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer_diff)
            scaler.update()

            epoch_loss += loss.item()
            if step % args.log_every == 0 or step == n_steps - 1:
                print(f"Epoch {epoch} [{step + 1}/{n_steps}] loss={epoch_loss / (step + 1):.6f}")

        avg_loss = epoch_loss / n_steps
        epoch_loss_list.append(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                "epoch": epoch,
                "unet_state_dict": unet.state_dict(),
                "label_embed_state_dict": label_embed.state_dict(),
                "cross_attention_dim": args.cross_attention_dim,
                "scale_factor": scale_factor,
                "optimizer_state_dict": optimizer_diff.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "loss": avg_loss,
            }
            torch.save(checkpoint, os.path.join(args.out_dir, f"ldm_checkpoint_modified_epoch_{epoch}.pt"))
            print(f"Saved new best model at epoch {epoch} (loss: {avg_loss:.6f})")

    final_checkpoint = {
        "unet_state_dict": unet.state_dict(),
        "label_embed_state_dict": label_embed.state_dict(),
        "cross_attention_dim": args.cross_attention_dim,
        "scale_factor": scale_factor,
        "optimizer_state_dict": optimizer_diff.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }
    ckpt_path = os.path.join(args.out_dir, args.ckpt_name)
    torch.save(final_checkpoint, ckpt_path)
    print(f"Saved final diffusion checkpoint to {ckpt_path}")

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(epoch_loss_list)), epoch_loss_list, label="Train loss")
    plt.title("Learning Curves", fontsize=20)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend()
    plt.savefig(os.path.join(args.out_dir, "diffusion_train_loss.png"))


if __name__ == "__main__":
    main()
