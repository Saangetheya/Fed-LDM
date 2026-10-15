#!/usr/bin/env python
"""
Stage 2: Train the DiffusionModelUNet (LDM) on top of a frozen, pre-trained AutoencoderKL,
conditioned on the diagnosis (0 = CN, 1 = AD) with classifier-free guidance.

The UNet learns to predict the noise added to the SAMPLED latents of the frozen autoencoder
(encode_stage_2_inputs), multiplied by the precomputed --scale-factor (compute_scale_factor.py).
The diagnosis is replaced by the null label with probability --cond-drop-prob, so the same
network also learns the unconditional model needed for guided sampling.

Single GPU:   python train_diffusion.py --autoencoder-ckpt PATH --scale-factor F ...
Multi-GPU / multi-node (DDP): launch with torchrun (see slurm/train_diffusion_central_multinode.slurm).
Each rank trains on a disjoint shard of the scans per epoch; the UNet and the label embedding
are synchronised every step.

Outputs in --out-dir:
    loss_history.csv                one row per finished epoch (all-rank means)
    latest_state.pt                 rolling full training state, read by --resume
    ldm_state_epoch_{N}.pt          full training state every --save-every epochs (and at the end)
    <--ckpt-name>                   final model (UNet + label embedding + scale factor)
    diffusion_loss.png              learning curves from loss_history.csv
"""

import argparse
import csv
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")  # headless plotting for batch jobs
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from monai.utils import set_determinism
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

from dataset import NULL_LABEL, MonaiMRIDataset
from evaluate_autoencoders import load_autoencoder


def parse_args():
    p = argparse.ArgumentParser(description="Train diffusion UNet (stage 2 of 3D LDM)")
    p.add_argument("--train-csv", required=True, help="training CSV in the id,age,sex,dx,split,path format")
    p.add_argument("--output-size", type=int, nargs=3, default=(96, 112, 96))
    p.add_argument("--batch-size", type=int, default=2, help="volumes per GPU")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--lr", type=float, default=5e-5, help="peak Adam learning rate (reached after --warmup-steps)")
    p.add_argument("--warmup-steps", type=int, default=1000,
                   help="linear learning-rate warmup from ~0 to --lr over this many optimizer steps, then constant")
    p.add_argument("--grad-clip", type=float, default=1.0, help="clip the global gradient norm to this value")
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--num-workers", type=int, default=4, help="data-loading workers per GPU")
    p.add_argument(
        "--autoencoder-ckpt",
        required=True,
        help="frozen autoencoder: full-state .pt (autoencoder_state_epoch_N.pt, community_round_N.pt) or weights .pth",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ckpt-name", default="ldm_final.pt", help="final model file written at the end of training")
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
    p.add_argument("--save-every", type=int, default=200, help="keep a full-state checkpoint every N epochs")
    p.add_argument(
        "--resume",
        action="store_true",
        help="resume from <out-dir>/latest_state.pt if it exists; a no-op (fresh start) if it doesn't",
    )
    p.add_argument("--log-every", type=int, default=50, help="print a status line every N training steps")
    return p.parse_args()


class ConditionalUNet(torch.nn.Module):
    """UNet + diagnosis embedding as one module, so a single DDP wrapper syncs both."""

    def __init__(self, cross_attention_dim):
        super().__init__()
        self.unet = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=8,
            out_channels=8,
            num_channels=(128, 256, 512),
            num_res_blocks=2,
            attention_levels=(False, True, True),
            num_head_channels=(0, 32, 64),
            with_conditioning=True,
            cross_attention_dim=cross_attention_dim,
        )
        # 0 = CN, 1 = AD, 2 = null label (classifier-free guidance)
        self.label_embed = torch.nn.Embedding(NULL_LABEL + 1, cross_attention_dim)

    def forward(self, x, timesteps, labels):
        condition = self.label_embed(labels).unsqueeze(1)  # (B, 1, cross_attention_dim)
        return self.unet(x, timesteps=timesteps, context=condition)


def unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


@torch.no_grad()
def encode(autoencoder, images, scale_factor):
    """Scaled sampled latents, exactly what LatentDiffusionInferer feeds the UNet."""
    return autoencoder.encode_stage_2_inputs(images) * scale_factor


def main():
    args = parse_args()

    # torchrun sets LOCAL_RANK (and RANK/WORLD_SIZE); without it this is a single-process run.
    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        if rank != 0:
            # only rank 0's prints reach the job's stdout; stderr stays open so a crash on
            # any rank still shows its traceback in the job log.
            sys.stdout = open(os.devnull, "w")
    else:
        local_rank, rank, world_size = 0, 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)

    # Different seed per rank: otherwise every GPU would draw the same timesteps, noise and
    # CFG drops at each step. The data shuffle is seeded separately by DistributedSampler.
    set_determinism(args.seed + rank)
    print(f"Using {device}  (distributed={distributed}, world_size={world_size}, precision=fp32)")
    print(f"autoencoder={args.autoencoder_ckpt}  scale_factor={args.scale_factor}  cond_drop_prob={args.cond_drop_prob}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_dataset = MonaiMRIDataset(args.train_csv, output_size=tuple(args.output_size), apply_padding=True)
    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=args.seed) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    print(f"train: {len(train_dataset)} scans, {len(train_loader)} steps/epoch per GPU, "
          f"global batch {args.batch_size * world_size}")

    # ── Frozen autoencoder ────────────────────────────────────────────────
    autoencoder = load_autoencoder(args.autoencoder_ckpt, device)
    for p in autoencoder.parameters():
        p.requires_grad_(False)

    # ── Conditional diffusion UNet + scheduler ───────────────────────────
    model = ConditionalUNet(args.cross_attention_dim).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    # Linear warmup to args.lr over the first warmup_steps optimizer steps, then constant. Created
    # before any resume so that loading the saved optimizer/scheduler states restores the position.
    warmup = max(1, args.warmup_steps)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / warmup))

    # Resume into the plain (pre-DDP) module so the state_dict keys match.
    latest_ckpt_path = os.path.join(args.out_dir, "latest_state.pt")
    start_epoch = 0
    if args.resume and os.path.exists(latest_ckpt_path):
        state = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        if abs(state["scale_factor"] - args.scale_factor) > 1e-9:
            raise SystemExit(f"ERROR: {latest_ckpt_path} was trained with scale_factor {state['scale_factor']}, "
                             f"not {args.scale_factor}")
        model.unet.load_state_dict(state["unet_state_dict"])
        model.label_embed.load_state_dict(state["label_embed_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if "lr_scheduler_state_dict" in state:
            lr_scheduler.load_state_dict(state["lr_scheduler_state_dict"])
        start_epoch = state["epoch"] + 1
        print(f"Resumed from {latest_ckpt_path} at epoch {start_epoch}/{args.epochs}")
    elif args.init_ckpt:
        init = torch.load(args.init_ckpt, map_location=device, weights_only=False)["unet_state_dict"]
        missing, unexpected = model.unet.load_state_dict(init, strict=False)
        print(f"Warm-started UNet from {args.init_ckpt}: {len(missing)} new params, {len(unexpected)} unexpected")

    if start_epoch >= args.epochs:
        print(f"Already completed {args.epochs} epochs -- nothing to do.")
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
        return

    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)

    def full_state(epoch, means):
        m = unwrap(model)
        return {
            "epoch": epoch,
            "unet_state_dict": m.unet.state_dict(),
            "label_embed_state_dict": m.label_embed.state_dict(),
            "cross_attention_dim": args.cross_attention_dim,
            "scale_factor": args.scale_factor,
            "autoencoder_ckpt": os.path.abspath(args.autoencoder_ckpt),
            "cond_drop_prob": args.cond_drop_prob,
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "lr": args.lr,
            "warmup_steps": args.warmup_steps,
            "grad_clip": args.grad_clip,
            "train_loss": means[0],
        }

    # One row per finished epoch, appended across resumed/chained jobs. Losses are means over all
    # steps on ALL ranks. train_loss_cond / _uncond split the samples by whether the diagnosis was
    # kept or dropped to the null label. grad_norm_* is the global gradient norm BEFORE clipping
    # (identical on every rank after DDP's all-reduce); clipped_frac is the share of steps clipped;
    # lr is the learning rate at the end of the epoch (rises during warmup, then constant).
    loss_csv_path = os.path.join(args.out_dir, "loss_history.csv")
    loss_csv_columns = ["epoch", "train_loss", "train_loss_cond", "train_loss_uncond",
                        "grad_norm_mean", "grad_norm_max", "clipped_frac",
                        "lr", "epoch_seconds", "finished_at"]

    num_t = scheduler.num_train_timesteps
    for epoch in range(start_epoch, args.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        # Reseed per epoch so a resumed job does not replay the noise of the epochs it repeats.
        torch.manual_seed(args.seed * 1_000_003 + epoch * 1_009 + rank)
        model.train()
        epoch_start = time.time()
        # per-rank sums: [loss, steps, cond loss, cond samples, uncond loss, uncond samples]
        sums = torch.zeros(6, dtype=torch.float64, device=device)
        grad_norms = []
        n_steps = len(train_loader)
        for step, batch in enumerate(train_loader):
            images = batch["t1_image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            drop = torch.rand(labels.shape[0], device=device) < args.cond_drop_prob
            labels = torch.where(drop, torch.full_like(labels, NULL_LABEL), labels)
            uncond = labels == NULL_LABEL  # dropped, or no diagnosis in the CSV

            z = encode(autoencoder, images, args.scale_factor)
            noise = torch.randn_like(z)
            timesteps = torch.randint(0, num_t, (z.shape[0],), device=device).long()
            noisy = scheduler.add_noise(original_samples=z, noise=noise, timesteps=timesteps)

            optimizer.zero_grad(set_to_none=True)
            noise_pred = model(noisy, timesteps, labels)  # full fp32, no mixed precision
            per_sample = F.mse_loss(noise_pred, noise, reduction="none").mean(dim=(1, 2, 3, 4))
            loss = per_sample.mean()

            loss.backward()
            # Global gradient norm before clipping; gradients are already averaged across ranks.
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            grad_norms.append(grad_norm.item())
            optimizer.step()
            lr_scheduler.step()

            ps = per_sample.detach().double()
            sums += torch.stack([
                loss.detach().double(), torch.ones((), dtype=torch.float64, device=device),
                ps[~uncond].sum(), (~uncond).sum().double(),
                ps[uncond].sum(), uncond.sum().double(),
            ])
            if rank == 0 and (step % args.log_every == 0 or step == n_steps - 1):
                print(f"Epoch {epoch} [{step + 1}/{n_steps}] loss={sums[0].item() / sums[1].item():.6f}", flush=True)

        # Collectives: every rank must reach these, so they stay outside the rank-0 block.
        if distributed:
            dist.all_reduce(sums)
        train_loss = (sums[0] / sums[1].clamp(min=1)).item()
        loss_cond = (sums[2] / sums[3]).item() if sums[3] > 0 else float("nan")
        loss_uncond = (sums[4] / sums[5]).item() if sums[5] > 0 else float("nan")
        epoch_seconds = time.time() - epoch_start
        gn_mean = sum(grad_norms) / max(1, len(grad_norms))
        gn_max = max(grad_norms) if grad_norms else float("nan")
        clipped_frac = sum(g > args.grad_clip for g in grad_norms) / max(1, len(grad_norms))

        if rank == 0:
            state = full_state(epoch, (train_loss,))
            # Rolling checkpoint for --resume, written to a temp file then renamed (atomic), so a
            # crash mid-save leaves the previous one intact.
            tmp_path = latest_ckpt_path + ".tmp"
            torch.save(state, tmp_path)
            os.replace(tmp_path, latest_ckpt_path)

            if epoch % args.save_every == 0 or epoch == args.epochs - 1:
                snap_path = os.path.join(args.out_dir, f"ldm_state_epoch_{epoch}.pt")
                torch.save(state, snap_path)
                print(f"Saved full training state at epoch {epoch} -> {snap_path}")

            # After the checkpoint, so a crash can drop this epoch's row but never duplicate one.
            write_header = not os.path.exists(loss_csv_path) or os.path.getsize(loss_csv_path) == 0
            with open(loss_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(loss_csv_columns)
                writer.writerow([
                    epoch, f"{train_loss:.6f}", f"{loss_cond:.6f}", f"{loss_uncond:.6f}",
                    f"{gn_mean:.4f}", f"{gn_max:.4f}", f"{clipped_frac:.3f}",
                    f"{optimizer.param_groups[0]['lr']:.2e}", f"{epoch_seconds:.1f}",
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                ])
            print(f"Epoch {epoch} done in {epoch_seconds:.0f}s (all-rank means): train={train_loss:.6f} "
                  f"cond={loss_cond:.6f} uncond={loss_uncond:.6f} | grad norm mean {gn_mean:.3f} max {gn_max:.3f} "
                  f"clipped {clipped_frac:.0%} | lr {optimizer.param_groups[0]['lr']:.2e}", flush=True)

    if rank == 0:
        m = unwrap(model)
        final = {
            "epoch": args.epochs - 1,
            "unet_state_dict": m.unet.state_dict(),
            "label_embed_state_dict": m.label_embed.state_dict(),
            "cross_attention_dim": args.cross_attention_dim,
            "scale_factor": args.scale_factor,
            "autoencoder_ckpt": os.path.abspath(args.autoencoder_ckpt),
            "cond_drop_prob": args.cond_drop_prob,
        }
        ckpt_path = os.path.join(args.out_dir, args.ckpt_name)
        torch.save(final, ckpt_path)
        print(f"Saved final diffusion model to {ckpt_path}")

        hist = pd.read_csv(loss_csv_path)
        plt.figure(figsize=(10, 5))
        plt.plot(hist["epoch"], hist["train_loss"], label="Train loss")
        plt.title("Learning Curves", fontsize=20)
        plt.xlabel("Epochs", fontsize=16)
        plt.ylabel("Noise MSE", fontsize=16)
        plt.legend()
        plt.savefig(os.path.join(args.out_dir, "diffusion_loss.png"))

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
