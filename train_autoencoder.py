#!/usr/bin/env python
"""
Stage 1: Train the AutoencoderKL (+ PatchDiscriminator) for the 3D LDM (ADNI).

Converted from 3d_ldm_fedNeuro-ADNI.ipynb so it can run as a SLURM batch job
on CARC (see train_autoencoder.slurm).

Usage:
    python train_autoencoder.py [--train-csv PATH] [--val-csv PATH] [--epochs N] ...
"""

import argparse
import contextlib
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from monai.utils import set_determinism
from torch.nn import L1Loss

from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator

from dataset import MonaiMRIDatasetADNI
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser(description="Train autoencoder (stage 1 of 3D LDM)")
    p.add_argument("--train-csv", default="/project2/jambitem_1194/neuroimaging-data/FEDAD/ADNI_PATH/ADNI_ADvsCN_train_fedpath.csv")
    p.add_argument("--output-size", type=int, nargs=3, default=(96, 112, 96))
    p.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="per-GPU batch size; with torchrun the effective global batch size is this times the GPU count",
    )
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--warmup-epochs", type=int, default=5, help="epochs before adversarial loss kicks in")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--disc-lr", type=float, default=None, help="discriminator learning rate; defaults to --lr")
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out-dir", default="/data/ashaji/FedNeuro/abhi")
    p.add_argument("--ckpt-name", default="autoencoder_exp1_adni.pth")
    p.add_argument("--wandb-project", default="fedneuro")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-group", default="autoencoder-ldm-a100")
    p.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
        help="online streams live to wandb.ai; use offline if the compute node has no internet (needs `wandb sync` after)",
    )
    p.add_argument("--log-every", type=int, default=10, help="print a status line every N training steps")
    return p.parse_args()


def weighted_l1_loss(pred, target, thr=0.05, fg_weight=10.0, bg_weight=1.0):
    weights = torch.where(
        target > thr,
        torch.tensor(fg_weight, device=target.device, dtype=target.dtype),
        torch.tensor(bg_weight, device=target.device, dtype=target.dtype),
    )
    return (weights * torch.abs(pred - target)).mean()


def KL_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
    return torch.sum(kl_loss) / kl_loss.shape[0]


def unwrap(model):
    """Return the underlying module, stripping DistributedDataParallel's wrapper if present.

    Needed because a DDP-wrapped model's .state_dict() keys are prefixed with
    "module.", which would silently fail to load into the plain AutoencoderKL
    that train_diffusion.py instantiates downstream.
    """
    return model.module if isinstance(model, DistributedDataParallel) else model


def main():
    args = parse_args()

    # torchrun sets LOCAL_RANK (and RANK/WORLD_SIZE) for every worker process; its absence
    # means this is a plain single-process run, so everything below degenerates to rank 0
    # of a world size of 1 and behaves exactly as before.
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
            # keep logs readable: only rank 0's prints reach the job's stdout
            sys.stdout = sys.stderr = open(os.devnull, "w")
    else:
        local_rank, rank, world_size = 0, 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)

    set_determinism(args.seed)
    print(os.getpid())
    print(f"Using {device}  (distributed={distributed}, rank={rank}/{world_size})")

    if rank == 0:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            group=args.wandb_group,
            mode=args.wandb_mode,
            config=vars(args),
        )
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

    train_dataset = MonaiMRIDatasetADNI(args.train_csv, output_size=tuple(args.output_size), apply_padding=True)

    # DistributedSampler shards the dataset across ranks so each GPU trains on a disjoint slice per epoch.
    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=args.seed) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
    )

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

    discriminator = PatchDiscriminator(
        spatial_dims=3,
        num_layers_d=4,
        num_channels=64,
        in_channels=1,
        out_channels=1,
    ).to(device)

    if distributed:
        # find_unused_parameters=True tolerates the discriminator being called twice per
        # step (once for the generator's adversarial term, once for its own update) before
        # a backward() is issued for either call. device_ids is only meaningful for CUDA
        # modules -- DDP rejects it outright for CPU modules (e.g. a CPU-only smoke test).
        ddp_device_ids = [local_rank] if torch.cuda.is_available() else None
        autoencoder = DistributedDataParallel(autoencoder, device_ids=ddp_device_ids, find_unused_parameters=True)
        discriminator = DistributedDataParallel(discriminator, device_ids=ddp_device_ids, find_unused_parameters=True)

    l1_loss = L1Loss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
    loss_perceptual.to(device)

    adv_weight = 0.01
    perceptual_weight = 0.01
    kl_weight = 1e-6

    optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=args.lr)
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=args.disc_lr or args.lr)

    epoch_recon_loss_list = []
    epoch_gen_loss_list = []
    epoch_disc_loss_list = []

    for epoch in range(args.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)  # vary the shuffle per epoch, consistently across ranks
        autoencoder.train()
        discriminator.train()
        epoch_loss = 0
        gen_epoch_loss = 0
        disc_epoch_loss = 0
        n_steps = len(train_loader)
        for step, batch in enumerate(train_loader):
            images = batch["t1_image"].to(device)

            optimizer_g.zero_grad(set_to_none=True)
            reconstruction, z_mu, z_sigma = autoencoder(images)
            kl_loss = KL_loss(z_mu, z_sigma)

            recons_loss = weighted_l1_loss(reconstruction.float(), images.float())
            p_loss = loss_perceptual(reconstruction.float(), images.float())
            loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss

            if epoch > args.warmup_epochs:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g = loss_g + adv_weight * generator_loss
                backward_ctx = discriminator.no_sync() if distributed else contextlib.nullcontext()
            else:
                backward_ctx = contextlib.nullcontext()

            with backward_ctx:
                loss_g.backward()
            optimizer_g.step()

            if epoch > args.warmup_epochs:
                optimizer_d.zero_grad(set_to_none=True)
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                discriminator_loss.backward()
                optimizer_d.step()

            epoch_loss += recons_loss.item()
            if epoch > args.warmup_epochs:
                gen_epoch_loss += generator_loss.item()
                disc_epoch_loss += discriminator_loss.item()

            if rank == 0 and (step % args.log_every == 0 or step == n_steps - 1):
                print(
                    f"Epoch {epoch} [{step + 1}/{n_steps}] "
                    f"recons_loss={epoch_loss / (step + 1):.4f} "
                    f"gen_loss={gen_epoch_loss / (step + 1):.4f} "
                    f"disc_loss={disc_epoch_loss / (step + 1):.4f}"
                )

        avg_recon_loss = epoch_loss / n_steps
        avg_gen_loss = gen_epoch_loss / n_steps
        avg_disc_loss = disc_epoch_loss / n_steps
        epoch_recon_loss_list.append(avg_recon_loss)
        epoch_gen_loss_list.append(avg_gen_loss)
        epoch_disc_loss_list.append(avg_disc_loss)
        if rank == 0:
            wandb.log(
                {
                    "epoch": epoch,
                    "epoch_recons_loss": avg_recon_loss,
                    "gen_loss": avg_gen_loss,
                    "disc_loss": avg_disc_loss,
                }
            )

            if epoch % 80 == 0:
                with torch.no_grad():
                    mid_slice = reconstruction.shape[2] // 2
                    recon_img = reconstruction[0, 0, mid_slice].detach().cpu().numpy()
                    orig_img = images[0, 0, mid_slice].detach().cpu().numpy()
                    wandb.log(
                        {
                            "epoch": epoch,
                            "autoencoder/original": wandb.Image(orig_img),
                            "autoencoder/reconstruction": wandb.Image(recon_img),
                        }
                    )

            if epoch % 100 == 0:
                state_ckpt_path = os.path.join(args.out_dir, f"autoencoder_state_epoch_{epoch}.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "autoencoder_state_dict": unwrap(autoencoder).state_dict(),
                        "discriminator_state_dict": unwrap(discriminator).state_dict(),
                        "optimizer_g_state_dict": optimizer_g.state_dict(),
                        "optimizer_d_state_dict": optimizer_d.state_dict(),
                        "recon_loss": avg_recon_loss,
                        "gen_loss": avg_gen_loss,
                        "disc_loss": avg_disc_loss,
                    },
                    state_ckpt_path,
                )
                print(f"Saved full training state at epoch {epoch} -> {state_ckpt_path}")

    del loss_perceptual
    torch.cuda.empty_cache()

    if rank == 0:
        ckpt_path = os.path.join(args.out_dir, args.ckpt_name)
        torch.save(unwrap(autoencoder).state_dict(), ckpt_path)
        print(f"Saved autoencoder checkpoint to {ckpt_path}")

        plt.style.use("ggplot")
        plt.figure()
        plt.title("Learning Curves", fontsize=20)
        plt.plot(epoch_recon_loss_list)
        plt.xlabel("Epochs", fontsize=16)
        plt.ylabel("Loss", fontsize=16)
        plt.savefig(os.path.join(args.out_dir, "autoencoder_recon_loss.png"))

        plt.figure()
        plt.title("Adversarial Training Curves", fontsize=20)
        plt.plot(epoch_gen_loss_list, color="C0", linewidth=2.0, label="Generator")
        plt.plot(epoch_disc_loss_list, color="C1", linewidth=2.0, label="Discriminator")
        plt.xlabel("Epochs", fontsize=16)
        plt.ylabel("Loss", fontsize=16)
        plt.legend(prop={"size": 14})
        plt.savefig(os.path.join(args.out_dir, "autoencoder_adv_loss.png"))

        wandb.finish()

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
