#!/usr/bin/env python
"""
Stage 1: Train the AutoencoderKL (+ PatchDiscriminator) for the 3D LDM (ADNI).

Converted from 3d_ldm_fedNeuro-ADNI.ipynb so it can run as a SLURM batch job
on CARC (see train_autoencoder.slurm).

Usage:
    python train_autoencoder.py [--train-csv PATH] [--val-csv PATH] [--epochs N] ...
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import wandb
from monai.utils import first, set_determinism
from torch.nn import L1Loss

from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator

from dataset import MonaiMRIDatasetADNI
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser(description="Train autoencoder (stage 1 of 3D LDM)")
    p.add_argument("--train-csv", default="/project2/jambitem_1194/neuroimaging-data/FEDAD/ADNI_PATH/ADNI_ADvsCN_train_fedpath.csv")
    p.add_argument("--val-csv", default="/project2/jambitem_1194/neuroimaging-data/FEDAD/ADNI_PATH/ADNI_ADvsCN_test_fedpath.csv")
    p.add_argument("--output-size", type=int, nargs=3, default=(96, 112, 96))
    p.add_argument("--batch-size", type=int, default=2)
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


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    set_determinism(args.seed)
    print(os.getpid())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

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
    val_dataset = MonaiMRIDatasetADNI(args.val_csv, output_size=tuple(args.output_size), apply_padding=True)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

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
                loss_g += adv_weight * generator_loss

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

            if step % args.log_every == 0 or step == n_steps - 1:
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
                    "autoencoder_state_dict": autoencoder.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "optimizer_g_state_dict": optimizer_g.state_dict(),
                    "optimizer_d_state_dict": optimizer_d.state_dict(),
                    "recon_loss": avg_recon_loss,
                    "gen_loss": avg_gen_loss,
                    "disc_loss": avg_disc_loss,
                },
                state_ckpt_path,
            )
            print(f"Saved full training state at epoch {epoch} -> {state_ckpt_path}")

    del discriminator
    del loss_perceptual
    torch.cuda.empty_cache()

    ckpt_path = os.path.join(args.out_dir, args.ckpt_name)
    torch.save(autoencoder.state_dict(), ckpt_path)
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

    autoencoder.eval()
    check_data = first(val_loader)
    images = check_data["t1_image"].to(device)
    with torch.no_grad():
        reconstruction, _, _ = autoencoder(images)
    mse = torch.mean((images - reconstruction) ** 2).item()
    mae = torch.mean(torch.abs(images - reconstruction)).item()
    print(f"Val batch MSE: {mse:.6f}  MAE: {mae:.6f}")
    wandb.log({"val_mse": mse, "val_mae": mae})
    wandb.finish()


if __name__ == "__main__":
    main()
