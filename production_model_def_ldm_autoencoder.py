"""
Federated Stage 1 of the 3D LDM: AutoencoderKL trained with L1 + KL + perceptual
+ adversarial losses, in the Metis PyTorchDef format, with each federation node
training on ALL of its GPUs (DistributedDataParallel).

How the two levels fit together:
- Federation level (unchanged Metis): one learner per node. Each round Metis calls
  fit() once and gets back one set of autoencoder weights, which the controller
  aggregates. Only the autoencoder is in state_dict(), so only it is aggregated.
- Node level (inside fit()): fit() writes the current weights to disk and launches
  `torchrun` with one process per GPU. Each process runs _ddp_worker_main(), which
  follows Abhijith's DDP train_autoencoder.py: DistributedSampler, DDP-wrapped
  autoencoder AND discriminator, find_unused_parameters=True, discriminator
  no_sync() during the generator backward. Rank 0 writes the updated weights back
  and fit() loads them. Running DDP as a separate program is needed because Metis
  trains inside a worker process that may not be allowed to start child
  processes itself.

Precision: full precision (fp32) by default, matching the centralized DDP
script, as agreed with Abhijith for the first runs. Batch size 2 per GPU is
close to the L40S's 46 GB in fp32, so check the "Peak GPU memory" line in the
first run. If a node runs out of memory, set USE_AMP = True (bfloat16 mixed
precision measured ~40.5 GB at the same batch size) or use BatchSize: 1.

Differences from the centralized DDP script:
- The discriminator stays LOCAL to each node and is never aggregated (confirmed
  with Abhijith). It is saved to the node's disk after each round and restored
  at the start of the next, along with a count of local epochs done, which
  drives the adversarial warmup.
- The generator optimizer is recreated every round (standard FedAvg practice).

The yaml's LocalModelConfig.BatchSize is the batch size PER GPU (as in the DDP
script), so a 2-GPU node takes 2 x BatchSize volumes per step.
"""

import contextlib
import importlib.machinery
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile

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
WARMUP_EPOCHS = 5          # counted in LOCAL epochs completed on this node, across rounds
DEFAULT_LR = 1e-4          # used only if the yaml's optimizer config has no learning rate
DISC_LR = None             # None = same as the generator learning rate
SEED = 12                  # DistributedSampler seed, as in the DDP script
USE_AMP = False            # False = full precision (fp32), as in the DDP script.
                           # True = bfloat16 mixed precision, if fp32 runs out of GPU memory.
NUM_GPUS = None            # GPUs per node for training; None = all GPUs visible to the learner
DDP_DATALOADER_WORKERS = 4 # data-loading workers per GPU process (as in the DDP script)
EVAL_DATALOADER_WORKERS = 0  # evaluation runs inside Metis's worker process: keep 0
# Evaluation runs on every requested split (train/val/test) every round. Cap the
# number of volumes per evaluation to keep rounds fast; None = evaluate all.
EVAL_MAX_SAMPLES = 64
# Per-node local state (discriminator) between rounds, and scratch space for the
# DDP hand-off. The hostname keeps nodes separate if they ever share a filesystem.
LOCAL_STATE_DIR = "/tmp/metis/ldm_local_state"
WORK_DIR_ROOT = "/tmp/metis/ldm_ddp_work"


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


def _autocast(device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                          enabled=USE_AMP and device.type == "cuda")


def _local_state_path():
    return os.path.join(LOCAL_STATE_DIR, socket.gethostname(), "local_state.pt")


# ---------------------------------------------------------------------------
# One round of local training, run by every GPU process (or in-process on 1 GPU)
# ---------------------------------------------------------------------------
def _ddp_worker_main(cfg):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data.distributed import DistributedSampler

    # torchrun sets LOCAL_RANK/RANK/WORLD_SIZE; without it this is a 1-process run.
    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", init_method="env://")
        rank, world_size = dist.get_rank(), dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        local_rank, rank, world_size = 0, 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    log(f"[DDP] rank 0 of {world_size} process(es) on {device}, "
        f"precision: {'bfloat16 mixed' if USE_AMP else 'full (fp32)'}, "
        f"batch size {cfg['batch_size']} per GPU")

    # Models: the incoming community autoencoder, and this node's local discriminator.
    autoencoder = AutoencoderKL(**AE_KWARGS)
    autoencoder.load_state_dict(torch.load(cfg["ae_in"], map_location="cpu"))
    autoencoder.to(device)
    discriminator = PatchDiscriminator(**DISC_KWARGS).to(device)

    state, epochs_done = None, 0
    if os.path.exists(cfg["state_path"]):
        state = torch.load(cfg["state_path"], map_location="cpu")
        discriminator.load_state_dict(state["discriminator"])
        epochs_done = int(state.get("epochs_done", 0))
        log(f"Restored local discriminator state ({epochs_done} local epochs done)")

    if distributed:
        ae_m = DistributedDataParallel(autoencoder, device_ids=[local_rank], find_unused_parameters=True)
        disc_m = DistributedDataParallel(discriminator, device_ids=[local_rank], find_unused_parameters=True, broadcast_buffers=False)
    else:
        ae_m, disc_m = autoencoder, discriminator

    optimizer_g = torch.optim.Adam(ae_m.parameters(), lr=cfg["lr"], betas=cfg["betas"], eps=cfg["eps"])
    optimizer_d = torch.optim.Adam(disc_m.parameters(), lr=cfg["disc_lr"])
    if state is not None and "optimizer_d" in state:
        optimizer_d.load_state_dict(state["optimizer_d"])

    dataset = cfg["dataset"]
    sampler = DistributedSampler(dataset, shuffle=True, seed=cfg["seed"]) if distributed else None
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=(sampler is None),
                        sampler=sampler, num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda"))

    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze",
                                is_fake_3d=True, fake_3d_ratio=0.2).to(device)
    perceptual.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history = {"loss": [], "recons_loss": [], "gen_loss": [], "disc_loss": []}
    for local_epoch in range(cfg["epochs"]):
        global_epoch = epochs_done + local_epoch
        adversarial = global_epoch > WARMUP_EPOCHS   # same gate as the centralized script
        if sampler is not None:
            sampler.set_epoch(global_epoch)          # vary the shuffle per epoch, consistently across ranks
        ae_m.train()
        disc_m.train()
        # [loss, recons, gen, disc, steps], summed over this rank's steps
        sums = torch.zeros(5, device=device)

        for batch in loader:
            images = batch["t1_image"].to(device, non_blocking=True)

            optimizer_g.zero_grad(set_to_none=True)
            with _autocast(device):
                reconstruction, z_mu, z_sigma = ae_m(images)
            reconstruction = reconstruction.float()
            recons = weighted_l1_loss(reconstruction, images.float())
            p_loss = perceptual(reconstruction, images.float())
            loss_g = recons + KL_WEIGHT * kl_loss(z_mu.float(), z_sigma.float()) + PERCEPTUAL_WEIGHT * p_loss

            gen = torch.zeros((), device=device)
            backward_ctx = contextlib.nullcontext()
            if adversarial:
                with _autocast(device):
                    logits_fake = disc_m(reconstruction.contiguous())[-1]
                gen = adv_loss(logits_fake.float(), target_is_real=True, for_discriminator=False)
                loss_g = loss_g + ADV_WEIGHT * gen
                # As in the DDP script: don't all-reduce the discriminator's gradients
                # from the generator's backward (they're discarded before its own step).
                if distributed:
                    backward_ctx = disc_m.no_sync()

            with backward_ctx:
                loss_g.backward()
            optimizer_g.step()

            disc = torch.zeros((), device=device)
            if adversarial:
                optimizer_d.zero_grad(set_to_none=True)
                with _autocast(device):
                    logits_fake = disc_m(reconstruction.detach().contiguous())[-1]
                    logits_real = disc_m(images.contiguous())[-1]
                disc = 0.5 * (adv_loss(logits_fake.float(), target_is_real=False, for_discriminator=True)
                              + adv_loss(logits_real.float(), target_is_real=True, for_discriminator=True))
                disc.backward()
                optimizer_d.step()

            sums += torch.stack([loss_g.detach(), recons.detach(), gen.detach(), disc.detach(),
                                 torch.ones((), device=device)])

        # Average over every step on every GPU of this node
        if distributed:
            dist.all_reduce(sums)
        steps = max(sums[4].item(), 1.0)
        for i, k in enumerate(history):
            history[k].append(sums[i].item() / steps)
        log(f"Local epoch {local_epoch + 1}/{cfg['epochs']} (global {global_epoch}, "
            f"adversarial={'on' if adversarial else 'off'}, {int(sums[4].item())} steps over {world_size} GPU(s)) - "
            + " - ".join(f"{k}: {history[k][-1]:.4f}" for k in history))

    if device.type == "cuda":
        log(f"Peak GPU memory (rank 0): {torch.cuda.max_memory_allocated(device) / 1e9:.1f} GB")

    if rank == 0:
        torch.save({k: v.detach().cpu() for k, v in autoencoder.state_dict().items()}, cfg["ae_out"])
        os.makedirs(os.path.dirname(cfg["state_path"]), exist_ok=True)
        torch.save({
            "discriminator": {k: v.detach().cpu() for k, v in discriminator.state_dict().items()},
            "optimizer_d": optimizer_d.state_dict(),
            "epochs_done": epochs_done + cfg["epochs"],
        }, cfg["state_path"])
        with open(cfg["history_out"], "w") as f:
            json.dump(history, f)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def _register_dynamic_modules_by_value(*objs):
    """Make cloudpickle embed the code of any class/function that comes from a
    module the torchrun child processes can't import (e.g. Metis's
    'custom_model_module' / 'custom_data_module', loaded from a file path).
    Otherwise cloudpickle stores only a reference to that module, and the
    children fail with ModuleNotFoundError. Real packages (torch, numpy, ...)
    are importable and keep being pickled by reference."""
    import cloudpickle
    names = set()
    for obj in objs:
        candidates = [obj] if isinstance(obj, type) or callable(obj) else []
        candidates += list(type(obj).__mro__)
        inner = getattr(obj, "dataset", None)          # e.g. torch Subset(dataset, ...)
        if inner is not None:
            candidates += list(type(inner).__mro__)
        for c in candidates:
            name = getattr(c, "__module__", None)
            if name and name not in ("builtins", "__main__"):
                names.add(name)
    for name in names:
        module = sys.modules.get(name)
        if module is None:
            continue
        # Ask what a FRESH process would see: search the import path only.
        # importlib.util.find_spec() would also consult sys.modules and wrongly
        # report modules loaded from a file path (spec_from_file_location) as
        # importable.
        top_level = name.split(".")[0]
        try:
            importable = importlib.machinery.PathFinder.find_spec(top_level) is not None
        except (ValueError, ImportError):
            importable = False
        if not importable:
            cloudpickle.register_pickle_by_value(module)


# Tiny launcher that torchrun runs on every GPU: unpickle the job and run it.
_WORKER_STUB = """import sys, cloudpickle
fn, cfg = cloudpickle.load(open(sys.argv[1], "rb"))
fn(cfg)
"""


class ProductionModel(nn.Module, PyTorchDef):

    def __init__(self, batch_size=2, optimizer_config=None, **kwargs):
        super().__init__()
        self.batch_size = batch_size          # per GPU
        self.optimizer_config = optimizer_config
        # The ONLY registered submodule: this is what Metis aggregates.
        self.autoencoder = AutoencoderKL(**AE_KWARGS)

    # ----- Metis interface ---------------------------------------------------
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
    def _as_torch_dataset(dataset):
        return dataset.x if hasattr(dataset, "x") else dataset

    # ----- training ------------------------------------------------------------
    def fit(self, dataset, val_dataset=None, epochs=1, **kwargs):
        # Free the GPUs in this (Metis worker) process for the DDP processes.
        self.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        n_gpus = torch.cuda.device_count() if NUM_GPUS is None else NUM_GPUS
        lr, betas, eps = self._optimizer_kwargs()
        os.makedirs(WORK_DIR_ROOT, exist_ok=True)
        work_dir = tempfile.mkdtemp(prefix="round_", dir=WORK_DIR_ROOT)
        try:
            cfg = dict(
                dataset=self._as_torch_dataset(dataset),
                epochs=int(epochs),
                batch_size=int(self.batch_size),
                lr=lr, betas=betas, eps=eps,
                disc_lr=DISC_LR or lr,
                seed=SEED,
                # DataLoader workers are fine inside torchrun's processes, but not
                # inside Metis's worker process (the 1-GPU in-process path).
                num_workers=DDP_DATALOADER_WORKERS if n_gpus > 1 else 0,
                ae_in=os.path.join(work_dir, "ae_in.pt"),
                ae_out=os.path.join(work_dir, "ae_out.pt"),
                history_out=os.path.join(work_dir, "history.json"),
                state_path=_local_state_path(),
            )
            torch.save({k: v.detach().cpu() for k, v in self.autoencoder.state_dict().items()}, cfg["ae_in"])

            if n_gpus > 1:
                import cloudpickle
                # Embed the code of the worker function and the dataset's classes
                # when the child processes can't import their modules.
                _register_dynamic_modules_by_value(_ddp_worker_main, cfg["dataset"])
                job_path = os.path.join(work_dir, "job.pkl")
                with open(job_path, "wb") as f:
                    cloudpickle.dump((_ddp_worker_main, cfg), f)
                stub_path = os.path.join(work_dir, "ddp_worker.py")
                with open(stub_path, "w") as f:
                    f.write(_WORKER_STUB)

                env = os.environ.copy()
                # Same Python environment (Bazel runfiles) for the child processes.
                env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
                cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                       f"--nproc_per_node={n_gpus}", stub_path, job_path]
                print(f"Launching DDP training on {n_gpus} GPUs: {' '.join(cmd)}", flush=True)
                result = subprocess.run(cmd, env=env)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"DDP training failed (exit code {result.returncode}); see the worker "
                        f"traceback above. If it says 'CUDA out of memory', set USE_AMP = True "
                        f"in this file or use LocalModelConfig BatchSize: 1 in the yaml. If NCCL "
                        f"errors mention shared memory, start the container with --ipc=host.")
            else:
                print("One GPU (or none) visible: training in-process without DDP", flush=True)
                _ddp_worker_main(cfg)

            self.autoencoder.load_state_dict(torch.load(cfg["ae_out"], map_location="cpu"))
            with open(cfg["history_out"]) as f:
                history = json.load(f)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

        if val_dataset is not None:
            val_metrics = self.evaluate(val_dataset)
            for k, v in val_metrics.items():
                history[f"val_{k}"] = [float(v)]
        return history

    # ----- evaluation ----------------------------------------------------------
    def evaluate(self, dataset, **kwargs):
        """Reconstruction metrics on up to EVAL_MAX_SAMPLES volumes, on one GPU.
        Returns strings (Metis protobuf requirement). Images are in [0, 1], so
        PSNR uses a peak value of 1."""
        if dataset is None:
            return {}
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Community-model evaluation can overlap with the next DDP training round.
        # Keep it off the GPUs so it cannot consume DDP training memory.
        device = torch.device("cpu")
        self.to(device)
        self.autoencoder.eval()

        data = self._as_torch_dataset(dataset)
        if EVAL_MAX_SAMPLES is not None and len(data) > EVAL_MAX_SAMPLES:
            data = Subset(data, range(EVAL_MAX_SAMPLES))
        loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=EVAL_DATALOADER_WORKERS)

        mses, maes, wl1s, psnrs = [], [], [], []
        with torch.no_grad():
            for batch in loader:
                images = batch["t1_image"].to(device).float()
                with _autocast(device):
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