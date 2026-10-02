"""
Metis data module for the 3D LDM (Stage 1 autoencoder): ADNI and NACC sites.

Each federation node trains on its OWN site's CSV (ADNI1, ADNI2GO, ADNI3,
NACC GE, NACC Siemens/Philips), so no partitioning step is needed: run the
learners WITHOUT --prepare_data and point each learner's TrainDatasetPath at
its site's CSV.

CSV format handled (all sites):
  - Image path: the first of IMAGE_PATH_COLUMNS present in the CSV
    (ACCEL_DL_9DOF_2MM_T1 / NONACCEL_DL_9DOF_2MM_T1 / DL_9DOF_2MM_T1 / FED_Path).
    Paths start with /ifs/...; they are rewritten to DATA_ROOT + /ifs/...
  - Diagnosis: DX as 0/1 (also "0.0"/"1.0", or CN/Dementia text), or NACC's
    DX_ADSP 1/3. Rows with a missing or unrecognized diagnosis are dropped.
  - SEX is passed through.

Mount each node's data folder (the one containing ifs/ and the CSV) at
DATA_ROOT inside the container, e.g.
  -v ~/ADNI3:/projectmetis-rc/FEDAD_LDM:ro

Loading and normalization are the same as the colleague's dataset.py:
SimpleITK read, B-spline pad up to 96x112x96, min-max scale to [0, 1].
"""

import os

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

from core.models.model_dataset import ModelDatasetClassification

# ---------------------------------------------------------------------------
# Paths and columns (inside the container)
# ---------------------------------------------------------------------------
DATA_ROOT = "/projectmetis-rc/FEDAD_LDM"   # where each node's data folder is mounted
OUTPUT_SIZE = (96, 112, 96)
# Image-path column, by site; the first one present in a CSV is used.
IMAGE_PATH_COLUMNS = ["FED_Path", "ACCEL_DL_9DOF_2MM_T1", "NONACCEL_DL_9DOF_2MM_T1", "DL_9DOF_2MM_T1"]
# (old_prefix, new_prefix) applied to each image path; first match wins.
PATH_REWRITES = [
    ("/ifs/", DATA_ROOT + "/ifs/"),
]
# Only used by the optional partitioning step (not needed when each node has its own site CSV).
LDM_MASTER_TRAIN_CSV = DATA_ROOT + "/train.csv"

CN_LABEL, AD_LABEL, NULL_LABEL = 0, 1, 2   # NULL_LABEL is used by Stage 2 (classifier-free guidance)
# ASSUMPTION (confirm with Abhijith): numeric DX is 0 = CN, 1 = AD.
_DX_MAP = {"0": CN_LABEL, "1": AD_LABEL, "0.0": CN_LABEL, "1.0": AD_LABEL,
           "CN": CN_LABEL, "Dementia": AD_LABEL}
# NACC DX_ADSP codes (only used if a CSV has DX_ADSP and no DX): 1 = CN, 3 = AD.
_DX_ADSP_MAP = {"1": CN_LABEL, "3": AD_LABEL, "1.0": CN_LABEL, "3.0": AD_LABEL}


def rewrite_path(path):
    path = str(path).strip()
    for old, new in PATH_REWRITES:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def prepare_site_table(csv_path):
    """Read a site CSV and return a DataFrame with standard columns
    image_path / label / sex, dropping rows without a usable diagnosis."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    image_col = next((c for c in IMAGE_PATH_COLUMNS if c in df.columns), None)
    if image_col is None:
        raise ValueError(f"{csv_path}: none of the image columns {IMAGE_PATH_COLUMNS} found; "
                         f"columns are {list(df.columns)}")
    if "DX" in df.columns:
        labels = df["DX"].str.strip().map(_DX_MAP)
    elif "DX_ADSP" in df.columns:
        labels = df["DX_ADSP"].str.strip().map(_DX_ADSP_MAP)
    else:
        raise ValueError(f"{csv_path}: no DX or DX_ADSP column; columns are {list(df.columns)}")
    out = pd.DataFrame({
        "image_path": df[image_col].map(rewrite_path),
        "label": labels,
        "sex": df["SEX"] if "SEX" in df.columns else "",
    })
    n_bad = int(out["label"].isna().sum())
    if n_bad:
        print(f"Warning: {os.path.basename(csv_path)}: dropping {n_bad} rows with missing or "
              f"unrecognized diagnosis ({len(out) - n_bad} rows kept)", flush=True)
        out = out.dropna(subset=["label"])
    out["label"] = out["label"].astype(int)
    return out.reset_index(drop=True)


class Padding(object):
    """Pads (B-spline resample) up to output_size when the volume is smaller on
    any axis; otherwise passes the volume through unchanged. From dataset.py."""

    def __init__(self, output_size):
        if isinstance(output_size, int):
            output_size = (output_size,) * 3
        assert len(output_size) == 3 and all(i > 0 for i in output_size)
        self.output_size = tuple(output_size)

    def __call__(self, sample):
        image = sample["t1_image"]
        size_old = image.GetSize()
        if all(size_old[i] >= self.output_size[i] for i in range(3)):
            return sample
        output_size = tuple(max(size_old[i], self.output_size[i]) for i in range(3))
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(image.GetSpacing())
        resampler.SetSize(output_size)
        resampler.SetInterpolator(sitk.sitkBSpline)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        return {"t1_image": resampler.Execute(image)}


def _load_t1_tensor(path, padding):
    sample = {"t1_image": sitk.ReadImage(path)}
    if padding is not None:
        sample = padding(sample)
    t1_np = sitk.GetArrayFromImage(sample["t1_image"])  # (D, H, W)
    t1_np = (t1_np - t1_np.min()) / (t1_np.max() - t1_np.min() + 1e-5)
    return torch.from_numpy(t1_np).unsqueeze(0).float()  # (1, D, H, W)


class SiteMRIDataset(Dataset):
    """One T1 volume per CSV row, for any of the ADNI/NACC site CSVs."""

    def __init__(self, csv_path, output_size=OUTPUT_SIZE, apply_padding=True, max_rows=None):
        self.df = prepare_site_table(csv_path)
        if max_rows is not None:
            self.df = self.df.head(max_rows).reset_index(drop=True)
        self.padding = Padding(output_size) if apply_padding else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row["label"])
        return {
            "t1_image": _load_t1_tensor(row["image_path"], self.padding),
            "label": torch.tensor(label, dtype=torch.long),
            "dx": "CN" if label == CN_LABEL else "AD",
            "sex": str(row["sex"]),
        }


# Backward-compatible name used elsewhere in this file / earlier smoke tests
MonaiMRIDatasetADNI = SiteMRIDataset


# ---------------------------------------------------------------------------
# Standardized interface for the driver entrypoint
# ---------------------------------------------------------------------------
def dataset_recipe_fn(dataset_fp):
    torch_dataset = SiteMRIDataset(dataset_fp)
    classes = torch_dataset.df["label"].astype(int).tolist()
    examples_per_class = {cid: classes.count(cid) for cid in set(classes)}
    return ModelDatasetClassification(x=torch_dataset, size=len(classes),
                                      examples_per_class=examples_per_class)


def get_dataset_recipe_fn():
    return dataset_recipe_fn


class _RandomVolume(Dataset):
    """Stand-in used only when no partition exists yet (e.g. on the controller)."""

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {"t1_image": torch.rand(1, *OUTPUT_SIZE), "label": torch.tensor(0), "dx": "CN", "sex": "M"}


def get_dummy_dataset(dummy_data_filepath):
    """One volume for the entrypoint's startup evaluate() call. Reading a whole
    partition here would make every learner start slowly."""
    if dummy_data_filepath and os.path.exists(dummy_data_filepath) and dummy_data_filepath.endswith(".csv"):
        return SiteMRIDataset(dummy_data_filepath, max_rows=1)
    return _RandomVolume()


# ---------------------------------------------------------------------------
# Partitioning (same class-stratified logic as the AD experiments)
# ---------------------------------------------------------------------------
class ProductionDataLoader:

    @staticmethod
    def get_distribution_list(distribution_name, num_learners, total_rows=None):
        if distribution_name == "iid_all":
            # The whole training CSV, split evenly across all learners
            return [total_rows // num_learners] * num_learners
        if distribution_name == "homo_100": return [100] * num_learners
        if distribution_name == "homo_200": return [200] * num_learners
        if distribution_name == "homo_300": return [300] * num_learners
        if distribution_name == "homo_400": return [400] * num_learners
        if distribution_name == "hetero_50": return [450] + ([50] * (num_learners - 1))
        if distribution_name == "hetero_150": return [550] + ([150] * (num_learners - 1))
        if distribution_name == "hetero_250": return [650] + ([250] * (num_learners - 1))
        if distribution_name == "hetero_350": return [750] + ([350] * (num_learners - 1))
        raise ValueError(f"Unknown distribution strategy: {distribution_name}")

    @staticmethod
    def load_partitioned_dataset(federation_environment, distribution_name="iid_all"):
        print(f"Loading master dataset from {LDM_MASTER_TRAIN_CSV} for partitioning...")
        df = pd.read_csv(LDM_MASTER_TRAIN_CSV)
        label_col = "DX"
        if label_col not in df.columns:
            raise ValueError(f"No '{label_col}' column in {LDM_MASTER_TRAIN_CSV}")

        class_proportions = df[label_col].value_counts(normalize=True)
        learners = federation_environment.learners.learners
        row_distribution = ProductionDataLoader.get_distribution_list(
            distribution_name, len(learners), total_rows=len(df))
        print(f"Applying data distribution across {len(learners)} learners: {row_distribution}")

        available_df = df.copy()
        for i, (learner, target_rows) in enumerate(zip(learners, row_distribution)):
            partition_dfs = []
            for cls_label, proportion in class_proportions.items():
                n_samples = int(np.round(target_rows * proportion))
                cls_df = available_df[available_df[label_col] == cls_label]
                # iid_all can round one row past what's left on the last learner
                n_samples = min(n_samples, len(cls_df)) if distribution_name == "iid_all" else n_samples
                if len(cls_df) < n_samples:
                    raise ValueError(f"Not enough data left for class {cls_label} "
                                     f"(needs {n_samples}, has {len(cls_df)}).")
                sampled = cls_df.sample(n=n_samples, random_state=42 + i)
                partition_dfs.append(sampled)
                available_df = available_df.drop(sampled.index)

            learner_df = pd.concat(partition_dfs).sample(frac=1, random_state=42 + i).reset_index(drop=True)
            out_path = learner.dataset_configs.train_dataset_path
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            learner_df.to_csv(out_path, index=False)
            print(f"Learner {learner.learner_id} partitioned: {len(learner_df)} rows saved to {out_path}")