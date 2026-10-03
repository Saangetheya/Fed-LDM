import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import SimpleITK as sitk

def load_sitk_image(path):
    image = sitk.ReadImage(path)
    return image

# Diagnosis labels shared by all sites: 0 = CN, 1 = AD. NULL_LABEL (2) is the "no label"
# token used for classifier-free guidance in train_diffusion.py -- it covers MCI, other
# diagnoses outside this AD-vs-CN scheme, and missing/NaN values.
CN_LABEL, AD_LABEL, NULL_LABEL = 0, 1, 2


def _adni_label(dx):
    # Tolerates both representations in use across ADNI CSVs: the original ADNI1/2/GO
    # fedpath files still spell this out as "CN"/"Dementia" strings, while
    # ADNI3_T1_9DOF_demographics.csv has DX remapped in place to 0/1 directly.
    if isinstance(dx, str):
        return {"CN": CN_LABEL, "Dementia": AD_LABEL}.get(dx.strip(), NULL_LABEL)
    try:
        code = int(dx)  # raises on NaN, which is the missing-DX case
    except (ValueError, TypeError):
        return NULL_LABEL
    return {CN_LABEL: CN_LABEL, AD_LABEL: AD_LABEL}.get(code, NULL_LABEL)


def _nacc_label(dx):
    # The NACC_ADvsCN_*.csv files were remapped in place (original DX_ADSP codes 1=CN,
    # 3=AD -> 0=CN, 1=AD; column itself renamed DX_ADSP -> DX for consistency with ADNI) so
    # this already matches CN_LABEL/AD_LABEL directly. Falls back to NULL_LABEL (not an
    # error) for any other value -- including NaN/missing, which int() can't convert -- in
    # case a differently-coded or incomplete CSV is ever pointed at this loader.
    try:
        code = int(dx)
    except (ValueError, TypeError):
        return NULL_LABEL
    return {CN_LABEL: CN_LABEL, AD_LABEL: AD_LABEL}.get(code, NULL_LABEL)


def _load_t1_tensor(path, padding):
    t1_image = load_sitk_image(path)
    sample = {'t1_image': t1_image}
    if padding is not None:
        sample = padding(sample)
    t1_np = sitk.GetArrayFromImage(sample['t1_image'])  # (D, H, W)
    t1_np = (t1_np - t1_np.min()) / (t1_np.max() - t1_np.min() + 1e-5)  # normalize to [0, 1]
    return torch.from_numpy(t1_np).unsqueeze(0).float()  # (1, D, H, W)


class MonaiMRIDatasetADNI(Dataset):
    def __init__(self, root_dir, output_size, apply_padding=True):
        self.root_dir = root_dir
        self.output_size = output_size
        self.apply_padding = apply_padding
        self.df = pd.read_csv(root_dir)
        self.padding = Padding(output_size) if apply_padding else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        t1_tensor = _load_t1_tensor(str(row["path"]).strip(), self.padding)
        return {
            "t1_image": t1_tensor,
            "label": torch.tensor(_adni_label(row["DX"]), dtype=torch.long),
            "dx": str(row["DX"]),
            "sex": str(row["SEX"]),
        }

class MonaiMRIDatasetNACC(Dataset):
    def __init__(self, root_dir, output_size, apply_padding=True):
        self.root_dir = root_dir
        self.output_size = output_size
        self.apply_padding = apply_padding
        self.df = pd.read_csv(root_dir)
        self.padding = Padding(output_size) if apply_padding else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        t1_tensor = _load_t1_tensor(str(row["path"]).strip(), self.padding)
        return {
            "t1_image": t1_tensor,
            "label": torch.tensor(_nacc_label(row["DX"]), dtype=torch.long),
            "sex": str(row["SEX"]),
        }

class Padding(object):
    """
    Add padding to the image if size is smaller than patch size

      Args:
          output_size (tuple or int): Desired output size. If int, a cubic volume is formed
      """

    def __init__(self, output_size):
        self.name = 'Padding'

        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

        assert all(i > 0 for i in list(self.output_size))

    def __call__(self, sample):
        image = sample['t1_image']
        size_old = image.GetSize()

        if (size_old[0] >= self.output_size[0]) and (size_old[1] >= self.output_size[1]) and (
                size_old[2] >= self.output_size[2]):
            return sample
        else:
            output_size = self.output_size
            output_size = list(output_size)
            if size_old[0] > self.output_size[0]:
                output_size[0] = size_old[0]
            if size_old[1] > self.output_size[1]:
                output_size[1] = size_old[1]
            if size_old[2] > self.output_size[2]:
                output_size[2] = size_old[2]

            output_size = tuple(output_size)

            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(image.GetSpacing())
            resampler.SetSize(output_size)

            # resample on image
            resampler.SetInterpolator(sitk.sitkBSpline)
            resampler.SetOutputOrigin(image.GetOrigin())
            resampler.SetOutputDirection(image.GetDirection())
            image = resampler.Execute(image)

            return {'t1_image': image}

