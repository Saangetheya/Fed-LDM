import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import nibabel as nib
import numpy as np
import random
from scipy.ndimage import zoom
import SimpleITK as sitk

def load_sitk_image(path):
    image = sitk.ReadImage(path)
    return image

def resize_3d(volume, target_shape):
    """
    Resize a 3D volume to target shape using scipy.ndimage.zoom
    
    Args:
        volume (np.ndarray): Input 3D volume of shape [H, W, L]
        target_shape (tuple): Target shape [h, w, l]
        
    Returns:
        np.ndarray: Resized volume of shape [h, w, l]
    """
    # Calculate zoom factors for each dimension
    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
    
    # Apply zoom
    resized_volume = zoom(volume, zoom_factors, order=3)  # order=3 for cubic interpolation
    
    return resized_volume

class MonaiMRIDataset(Dataset):
    def __init__(self, root_dir, index_file, output_size, apply_padding=True):
        with open(os.path.join(root_dir, index_file), 'r') as f:
            self.file_paths = [line.strip() for line in f if line.strip()]
        
        self.root_dir = root_dir
        self.output_size = output_size
        self.apply_padding = apply_padding

        if self.apply_padding:
            self.padding = Padding(output_size)

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        item = self.file_paths[idx].split(",")
        
        t1_image_path = os.path.join(item[1])
        
        # Load with SimpleITK
        t1_image = load_sitk_image(t1_image_path)

        sample = {'t1_image': t1_image}

        # Apply padding if needed
        if self.apply_padding:
            sample = self.padding(sample)

        # Convert to NumPy
        t1_np = sitk.GetArrayFromImage(sample['t1_image'])  # (D, H, W)

        # Normalize to [0, 1]
        t1_np = (t1_np - t1_np.min()) / (t1_np.max() - t1_np.min() + 1e-5)

        # Convert to tensors and add channel dim
        t1_tensor = torch.from_numpy(t1_np).unsqueeze(0).float()  # (1, D, H, W)

        return {"t1_image": t1_tensor, "dx": str(row["DX"]), "sex": str(row["SEX"])}

# Diagnosis labels shared by all sites: 0 = CN, 1 = AD. NULL_LABEL (2) is the "no label"
# token used for classifier-free guidance in train_diffusion.py.
CN_LABEL, AD_LABEL, NULL_LABEL = 0, 1, 2


def _adni_label(dx):
    return {"CN": CN_LABEL, "Dementia": AD_LABEL}[str(dx).strip()]


def _nacc_label(dx_adsp):
    # ASSUMPTION: DX_ADSP 1 = CN, 3 = AD (only values present in the ADvsCN split). Verify against the NACC dictionary.
    return {1: CN_LABEL, 3: AD_LABEL}[int(dx_adsp)]


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
        t1_tensor = _load_t1_tensor(str(row["FED_Path"]).strip(), self.padding)
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
        t1_tensor = _load_t1_tensor(str(row["FED_Path"]).strip(), self.padding)
        return {
            "t1_image": t1_tensor,
            "label": torch.tensor(_nacc_label(row["DX_ADSP"]), dtype=torch.long),
            "sex": str(row["SEX"]),
        }

class Resample(object):
    """
    Resample the volume in a sample to a given voxel size

      Args:
          voxel_size (float or tuple): Desired output size.
          If float, output volume is isotropic.
          If tuple, output voxel size is matched with voxel size
          Currently only support linear interpolation method
    """

    def __init__(self, new_resolution, check):
        self.name = 'Resample'

        # assert isinstance(new_resolution, (float, tuple))
        if isinstance(new_resolution, float):
            self.new_resolution = new_resolution
            self.check = check
        else:
            # assert len(new_resolution) == 3
            self.new_resolution = new_resolution
            self.check = check

    def __call__(self, sample):
        image = sample['t1_image']

        new_resolution = self.new_resolution
        check = self.check

        if check is True:
            image = resample_sitk_image(image, spacing=new_resolution, interpolator=_interpolator_image)

            return {'t1_image': image}

        if check is False:
            return {'t1_image': image}

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

