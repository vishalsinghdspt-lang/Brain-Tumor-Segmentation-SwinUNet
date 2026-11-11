import os
import nibabel as nib
import numpy as np
import torch

from torch.utils.data import Dataset
from monai.transforms import Resize


class BraTSDataset(Dataset):

    def __init__(
        self,
        patient_ids,
        root_dir
    ):

        self.patient_ids = patient_ids
        self.root_dir = root_dir

    def __len__(self):

        return len(self.patient_ids)

    def __getitem__(self, idx):

        patient_id = self.patient_ids[idx]

        patient_path = os.path.join(
            self.root_dir,
            patient_id
        )

        flair = nib.load(
            os.path.join(
                patient_path,
                f"{patient_id}_flair.nii.gz"
            )
        ).get_fdata()

        t1 = nib.load(
            os.path.join(
                patient_path,
                f"{patient_id}_t1.nii.gz"
            )
        ).get_fdata()

        t1ce = nib.load(
            os.path.join(
                patient_path,
                f"{patient_id}_t1ce.nii.gz"
            )
        ).get_fdata()

        t2 = nib.load(
            os.path.join(
                patient_path,
                f"{patient_id}_t2.nii.gz"
            )
        ).get_fdata()

        seg = nib.load(
            os.path.join(
                patient_path,
                f"{patient_id}_seg.nii.gz"
            )
        ).get_fdata()
        seg[seg == 4] = 3
        print("Labels:", np.unique(seg))

        image = np.stack(
            [flair, t1, t1ce, t2],
            axis=0
        )

        resize_image = Resize(
            spatial_size=(128, 128, 128)
        )

        resize_mask = Resize(
            spatial_size=(128, 128, 128),
            mode="nearest"
        )

        image = resize_image(image)

        seg = torch.tensor(
            seg,
            dtype=torch.float32
        )

        seg = seg.unsqueeze(0)

        seg = resize_mask(seg)

        seg = seg.squeeze(0)

        seg = seg.long()

        seg = seg.unsqueeze(0)

        image = torch.tensor(
            image,
        dtype=torch.float32
        )

        seg = torch.tensor(
            seg,
            dtype=torch.long
        )

        return image, seg
        