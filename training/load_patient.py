import os
import nibabel as nib
import numpy as np


def load_patient(patient_id):

    base_path = (
        f"dataset/archive/"
        f"BraTS2021_Training_Data/"
        f"{patient_id}"
    )

    flair = nib.load(
        f"{base_path}/{patient_id}_flair.nii.gz"
    ).get_fdata()

    t1 = nib.load(
        f"{base_path}/{patient_id}_t1.nii.gz"
    ).get_fdata()

    t1ce = nib.load(
        f"{base_path}/{patient_id}_t1ce.nii.gz"
    ).get_fdata()

    t2 = nib.load(
        f"{base_path}/{patient_id}_t2.nii.gz"
    ).get_fdata()

    seg = nib.load(
        f"{base_path}/{patient_id}_seg.nii.gz"
    ).get_fdata()

    return flair, t1, t1ce, t2, seg


if __name__ == "__main__":

    patient_id = "BraTS2021_00000"

    flair, t1, t1ce, t2, seg = load_patient(patient_id)

    print("FLAIR Shape :", flair.shape)
    print("T1 Shape    :", t1.shape)
    print("T1CE Shape  :", t1ce.shape)
    print("T2 Shape    :", t2.shape)
    print("MASK Shape  :", seg.shape)