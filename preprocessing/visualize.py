import nibabel as nib
import matplotlib.pyplot as plt


def visualize_patient(patient_id="BraTS2021_00000", slice_number=80):

    base_path = (
        f"dataset/archive/BraTS2021_Training_Data/{patient_id}"
    )

    flair_path = (
        f"{base_path}/{patient_id}_flair.nii.gz"
    )

    seg_path = (
        f"{base_path}/{patient_id}_seg.nii.gz"
    )

    flair = nib.load(flair_path).get_fdata()
    seg = nib.load(seg_path).get_fdata()

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(flair[:, :, slice_number], cmap="gray")
    plt.title("FLAIR MRI")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(seg[:, :, slice_number])
    plt.title("Tumor Mask")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    visualize_patient()