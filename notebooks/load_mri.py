import nibabel as nib
import matplotlib.pyplot as plt

mri_path = r"dataset/archive/BraTS2021_Training_Data/BraTS2021_00000/BraTS2021_00000_flair.nii.gz"

mask_path = r"dataset/archive/BraTS2021_Training_Data/BraTS2021_00000/BraTS2021_00000_seg.nii.gz"

mri = nib.load(mri_path).get_fdata()
mask = nib.load(mask_path).get_fdata()

slice_number = 80

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(mri[:,:,slice_number], cmap="gray")
plt.title("MRI")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(mask[:,:,slice_number])
plt.title("Tumor Mask")
plt.axis("off")

plt.show()