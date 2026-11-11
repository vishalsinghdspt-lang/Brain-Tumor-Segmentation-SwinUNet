import nibabel as nib

file_path = r"../dataset/archive/BraTS2021_Training_Data/BraTS2021_00000/BraTS2021_00000_flair.nii.gz"

img = nib.load(file_path)

print("Shape:", img.shape)

data = img.get_fdata()

print("Loaded Successfully")
print("Data Type:", data.dtype)
print("Min Value:", data.min())
print("Max Value:", data.max())