import nibabel as nib
import numpy as np
import cv2


def normalize_image(image):
    image = image.astype(np.float32)

    mean = np.mean(image)
    std = np.std(image)

    return (image - mean) / (std + 1e-8)


def resize_slice(slice_image, size=(128, 128)):
    return cv2.resize(
        slice_image,
        size,
        interpolation=cv2.INTER_LINEAR
    )

def horizontal_flip(image):

    return cv2.flip(image, 1)


def rotate_image(image):

    height, width = image.shape

    center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        15,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height)
    )

    return rotated

if __name__ == "__main__":

    path = (
        "dataset/archive/"
        "BraTS2021_Training_Data/"
        "BraTS2021_00000/"
        "BraTS2021_00000_flair.nii.gz"
    )

    image = nib.load(path).get_fdata()

    normalized_image = normalize_image(image)

    slice_80 = normalized_image[:, :, 80]

    flipped_slice = horizontal_flip(
        slice_80
    )

    rotated_slice = rotate_image(
        slice_80
    )

    resized_slice = resize_slice(
        slice_80,
        (128, 128)
    )

    print("Original Shape :", slice_80.shape)
    print("Resized Shape  :", resized_slice.shape)
    print("Flipped Shape :", flipped_slice.shape)
    print("Rotated Shape :", rotated_slice.shape)