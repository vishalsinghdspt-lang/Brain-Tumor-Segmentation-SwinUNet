import os

from torch.utils.data import DataLoader

from training.dataset import BraTSDataset


def get_dataloader():

    root_dir = (
        "dataset/archive/"
        "BraTS2021_Training_Data"
    )

    patients = sorted(
        [
            patient
            for patient in os.listdir(root_dir)
            if patient.startswith("BraTS")
        ]
    )

    dataset = BraTSDataset(
        patients[:5],
        root_dir
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True
    )

    return loader

if __name__ == "__main__":

    loader = get_dataloader()

    for image, mask in loader:

        print(
            "Image Shape:",
            image.shape
        )

        print(
            "Mask Shape:",
            mask.shape
        )

        break