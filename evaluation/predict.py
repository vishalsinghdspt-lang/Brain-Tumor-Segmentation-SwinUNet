import torch

from models.swin_unet import build_model


def load_model():

    model = build_model()

    checkpoint = torch.load(
        "checkpoints/model.pth",
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint
    )

    model.eval()

    return model


if __name__ == "__main__":

    model = load_model()

    print(
        "Prediction Model Ready"
    )