from monai.networks.nets import SwinUNETR


def build_model():

    model = SwinUNETR(
        in_channels=4,
        out_channels=4,
        feature_size=48,
        use_checkpoint=False
    )

    return model


if __name__ == "__main__":

    model = build_model()

    print(model)