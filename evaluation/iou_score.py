import torch


def iou_score(
    prediction,
    target
):

    prediction = torch.argmax(
        prediction,
        dim=1
    )

    intersection = (
        prediction * target
    ).sum()

    union = (
        prediction.sum()
        + target.sum()
        - intersection
    )

    return (
        intersection
        /
        (union + 1e-8)
    ).item()