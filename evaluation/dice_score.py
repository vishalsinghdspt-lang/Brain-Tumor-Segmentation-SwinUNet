import torch


def dice_score(
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

    score = (
        2.0 * intersection
    ) / (
        prediction.sum()
        + target.sum()
        + 1e-8
    )

    return score.item()