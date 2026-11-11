import torch


def dice_score(pred, target):

    pred = torch.argmax(
        pred,
        dim=1
    )

    intersection = (
        pred * target
    ).sum()

    return (
        (2.0 * intersection)
        /
        (
            pred.sum()
            + target.sum()
            + 1e-8
        )
    )

def iou_score(pred, target):

    pred = torch.argmax(
        pred,
        dim=1
    )

    intersection = (
        pred * target
    ).sum()

    union = (
        pred.sum()
        + target.sum()
        - intersection
    )

    return (
        intersection
        / (union + 1e-8)
    )

def iou_score(pred, target):

    pred = torch.argmax(
        pred,
        dim=1
    )

    intersection = (
        pred * target
    ).sum()

    union = (
        pred.sum()
        + target.sum()
        - intersection
    )

    return (
        intersection
        / (union + 1e-8)
    )