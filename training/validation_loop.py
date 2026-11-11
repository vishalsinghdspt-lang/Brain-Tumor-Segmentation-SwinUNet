import torch

from training.losses import combined_loss

from training.metrics import dice_score, iou_score


def validate_one_epoch(
    model,
    dataloader,
    device
):

    model.eval()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    with torch.no_grad():

        for images, masks in dataloader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss = combined_loss(
                outputs,
                masks
            )

            dice = dice_score(
                outputs,
                masks
            )

            iou = iou_score(
                outputs,
                masks
            )

            running_loss += loss.item()
            running_dice += dice.item()
            running_iou += iou.item()

    return (
    running_loss,
    running_dice,
    running_iou
    )