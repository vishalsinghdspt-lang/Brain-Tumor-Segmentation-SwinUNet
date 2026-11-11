from monai.losses import DiceLoss
import torch.nn as nn


dice_loss = DiceLoss(
    to_onehot_y=True,
    softmax=True
)

ce_loss = nn.CrossEntropyLoss()


def combined_loss(pred, target):

    dice = dice_loss(
        pred,
        target
    )

    ce = ce_loss(
        pred,
        target.squeeze(1)
    )

    return dice + ce