import torch


def save_checkpoint(
    model,
    path
):

    torch.save(
        model.state_dict(),
        path
    )