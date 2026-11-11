import torch


def get_optimizer(model):

    return torch.optim.AdamW(
        model.parameters(),
        lr=0.0001
    )