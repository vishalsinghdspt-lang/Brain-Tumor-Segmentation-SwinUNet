import torch

checkpoint = torch.load(
    "checkpoints/model.pth",
    map_location="cpu"
)

print(
    "Model Checkpoint Loaded Successfully"
)