import torch

from models.swin_unet import build_model

from evaluation.dice_score import dice_score
from evaluation.iou_score import iou_score


print("Evaluation Pipeline Started")


model = build_model()

checkpoint = torch.load(
    "checkpoints/model.pth",
    map_location="cpu"
)

model.load_state_dict(
    checkpoint
)

model.eval()

print("Model Loaded Successfully")