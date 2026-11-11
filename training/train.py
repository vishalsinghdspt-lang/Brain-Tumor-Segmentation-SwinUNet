import torch
import matplotlib.pyplot as plt
from models.swin_unet import build_model

from training.validation_loop import validate_one_epoch
from training.metrics import (
    dice_score,
    iou_score
)


from training.optimizer import (
    get_optimizer
)

from training.dataloader import (
    get_dataloader
)

from training.training_loop import (
    train_one_epoch
)
from training.checkpoint import (
    save_checkpoint
)


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    model = build_model()

    model = model.to(device)

    loader = get_dataloader()

    optimizer = get_optimizer(
        model
    )

    print(
        "Starting Training..."
    )

    loss = train_one_epoch(
        model,
        loader,
        optimizer,
        device
    )

    print(
        "Training Loss:",
        loss
    )

    val_loss, dice, iou = validate_one_epoch(
        model,
        loader,
        device
    )


    print(
        "Validation Loss:",
        val_loss
)

    with open(
        "outputs/training_results.txt",
        "w"
    ) as f:

        f.write(
            f"Training Loss: {loss}\n"
        )

        with open(
            "outputs/val_loss.txt",
            "w"
        ) as f:
            f.write(
                f"Validation Loss: {val_loss}\n"
            )
            plt.figure(figsize=(6,4))
            plt.plot(
                [loss],
                marker="o"
            )
            plt.title("Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.savefig(
                "outputs/loss_curve.png"
            )
            plt.close()

            plt.figure(figsize=(6,4))
            plt.plot(
                [val_loss],
                marker="o"
            )
            plt.title("Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.savefig(
                "outputs/validation_curve.png"
            )
            plt.close()

        with open(
            "outputs/metrics.txt",
            "w"
        ) as f:
            f.write(
                f"DICE SCORE: {dice}\n"
            )
            f.write(
                f"IOU SCORE: {iou}\n"
            )

    save_checkpoint(
        model,
        "checkpoints/model.pth"
    )

    print(
        "Checkpoint Saved Successfully"
    )

    print(
        "Model Loaded Successfully"
    )

    print(
        "Dataloader Created Successfully"
    )

    print(
        "Optimizer Created Successfully"
    )
if __name__ == "__main__":
    main()