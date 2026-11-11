from training.losses import combined_loss


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    for images, masks in dataloader:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = combined_loss(
            outputs,
            masks
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    return running_loss