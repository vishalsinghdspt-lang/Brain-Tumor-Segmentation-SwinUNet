import nibabel as nib
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch.nn.functional as F


import os
import sys

import urllib.request

project_root = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.append(project_root)

from models.swin_unet import build_model

MODEL_PATH = "../checkpoints/model.pth"

MODEL_URL = "https://huggingface.co/vishalsinghdspt/brain-tumor-swinunet-model/resolve/main/model.pth"

if not os.path.exists(MODEL_PATH):

    os.makedirs("../checkpoints", exist_ok=True)

    print("Downloading model from Hugging Face...")

    urllib.request.urlretrieve(
        MODEL_URL,
        MODEL_PATH
    )

    print("Model downloaded successfully.")


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = build_model()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)
model.eval()


def predict_mri(
    flair_path,
    t1_path,
    t1ce_path,
    t2_path,
    seg_path
):

    flair = nib.load(flair_path).get_fdata()
    t1 = nib.load(t1_path).get_fdata()
    t1ce = nib.load(t1ce_path).get_fdata()
    t2 = nib.load(t2_path).get_fdata()

    image = np.stack(
        [flair, t1, t1ce, t2],
        axis=0
    )

    image = torch.tensor(
        image,
        dtype=torch.float32
    )

    image = image.unsqueeze(0)

    image = F.interpolate(
        image,
        size=(128, 128, 128),
        mode="trilinear",
        align_corners=False
    )


    image = image.to(device)

    with torch.no_grad():

        prediction = model(image)

    prediction = torch.argmax(
        prediction,
        dim=1
    )

    prediction = prediction.cpu().numpy()

    wt = (prediction > 0).astype(np.uint8)

    wt_slice = wt[
        0,
        :,
        :,
        64
    ]

    plt.imshow(
        wt_slice,
        cmap="Reds"
    )

    plt.axis("off")

    plt.savefig(
        "static/wt.png",
        bbox_inches="tight",
        dpi=300
    )

    plt.close()

    tc = np.logical_or(
    prediction == 1,
    prediction == 3
    ).astype(np.uint8)

    tc_slice = tc[
        0,
        :,
        :,
        64
    ]

    plt.imshow(
        tc_slice,
        cmap="Blues"
    )

    plt.axis("off")

    plt.savefig(
        "static/tc.png",
        bbox_inches="tight",
        dpi=300
    )

    plt.close()

    et = (prediction == 3).astype(np.uint8)

    et_slice = et[
        0,
        :,
        :,
        64
    ]

    plt.imshow(
        et_slice,
        cmap="Greens"
    )

    plt.axis("off")

    plt.savefig(
        "static/et.png",
        bbox_inches="tight",
        dpi=300
    )

    plt.close()    

    original_slice = image[
        0,
        0,
        :,
        :,
        64
    ].cpu().numpy()

    seg = nib.load(seg_path).get_fdata()

    best_slice = 0
    max_pixels = 0

    for i in range(seg.shape[2]):

        tumor_pixels = np.sum(seg[:, :, i] > 0)

        if tumor_pixels > max_pixels:
            max_pixels = tumor_pixels
            best_slice = i

    ground_truth_slice = seg[:, :, best_slice]

    plt.figure(figsize=(6,6))

    plt.imshow(
        ground_truth_slice,
        cmap="turbo",
        interpolation="nearest"
    )

    plt.axis("off")

    
    plt.savefig(
        "static/ground_truth.png",
        bbox_inches="tight",
        pad_inches=0,
        dpi=300
    )
    

    plt.close()

    plt.imshow(
        original_slice,
        cmap="bone"
    )

    plt.axis("off")

    plt.savefig(
        "static/original.png",
        bbox_inches="tight",
        dpi=300
    )

    plt.close()

    middle_slice = prediction[
        0,
        :,
        :,
        64
    ]

    plt.imshow(
        middle_slice,
        cmap="turbo"
    )

    plt.axis("off")

    plt.savefig(
        "static/prediction.png",
        bbox_inches="tight",
        dpi=300
    )

    plt.close()

    print(
        "Prediction image saved successfully"
    )

    return "/static/prediction.png"