import os
import sys
import torch

from datetime import datetime

from flask import Flask
from flask import render_template
from flask import request

# allow access to project modules
project_root = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.append(project_root)

from models.swin_unet import build_model
from predict_mri import predict_mri

from database import (
    init_db,
    save_prediction,
    get_predictions
)

app = Flask(__name__)

# ----------------------------
# MODEL LOADING
# ----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = build_model()

model.load_state_dict(
    torch.load(
        "../checkpoints/model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

print("Model Loaded Successfully")

# ----------------------------
# HOME PAGE
# ----------------------------

@app.route("/")
def home():
    return render_template("index.html")

# ----------------------------
# ABOUT PAGE
# ----------------------------

@app.route("/about")
def about():
    return render_template("about.html")

# ----------------------------
# DASHBOARD PAGE
# ----------------------------

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/history")
def history():

    records = get_predictions()

    return render_template(
        "history.html",
        records=records
    )



# ----------------------------
# DUMMY PREDICTION
# ----------------------------

    

# ----------------------------
# MRI UPLOAD
# ----------------------------

@app.route(
    "/upload",
    methods=["GET", "POST"]
)
def upload():

    if request.method == "POST":

        flair = request.files["flair"]
        t1 = request.files["t1"]
        t1ce = request.files["t1ce"]
        t2 = request.files["t2"]
        seg = request.files["seg"]


        flair_path = (
            f"uploads/{flair.filename}"
        )

        t1_path = (
            f"uploads/{t1.filename}"
        )

        t1ce_path = (
            f"uploads/{t1ce.filename}"
        )

        t2_path = (
            f"uploads/{t2.filename}"
        )

        seg_path = (
            f"uploads/{seg.filename}"
        )

        flair.save(flair_path)
        t1.save(t1_path)
        t1ce.save(t1ce_path)
        t2.save(t2_path)
        seg.save(seg_path)
        prediction = predict_mri(
            flair_path,
            t1_path,
            t1ce_path,
            t2_path,
            seg_path=seg_path
        )

        print("Prediction completed successfully")

        result_image = predict_mri(
            flair_path,
            t1_path,
            t1ce_path,
            t2_path,
            seg_path=seg_path
        )

        save_prediction(

            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            0.415,

            0.218,

            result_image

        )

        return render_template(
            "result.html",
            image=result_image
        )

    return render_template(
        "upload.html"
    )

# ----------------------------
# START FLASK
# ----------------------------

if __name__ == "__main__":

    init_db()

    app.run(
        debug=True
    )