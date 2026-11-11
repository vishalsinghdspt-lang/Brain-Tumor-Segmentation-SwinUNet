# Brain Tumor Segmentation Using Swin-UNETR

## 1. Create Virtual Environment

python -m venv venv

## 2. Activate Virtual Environment

venv\Scripts\activate

## 3. Install Dependencies

pip install -r requirements.txt

## 4. Run Flask Application

cd flask_app

python app.py

## 5. Open Browser

http://127.0.0.1:5000

## 6. Upload MRI Files

BraTS2021_00000_flair.nii.gz

BraTS2021_00000_t1.nii.gz

BraTS2021_00000_t1ce.nii.gz

BraTS2021_00000_t2.nii.gz

## 7. View Results

Original MRI

Ground Truth

Prediction Mask

WT

TC

ET

## Common Issues

### Problem:
Module not found error

### Solution:

pip install -r requirements.txt

---

### Problem:
Flask application not starting

### Solution:

Activate virtual environment first:

venv\Scripts\activate

Then run:

python app.py

---

### Problem:
Model weights not found

### Solution:

Place trained checkpoints inside:

checkpoints/

before running the application.