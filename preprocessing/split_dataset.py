import os
import pandas as pd
from sklearn.model_selection import train_test_split

dataset_path = r"dataset/archive/BraTS2021_Training_Data"

patients = [
    folder
    for folder in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, folder))
]

print("Total Patients:", len(patients))

patients = patients[:300]

train_patients, valid_patients = train_test_split(
    patients,
    test_size=0.40,
    random_state=42
)

train_df = pd.DataFrame(
    train_patients,
    columns=["Patient_ID"]
)

valid_df = pd.DataFrame(
    valid_patients,
    columns=["Patient_ID"]
)

train_df.to_csv(
    "training/train.csv",
    index=False
)

valid_df.to_csv(
    "training/validation.csv",
    index=False
)

print("\nTraining Patients:", len(train_patients))
print("Validation Patients:", len(valid_patients))

print("\nCSV files created successfully!")