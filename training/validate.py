from training.dataset import load_patient

patient_id = "BraTS2021_00000"

flair, t1, t1ce, t2, seg = load_patient(patient_id)

print("FLAIR:", flair.shape)
print("T1:", t1.shape)
print("T1CE:", t1ce.shape)
print("T2:", t2.shape)
print("SEG:", seg.shape)