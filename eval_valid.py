# eval_valid.py
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

BASE = Path(__file__).resolve().parent

VAL_DIR = BASE / "valid_data"                # đổi đúng tên thư mục của bạn
MODEL_PATH = BASE / "outputs_multi" / "fruit_model.keras"
CLASSMAP_PATH = BASE / "outputs_multi" / "class_indices.json"
IMG_SIZE = 224
BATCH = 32

# ----- Load class map (nếu có) -----
classes_from_json = None
if CLASSMAP_PATH.exists():
    with open(CLASSMAP_PATH, "r", encoding="utf-8") as f:
        mp = json.load(f)         # ví dụ: {"0":"apple", "1":"banana", ...}
    classes_from_json = [mp[str(i)] for i in range(len(mp))]

# ----- Load model -----
model = tf.keras.models.load_model(MODEL_PATH)

# KHÔNG shuffle để giữ thứ tự file_paths
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    shuffle=False,
)

# Class names theo thư mục validation
class_names = val_ds.class_names
num_classes = len(class_names)

# Nếu bạn muốn dùng thứ tự class theo class_indices.json (nếu khớp khi train)
# thì có thể in ra so sánh:
print("Classes in valid_data:", class_names)
if classes_from_json:
    print("Classes in class_indices.json:", classes_from_json)

# Lấy y_true và dự đoán
y_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0)
probs = model.predict(val_ds, verbose=0)
y_pred = probs.argmax(axis=1)

# Per-class accuracy
print("\nPer-class accuracy:")
for i, name in enumerate(class_names):
    mask = (y_true == i)
    n = int(mask.sum())
    if n == 0:
        acc = 0.0
    else:
        acc = float((y_pred[mask] == i).sum()) / n
    print(f"{name:>10}: {(y_pred[mask] == i).sum():>d}/{n} = {acc:.3f}")

# Overall
overall = float((y_pred == y_true).mean())
print(f"\nOverall accuracy: {overall:.3f}")
