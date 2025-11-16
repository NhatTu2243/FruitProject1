# train_multi.py  —  Train 14 lớp trái cây từ folder, lưu model .keras + class_indices.json

from pathlib import Path
import json, os, math
import numpy as np
import tensorflow as tf
from tensorflow import keras

# ====== Đường dẫn ======
BASE = Path(__file__).resolve().parent
TRAIN_DIR = BASE / "fruits_data"     # train
VAL_DIR   = BASE / "valid_data"      # validation
OUT_DIR   = BASE / "outputs_multi"   # nơi lưu model
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ====== Tham số ======
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 25
SEED       = 42
LR         = 1e-4
BASE_WEIGHTS = "imagenet"    # hoặc None nếu muốn train từ đầu

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)

# ====== Tạo dataset từ thư mục ======
def build_ds(root: Path, shuffle=True):
    return keras.utils.image_dataset_from_directory(
        root,
        labels="inferred",
        label_mode="int",
        class_names=None,           # tự suy ra theo thứ tự tên thư mục
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )

train_ds = build_ds(TRAIN_DIR, shuffle=True)
val_ds   = build_ds(VAL_DIR,   shuffle=False)

# Lấy tên lớp & lưu map
class_names = train_ds.class_names
class_map = {str(i): name for i, name in enumerate(class_names)}
with open(OUT_DIR / "class_indices.json", "w", encoding="utf-8") as f:
    json.dump(class_map, f, ensure_ascii=False, indent=2)
print("Classes:", class_names)

# ====== Prefetch & cache ======
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)

# ====== Augmentation & chuẩn hoá ======
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.05),
    keras.layers.RandomZoom(0.1),
])
# MobileNetV2 kỳ vọng [-1,1] → dùng preprocessing bên trong model
preprocess = keras.applications.mobilenet_v2.preprocess_input

# ====== Backbone ======
base = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights=BASE_WEIGHTS
)
base.trainable = False  # fine-tune sau

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = preprocess(x)
x = base(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(len(class_names), activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ====== Callback ======
ckpt_path = OUT_DIR / "best.keras"
callbacks = [
    keras.callbacks.ModelCheckpoint(
        ckpt_path, monitor="val_accuracy",
        save_best_only=True, mode="max", verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, mode="max",
        restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, verbose=1
    ),
]

# ====== Train giai đoạn 1 (đóng băng backbone) ======
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# ====== Fine-tune (mở khoá một phần backbone) ======
base.trainable = True
# mở khoá từ block cuối (tương đối an toàn)
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_ft = model.fit(
    train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks
)

# ====== Lưu model cuối cùng (.keras) ======
save_path = OUT_DIR / "fruit_model.keras"
model.save(save_path)
print("Saved:", save_path)

# ====== Đánh giá nhanh & in per-class accuracy ======
from collections import defaultdict
y_true, y_pred = [], []
for imgs, labels in val_ds:
    probs = model.predict(imgs, verbose=0)
    preds = np.argmax(probs, axis=1)
    y_true.extend(labels.numpy().tolist())
    y_pred.extend(preds.tolist())

y_true = np.array(y_true); y_pred = np.array(y_pred)
overall = (y_true == y_pred).mean()
print(f"Validation accuracy: {overall:.4f}")

# per-class
counts = defaultdict(int)
correct = defaultdict(int)
for t,p in zip(y_true, y_pred):
    counts[int(t)] += 1
    if t == p: correct[int(t)] += 1

print("\nPer-class accuracy:")
for i, name in enumerate(class_names):
    n = counts.get(i, 0)
    c = correct.get(i, 0)
    acc = c / n if n else 0.0
    print(f"  {name:>10}: {c}/{n} = {acc:.3f}")
