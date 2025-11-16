# app.py — Demo Streamlit dự đoán trái cây theo ảnh upload (phong cách tối giản)
import json
from pathlib import Path
import inspect
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Fruit Classifier", page_icon="🍎", layout="centered")

# ====== HẰNG SỐ CẤU HÌNH GIỐNG LÚC ĐẦU ======
BASE = Path.cwd()
MODEL_PATH = BASE / "outputs_multi" / "fruit_model.keras"      # hoặc .h5 nếu bạn đã convert
CLASSMAP_PATH = BASE / "outputs_multi" / "class_indices.json"
IMG_SIZE = (224, 224)

# ====== TƯƠNG THÍCH HIỂN THỊ ẢNH CHO MỌI PHIÊN BẢN STREAMLIT ======
def show_image(img, caption=None):
    """Hiển thị ảnh tương thích nhiều phiên bản Streamlit."""
    params = inspect.signature(st.image).parameters
    if "use_container_width" in params:
        st.image(img, caption=caption, use_container_width=True)
    else:
        st.image(img, caption=caption, use_column_width=True)

# ====== TẢI MODEL / CLASS NAMES ======
@st.cache_resource(show_spinner=False)
def load_model():
    m = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return m

def _read_class_names(path: Path):
    """Trả về danh sách tên lớp theo index.
    Hỗ trợ:
      1) { "apple": 0, "banana": 1, ... }  (label -> index)
      2) { "0": "apple", "1": "banana", ... } (index -> label, key dạng chuỗi số)
      3) ["apple", "banana", ...] (list theo index)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # list -> dùng trực tiếp
    if isinstance(data, list):
        return data

    # dict -> phân biệt 2 kiểu
    if isinstance(data, dict):
        keys = list(data.keys())
        # index->label (key là số)
        if all(k.isdigit() for k in keys):
            pairs = sorted(((int(k), v) for k, v in data.items()), key=lambda x: x[0])
            return [label for _, label in pairs]
        # label->index
        vals = list(data.values())
        if all(isinstance(v, int) for v in vals):
            pairs = sorted(((v, k) for k, v in data.items()), key=lambda x: x[0])
            return [label for _, label in pairs]
        # fallback: trả về theo key
        return list(data.keys())

    # fallback cuối
    return [str(x) for x in data]

@st.cache_resource(show_spinner=False)
def load_classes():
    return _read_class_names(CLASSMAP_PATH)

# ====== KHỞI TẠO ======
model = load_model()
classes = load_classes()

st.title("🍎🍌🍊 Fruit Classifier Demo")
st.caption("Upload ảnh để mô hình dự đoán loại trái cây (MobileNetV2 fine-tune).")

files = st.file_uploader(
    "Chọn 1 hoặc nhiều ảnh",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True
)

# ====== SUY LUẬN ======
if files:
    for f in files:
        # Đọc & chuẩn hoá ảnh
        img = Image.open(f).convert("RGB").resize(IMG_SIZE, Image.BICUBIC)
        x = np.asarray(img, dtype=np.float32)[None, ...] / 255.0

        # Dự đoán
        logits = model.predict(x, verbose=0)
        probs = tf.nn.softmax(logits, axis=1).numpy()[0]

        idx = int(np.argmax(probs))
        pred_name = classes[idx]
        conf = float(probs[idx]) * 100.0

        # Hiển thị
        show_image(img, caption=f.name)
        st.markdown(f"**Dự đoán:** `{pred_name}`  —  **Độ tự tin:** **{conf:.2f}%**")

        # Top-3
        top3 = probs.argsort()[-3:][::-1]
        st.write("**Top-3:**")
        for k in top3:
            st.write(f"- {classes[int(k)]}: {probs[int(k)]*100:.2f}%")
        st.divider()
else:
    st.info("Hãy chọn ảnh để bắt đầu.")
