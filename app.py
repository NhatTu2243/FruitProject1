# app.py — Demo Streamlit dự đoán trái cây theo ảnh upload (đã fix hiển thị ảnh)
import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Fruit Classifier", page_icon="🍎", layout="centered")

BASE = Path.cwd()
MODEL_PATH = BASE / "outputs_multi" / "fruit_model.keras"
CLASSMAP_PATH = BASE / "outputs_multi" / "class_indices.json"
IMG_SIZE = (224, 224)

def show_image(img, caption=None):
    """Hiển thị ảnh an toàn trên mọi phiên bản Streamlit."""
    try:
        st.image(img, caption=caption, use_container_width=True)
    except TypeError:
        st.image(img, caption=caption, use_column_width=True)

@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def read_classes(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        mp = json.load(f)
    if isinstance(mp, list):
        return mp
    if isinstance(mp, dict):
        # { "0": "apple", ... } hoặc { "apple": 0, ... }
        if all(str(k).isdigit() for k in mp.keys()):
            return [mp[str(i)] for i in range(len(mp))]
        else:
            inv = sorted(((v, k) for k, v in mp.items()), key=lambda x: x[0])
            return [name for _, name in inv]
    return [str(x) for x in mp]

@st.cache_resource(show_spinner=False)
def load_classes():
    return read_classes(CLASSMAP_PATH)

model = load_model()
classes = load_classes()

st.title("🍎🍌🍊 Fruit Classifier Demo")
st.caption("Upload ảnh để mô hình dự đoán loại trái cây (MobileNetV2 fine-tune).")

files = st.file_uploader(
    "Chọn 1 hoặc nhiều ảnh",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True
)

if files:
    for f in files:
        img = Image.open(f).convert("RGB").resize(IMG_SIZE, Image.BICUBIC)
        x = np.asarray(img, dtype=np.float32)[None, ...] / 255.0
        logits = model.predict(x, verbose=0)
        probs = tf.nn.softmax(logits, axis=1).numpy()[0]
        idx = int(np.argmax(probs))

        show_image(img, caption=f.name)
        st.markdown(f"**Dự đoán:** `{classes[idx]}` — **Độ tự tin:** {probs[idx]*100:.2f}%")

        top3 = probs.argsort()[-3:][::-1]
        st.write("**Top-3:**")
        for k in top3:
            st.write(f"- {classes[int(k)]}: {probs[int(k)]*100:.2f}%")
        st.divider()
else:
    st.info("Hãy chọn ảnh để bắt đầu.")
