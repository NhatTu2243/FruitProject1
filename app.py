# app.py — Streamlit Fruit Classifier (fix hiển thị ảnh cho mọi bản Streamlit)
import json
import inspect
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------- Cấu hình ----------
st.set_page_config(page_title="Fruit Classifier", page_icon="🍎", layout="centered")

BASE = Path.cwd()
MODEL_PATH = BASE / "outputs_multi" / "fruit_model.keras"        # giữ nguyên như bạn muốn
CLASSMAP_PATH = BASE / "outputs_multi" / "class_indices.json"
IMG_SIZE = (224, 224)

# ---------- Helper: hiển thị ảnh tương thích mọi bản Streamlit ----------
def show_image(img, caption=None):
    """Hiển thị ảnh an toàn: thử use_container_width, nếu không có thì dùng use_column_width."""
    params = inspect.signature(st.image).parameters
    if "use_container_width" in params:
        st.image(img, caption=caption, use_container_width=True)
    else:
        st.image(img, caption=caption, use_column_width=True)

# ---------- Tải model & lớp ----------
@st.cache_resource(show_spinner=False)
def load_model():
    # compile=False để tránh phải khớp optimizer/metrics lúc load
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def _read_classes(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        mp = json.load(f)
    # Hỗ trợ cả dạng ["apple", ...] hoặc {"0": "apple", ...} hoặc {"apple": 0, ...}
    if isinstance(mp, list):
        return mp
    if isinstance(mp, dict):
        if all(str(k).isdigit() for k in mp.keys()):
            return [mp[str(i)] for i in range(len(mp))]
        # {"class_name": index} -> sắp theo index
        inv = sorted(((v, k) for k, v in mp.items()), key=lambda x: x[0])
        return [name for _, name in inv]
    return [str(x) for x in mp]

@st.cache_resource(show_spinner=False)
def load_classes():
    return _read_classes(CLASSMAP_PATH)

# ---------- Khởi tạo ----------
try:
    model = load_model()
    classes = load_classes()
except Exception as e:
    st.error(f"Không thể load model/class map: {e}")
    st.stop()

# ---------- UI ----------
st.title("🍎🍌🍊 Fruit Classifier Demo")
st.caption("Upload ảnh để mô hình dự đoán loại trái cây (MobileNetV2 fine-tune).")

files = st.file_uploader(
    "Chọn 1 hoặc nhiều ảnh",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True
)

# ---------- Dự đoán ----------
def predict_img(pil_img: Image.Image):
    img = pil_img.convert("RGB").resize(IMG_SIZE, Image.BICUBIC)
    x = np.asarray(img, dtype=np.float32)[None, ...] / 255.0  # đã chuẩn hoá /255 nếu model của bạn không có layer chuẩn hoá nội bộ
    logits = model.predict(x, verbose=0)
    probs = tf.nn.softmax(logits, axis=1).numpy()[0]
    top1 = int(np.argmax(probs))
    return img, probs, top1

if files:
    for f in files:
        try:
            pil = Image.open(f)
        except Exception as e:
            st.warning(f"Lỗi mở ảnh {getattr(f, 'name', '')}: {e}")
            continue

        img_resized, probs, idx = predict_img(pil)

        # HIỂN THỊ ẢNH: luôn gọi qua show_image để tránh lỗi tham số
        show_image(img_resized, caption=getattr(f, "name", "uploaded"))

        st.markdown(f"**Dự đoán:** `{classes[idx]}` — **Độ tự tin:** {probs[idx]*100:.2f}%")

        # Top-3
        top3 = probs.argsort()[-3:][::-1]
        st.write("**Top-3:**")
        for k in top3:
            st.write(f"- {classes[int(k)]}: {probs[int(k)]*100:.2f}%")
        st.divider()
else:
    st.info("Hãy chọn ảnh để bắt đầu.")

