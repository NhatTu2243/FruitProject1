# app.py
# 🍎 Fruit Classifier (14 classes) + Unknown detector
# - Hỗ trợ .keras (TF 2.20), Streamlit 1.36+
# - Tương thích class_indices.json dạng {"apple":0,...} hoặc {"0":"apple",...}

import io
import json
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import tensorflow as tf


# ========== UI CONFIG ==========
st.set_page_config(page_title="🍎 Fruit Classifier + Unknown", layout="wide")
st.markdown(
    "<style>.small{opacity:.7;font-size:12px}</style>",
    unsafe_allow_html=True
)

# ========== HELPERS ==========

def show_image(img, caption=None):
    """Hiển thị ảnh tương thích nhiều phiên bản Streamlit."""
    try:
        st.image(img, caption=caption, use_container_width=True)
    except TypeError:
        st.image(img, caption=caption, use_column_width=True)


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path, compile=False)


@st.cache_data(show_spinner=False)
def load_class_names(ci_path: str) -> List[str]:
    """Đọc class_indices.json và trả về list theo index tăng dần."""
    with open(ci_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2 trường hợp phổ biến:
    # 1) {"apple":0,"banana":1,...} -> map tên->idx
    # 2) {"0":"apple","1":"banana",...} -> map idx(str)->tên
    if not data:
        raise ValueError("class_indices.json trống!")

    # Trường hợp 2: tất cả key là số
    if all(str(k).isdigit() for k in data.keys()):
        items = sorted([(int(k), v) for k, v in data.items()], key=lambda x: x[0])
        return [name for _, name in items]

    # Trường hợp 1: key là tên lớp
    items = sorted([(int(v), k) for k, v in data.items()], key=lambda x: x[0])
    return [name for _, name in items]


def preprocess_pil(pil: Image.Image, img_size: int) -> np.ndarray:
    pil = pil.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(pil).astype("float32") / 255.0
    return arr


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def entropy_of(p: np.ndarray) -> float:
    # p shape: (C,)
    p_safe = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p_safe * np.log(p_safe)) / math.log(len(p_safe)))


def predict_one(model, arr: np.ndarray) -> np.ndarray:
    # arr shape: (H, W, 3) -> (1, H, W, 3)
    logits = model.predict(arr[None, ...], verbose=0)
    probs = logits[0] if logits.ndim == 2 else logits.squeeze()
    # Nếu model trả logit, dùng softmax; nếu đã là prob thì tổng xấp xỉ 1
    if not np.isclose(np.sum(probs), 1.0, atol=1e-3):
        probs = softmax(probs)
    return probs


def decide_unknown(probs: np.ndarray, strict: bool, conf_th: float, gap_th: float) -> Tuple[bool, float, float]:
    """Trả về (is_unknown, max_conf, margin)."""
    top2 = np.sort(probs)[-2:]
    max_conf = float(top2[-1])
    margin = float(top2[-1] - top2[-2])
    if strict:
        return (max_conf < conf_th) or (margin < gap_th), max_conf, margin
    return (max_conf < conf_th) and (margin < gap_th), max_conf, margin


# ========== SIDEBAR ==========
st.sidebar.header("⚙️ Cấu hình")
model_path = st.sidebar.text_input("Model file (.keras)", "outputs_multi/fruit_model.keras")
ci_path = st.sidebar.text_input("class_indices.json", "outputs_multi/class_indices.json")
img_size = st.sidebar.number_input("Kích thước ảnh (img_size)", min_value=96, max_value=512, value=224, step=4)
topk_view = st.sidebar.slider("Top-k hiển thị", 1, 10, 3)

enable_unknown = st.sidebar.checkbox("🚫 Phát hiện 'không phải trái cây'", value=True)
strict_mode = st.sidebar.checkbox("🔒 Bật chế độ nghiêm ngặt (khuyên nghị)", value=True)
conf_th = float(st.sidebar.slider("Ngưỡng tự tin (0–1)", 0.0, 1.0, 0.60, 0.01))
gap_th = float(st.sidebar.slider("Ngưỡng chênh lệch top1–top2", 0.0, 1.0, 0.25, 0.01))

# ========== LOAD MODEL + CLASSES ==========
classes_box = st.empty()
classes_text = ""

model = None
class_names = []
ok_model = ok_classes = False

try:
    model = load_model(model_path)
    st.success(f"✅ Đã nạp model: {model_path}")
    ok_model = True
except Exception as e:
    st.error(f"Không thể nạp model: {e}")

try:
    class_names = load_class_names(ci_path)
    classes_text = ", ".join(class_names)
    ok_classes = True
except Exception as e:
    st.error(f"Không đọc được class_indices.json: {e}")

if ok_classes:
    st.caption(f"**Classes ({len(class_names)}):** {classes_text}")

st.divider()

# ========== UPLOAD ==========
st.header("📤 Tải ảnh lên để phân loại")
uploads = st.file_uploader(
    "Chọn 1 hoặc nhiều ảnh (png/jpg/jpeg/webp/bmp)",
    type=["png", "jpg", "jpeg", "webp", "bmp"],
    accept_multiple_files=True
)

if not uploads:
    st.info("Hãy tải lên một vài ảnh để bắt đầu.")
    st.stop()

if not (ok_model and ok_classes):
    st.warning("Cần nạp được **model** và **class_indices.json** trước khi dự đoán.")
    st.stop()

# ========== PREDICT ==========
cols = st.columns(3)

for idx, upl in enumerate(uploads):
    try:
        pil = Image.open(io.BytesIO(upl.read()))
    except Exception as e:
        st.error(f"Ảnh {upl.name} lỗi khi đọc: {e}")
        continue

    arr = preprocess_pil(pil, img_size)
    probs = predict_one(model, arr)

    top_idx = np.argsort(-probs)[: topk_view]
    top_labels = [class_names[i] for i in top_idx]
    top_scores = [float(probs[i]) for i in top_idx]

    ent = entropy_of(probs)
    is_unk, max_conf, margin = decide_unknown(probs, strict_mode, conf_th, gap_th) if enable_unknown else (False, float(np.max(probs)), float(np.max(probs) - np.partition(probs, -2)[-2]))

    c = cols[idx % len(cols)]
    with c:
        show_image(pil, upl.name)

        if enable_unknown and is_unk:
            st.warning(
                f"⚠️ **Không phải trái cây (unknown)** — "
                f"max conf **{max_conf:.3f}**, margin **{margin:.3f}**, entropy **{ent:.3f}**"
            )
        else:
            pred_idx = int(np.argmax(probs))
            pred_label = class_names[pred_idx]
            st.success(
                f"✅ **Pred:** {pred_label} — **Conf:** {max_conf:.3f}  "
                f"<span class='small'>(margin {margin:.3f}, entropy {ent:.3f})</span>",
                icon="✅"
            )

        # Top-k chart
        st.caption("Top-k:")
        df = pd.DataFrame({"class": top_labels, "score": [s * 100 for s in top_scores]})
        df = df.set_index("class")
        st.bar_chart(df, height=160)

st.caption(
    "Gợi ý: nếu model hay nhầm vật thể lạ là trái cây, hãy tăng **ngưỡng tự tin** hoặc **chênh lệch top1–top2**, "
    "và xem xét bổ sung ảnh 'không phải trái cây' để huấn luyện mở rộng."
)
