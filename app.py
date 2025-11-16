# app.py — Fruit classifier (14 classes) + Unknown detector
# Chạy local:  streamlit run app.py

import io
import os
import json
import math
import inspect
import numpy as np
from PIL import Image, ImageOps

import streamlit as st

# TensorFlow/Keras (không cần compile để load)
import tensorflow as tf
from tensorflow.keras.models import load_model

# =========================
# Utils hiển thị ảnh (FIX)
# =========================
def show_image(img, caption=None):
    """Hiển thị ảnh tương thích nhiều phiên bản Streamlit."""
    params = inspect.signature(st.image).parameters
    if "use_container_width" in params:
        st.image(img, caption=caption, use_container_width=True)
    else:
        st.image(img, caption=caption, use_column_width=True)

# =========================
# Tải model & labels
# =========================
@st.cache_resource(show_spinner=False)
def load_keras_model(model_path: str):
    model = load_model(model_path, compile=False)
    return model

@st.cache_resource(show_spinner=False)
def load_class_indices(path: str):
    """
    Hỗ trợ 3 kiểu:
      1) { "apple": 0, "banana": 1, ... }  (label->index)
      2) { "0": "apple", "1": "banana", ... } (index->label, key là string số)
      3) ["apple", "banana", ...]  (list label theo index)
    Trả về: list[str] class_names (index -> label)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Kiểu list
    if isinstance(data, list):
        return data

    # Kiểu dict
    if isinstance(data, dict):
        # Trường hợp key là số: {"0": "apple", "1": "banana"}
        all_keys = list(data.keys())
        if all(k.isdigit() for k in all_keys):
            pairs = sorted(((int(k), v) for k, v in data.items()), key=lambda x: x[0])
            return [label for _, label in pairs]

        # Trường hợp label->index: {"apple": 0, "banana": 1}
        values = list(data.values())
        if all(isinstance(v, int) for v in values):
            # Sắp xếp theo index tăng dần
            pairs = sorted(((v, k) for k, v in data.items()), key=lambda x: x[0])
            return [label for _, label in pairs]

        # Fallback: nếu dict bất thường, trả về theo key
        return list(data.keys())

    # Fallback nữa: cố ép sang list chuỗi
    return [str(x) for x in data]

# =========================
# Tiền xử lý & dự đoán
# =========================
def center_pad_resize(img: Image.Image, target_size: int) -> Image.Image:
    """Giữ tỉ lệ, thêm viền đen để thành vuông, rồi resize về target_size."""
    img = ImageOps.exif_transpose(img.convert("RGB"))
    w, h = img.size
    side = max(w, h)
    pad_img = Image.new("RGB", (side, side), (0, 0, 0))
    pad_img.paste(img, ((side - w) // 2, (side - h) // 2))
    return pad_img.resize((target_size, target_size), Image.BICUBIC)

def preprocess(img: Image.Image, img_size: int) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def softmax(x: np.ndarray, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def entropy_of(p: np.ndarray) -> float:
    """Entropy base-e của một phân phối p (1D)."""
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p)))

def predict_one(model, img: Image.Image, img_size: int) -> np.ndarray:
    x = preprocess(img, img_size)
    logits = model(x, training=False).numpy()
    probs = softmax(logits, axis=-1)[0]
    return probs

def decide_unknown(probs: np.ndarray, strict: bool, thr_conf: float, thr_margin: float):
    """
    Quyết định 'không phải trái cây' (unknown) dựa trên:
      - max_conf < thr_conf
      - (strict) chênh lệch top1 - top2 < thr_margin
    Trả về: (is_unknown: bool, max_conf, margin, entropy)
    """
    top2 = np.sort(probs)[-2:]  # [top2, top1]
    max_conf = float(top2[-1])
    second = float(top2[-2]) if probs.size >= 2 else 0.0
    margin = max_conf - second
    ent = entropy_of(probs)

    is_unknown = (max_conf < thr_conf) or (strict and (margin < thr_margin))
    return is_unknown, max_conf, margin, ent

# =========================
# Giao diện
# =========================
st.set_page_config(page_title="Fruit Classifier", page_icon="🍑", layout="wide")

with st.sidebar:
    st.header("⚙️ Cấu hình")
    model_path = st.text_input("Model file (.keras)", "outputs_multi/fruit_model.keras")
    indices_path = st.text_input("class_indices.json", "outputs_multi/class_indices.json")
    img_size = st.number_input("Kích thước ảnh (img_size)", 64, 1024, 224, step=16)
    topk_view = st.slider("Top-k hiển thị", 1, 10, 3)

    enable_unknown = st.checkbox("🚫 Phát hiện 'không phải trái cây'", value=True)
    strict_mode = st.checkbox("🔒 Bật chế độ nghiêm ngặt (khuyến nghị)", value=True)

    thr_conf = st.slider("Ngưỡng tự tin (0–1)", 0.0, 1.0, 0.60, 0.01)
    thr_margin = st.slider("Ngưỡng chênh lệch top1–top2", 0.0, 1.0, 0.25, 0.01)

st.title("🍑 Fruit Classifier (14 classes) + Unknown detector")

# Tải model
model = None
class_names = None

# Model
if os.path.exists(model_path):
    try:
        model = load_keras_model(model_path)
        st.success(f"✅ Đã nạp model: {model_path}")
    except Exception as e:
        st.error(f"Không thể nạp model: {e}")

else:
    st.warning("⚠️ Không tìm thấy file model. Hãy kiểm tra đường dẫn.")

# Labels
if os.path.exists(indices_path):
    try:
        class_names = load_class_indices(indices_path)
        st.caption(f"Classes ({len(class_names)}): " + ", ".join(class_names))
    except Exception as e:
        st.error(f"Không đọc được class_indices.json: {e}")
else:
    st.warning("⚠️ Không tìm thấy class_indices.json.")

# Upload
st.subheader("📤 Tải ảnh lên để phân loại")
uploads = st.file_uploader(
    "Chọn 1 hoặc nhiều ảnh (png/jpg/jpeg/webp/bmp)",
    accept_multiple_files=True,
    type=["png", "jpg", "jpeg", "webp", "bmp"],
)

# =========================
# Suy luận
# =========================
if uploads and model is not None and class_names is not None:
    cols = st.columns(3)

    for i, file in enumerate(uploads):
        col = cols[i % 3]
        with col:
            try:
                # Đọc ảnh
                img = Image.open(io.BytesIO(file.read()))
                show_image(img, caption=file.name)

                # Chuẩn hoá & dự đoán
                img_proc = center_pad_resize(img, img_size)
                probs = predict_one(model, img_proc, img_size)

                # Quyết định unknown
                is_unknown, max_conf, margin, ent = decide_unknown(
                    probs, strict_mode, thr_conf, thr_margin
                )

                # Top-k
                idx_top = np.argsort(-probs)[:topk_view]
                top_labels = [class_names[j] for j in idx_top]
                top_scores = [float(probs[j]) for j in idx_top]

                if enable_unknown and is_unknown:
                    st.warning(
                        f"⚠️ **Không phải trái cây (unknown)** — "
                        f"max conf **{max_conf:.3f}**, margin **{margin:.3f}**, entropy **{ent:.3f}**"
                    )
                else:
                    pred_idx = int(np.argmax(probs))
                    pred_label = class_names[pred_idx]
                    st.success(
                        f"✅ **Pred:** {pred_label} — **Conf:** {max_conf:.3f}  "
                        f"(margin {margin:.3f}, entropy {ent:.3f})"
                    )

                with st.expander("Top-k:"):
                    for lbl, sc in zip(top_labels, top_scores):
                        st.write(f"• **{lbl}**: {sc:.3f}")

            except Exception as e:
                st.error(f"Ảnh **{file.name}** lỗi: {e}")

elif uploads and (model is None or class_names is None):
    st.info("Hãy đảm bảo đã nạp **model** và **class_indices.json** trước khi phân loại.")
