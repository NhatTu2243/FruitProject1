# app.py
# Fruit classifier (14 classes) + Unknown detector
# Streamlit + TensorFlow Keras (.keras model)

import io
import json
import time
import numpy as np
import streamlit as st
from PIL import Image

# OpenCV cho tiền xử lý. Nếu không có sẽ cảnh báo rõ.
try:
    import cv2
    _HAS_CV2 = True
except Exception as e:
    _HAS_CV2 = False

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import load_model


# =========================
# Utils
# =========================
@st.cache_resource(show_spinner=False)
def load_keras_model(model_path: str):
    t0 = time.time()
    model = load_model(model_path, compile=False)
    dt = time.time() - t0
    st.info(f"✅ Đã nạp model: `{model_path}` (t={dt:.2f}s)")
    return model


@st.cache_resource(show_spinner=False)
def load_class_indices(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)  # {"cachua":0, "cam":1, ...}
    num_classes = max(class_to_idx.values()) + 1
    idx_to_class = [None] * num_classes
    for cls, idx in class_to_idx.items():
        idx_to_class[idx] = cls
    # Kiểm tra tính đầy đủ
    assert all(lbl is not None for lbl in idx_to_class), "class_indices.json thiếu/nhảy số!"
    return class_to_idx, idx_to_class


def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """PIL RGB -> OpenCV BGR (np.uint8)"""
    rgb = np.array(pil_img.convert("RGB"))
    bgr = rgb[:, :, ::-1].copy()
    return bgr


def center_crop_square(img_rgb: np.ndarray) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img_rgb[y0:y0 + side, x0:x0 + side]


def preprocess_image(img_bgr: np.ndarray, img_size: int) -> np.ndarray:
    """
    Tiền xử lý giống lúc train:
      - BGR->RGB
      - Center-crop hình vuông
      - Resize về (img_size, img_size) với INTER_AREA
      - Scale [0,1]  (nếu lúc train dùng preprocess khác, thay đổi tại đây)
    """
    if not _HAS_CV2:
        # fallback thuần PIL nếu thiếu OpenCV (ít gặp trên Cloud)
        img_rgb = Image.fromarray(img_bgr[:, :, ::-1]).convert("RGB")
        # crop vuông
        w, h = img_rgb.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img_rgb = img_rgb.crop((left, top, left + side, top + side))
        img_rgb = img_rgb.resize((img_size, img_size))
        arr = np.array(img_rgb).astype(np.float32) / 255.0
        return arr

    # có cv2
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = center_crop_square(img_rgb)
    img_rgb = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    arr = img_rgb.astype(np.float32) / 255.0
    return arr


def tta_predict(model, img_arr: np.ndarray) -> np.ndarray:
    """
    TTA nhẹ: gốc + flip ngang. img_arr: (H,W,3) [0,1]
    """
    batch = np.stack([img_arr, img_arr[:, ::-1, :]], axis=0)  # (2,H,W,3)
    probs = model.predict(batch, verbose=0)                   # (2,C)
    return probs.mean(axis=0)                                 # (C,)


def draw_topk_bar(top_labels, top_scores):
    import pandas as pd
    df = pd.DataFrame({"label": top_labels, "score": top_scores})
    df = df.set_index("label")
    st.bar_chart(df)


def predict_one(model, idx_to_class, img_bgr, img_size, topk_view,
                conf_thresh, margin_thresh):
    img = preprocess_image(img_bgr, img_size)         # (H,W,3), float32 [0,1]
    x = img  # giữ để hiển thị nếu cần
    probs = tta_predict(model, img)                   # (C,)

    # top-k
    top_idx = np.argsort(-probs)[:topk_view]
    top_labels = [idx_to_class[i] for i in top_idx]
    top_scores = [float(probs[i]) for i in top_idx]

    # unknown rule
    max_conf = float(probs.max())
    # chênh lệch top1-top2 (ổn định bằng partition)
    if probs.size >= 2:
        top2 = np.partition(probs, -2)[-2:]
        margin = float(top2[-1] - top2[-2])
    else:
        margin = 1.0

    is_unknown = (max_conf < conf_thresh) or (margin < margin_thresh)
    if is_unknown:
        pred_label = None
    else:
        pred_label = idx_to_class[int(np.argmax(probs))]

    return {
        "probs": probs,
        "top_labels": top_labels,
        "top_scores": top_scores,
        "max_conf": max_conf,
        "margin": margin,
        "pred_label": pred_label,
        "is_unknown": is_unknown,
        "processed_rgb": (x * 255).astype(np.uint8),
    }


# =========================
# UI
# =========================
st.set_page_config(page_title="Fruit Classifier + Unknown", layout="wide")
st.title("🍎 Fruit Classifier (14 classes) + Unknown detector")

with st.sidebar:
    st.header("⚙️ Cấu hình")

    # Đường dẫn mặc định khi chạy trên Streamlit Cloud
    default_model = "outputs_multi/fruit_model.keras"
    default_json = "outputs_multi/class_indices.json"

    model_path = st.text_input("Model file (.keras)", default_model)
    class_indices_path = st.text_input("class_indices.json", default_json)

    img_size = st.number_input("Kích thước ảnh (img_size)", 64, 640, 224, 1)
    topk_view = st.slider("Top-k hiển thị", 1, 10, 3, 1)

    st.markdown("### 🚫 Phát hiện 'không phải trái cây'")
    strict = st.checkbox("Bật chế độ nghiêm ngặt", value=True)
    if strict:
        conf_thresh = st.slider("Ngưỡng tự tin (0–1)", 0.0, 1.0, 0.60, 0.01)
        margin_thresh = st.slider("Ngưỡng chênh lệch top1–top2", 0.0, 1.0, 0.25, 0.01)
    else:
        conf_thresh = st.slider("Ngưỡng tự tin (0–1)", 0.0, 1.0, 0.50, 0.01)
        margin_thresh = st.slider("Ngưỡng chênh lệch top1–top2", 0.0, 1.0, 0.20, 0.01)

    # Nạp model & class map
    load_btn = st.button("📥 Nạp model & class map", type="primary")

# Tự động nạp khi mở lần đầu
if "model" not in st.session_state or load_btn:
    try:
        model = load_keras_model(model_path)
        class_to_idx, idx_to_class = load_class_indices(class_indices_path)
        st.session_state["model"] = model
        st.session_state["idx_to_class"] = idx_to_class
        st.sidebar.success("Đã load model & class map!")
    except Exception as e:
        st.sidebar.error(f"Không thể nạp model/map: {e}")
        st.stop()

model = st.session_state["model"]
idx_to_class = st.session_state["idx_to_class"]

# Debug nhanh cho đúng thứ tự nhãn
with st.expander("🔎 Debug: idx → class"):
    st.code(", ".join(f"{i}:{lbl}" for i, lbl in enumerate(idx_to_class)), language="text")

# =========================
# Nhập ảnh & dự đoán
# =========================
left, right = st.columns([1, 1])

with left:
    st.subheader("📤 Tải ảnh lên")
    files = st.file_uploader(
        "Chọn 1-n ảnh (png/jpg/webp...)",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        accept_multiple_files=True
    )

    st.caption(
        "Gợi ý: thử thêm ảnh 'vật thể lạ' (bút/xe/biển báo) để kiểm tra bộ lọc unknown."
    )

with right:
    if files:
        for upl in files:
            try:
                pil = Image.open(io.BytesIO(upl.read())).convert("RGB")
                bgr = _pil_to_bgr(pil)

                out = predict_one(
                    model, idx_to_class, bgr,
                    img_size, topk_view,
                    conf_thresh, margin_thresh
                )

                col1, col2 = st.columns([1, 1])
                with col1:
                    st.image(pil, caption=upl.name, use_container_width=True)

                with col2:
                    if out["is_unknown"]:
                        st.warning(
                            f"⚠️ **Không phải trái cây (unknown)** — "
                            f"max conf **{out['max_conf']:.3f}**, margin **{out['margin']:.3f}**"
                        )
                    else:
                        st.success(
                            f"✅ **Pred: {out['pred_label']}** — Conf: **{out['max_conf']:.3f}** "
                            f"(margin {out['margin']:.3f})"
                        )

                    st.markdown("**Top-k:**")
                    for lbl, sc in zip(out["top_labels"], out["top_scores"]):
                        st.write(f"• {lbl}: {sc:.3f}")

                    draw_topk_bar(out["top_labels"], out["top_scores"])

            except Exception as e:
                st.error(f"Ảnh `{upl.name}` lỗi: {e}")
    else:
        st.info("Hãy tải lên ít nhất một ảnh để dự đoán.")

