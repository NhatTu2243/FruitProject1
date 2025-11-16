# app.py
# 🍎 Fruit Classifier (14 classes) + Unknown Detector
# - Hỗ trợ nạp model Keras (.keras) + class_indices.json
# - Phát hiện "không phải trái cây" bằng ngưỡng xác suất & margin
# - Uploader hiển thị chắc chắn (không đặt trong columns)

import io
import json
import time
import numpy as np
from PIL import Image
import streamlit as st

# cv2 là tùy chọn (để resize / BGR). Nếu thiếu vẫn chạy với PIL.
try:
    import cv2
    CV2_OK = True
except Exception:
    CV2_OK = False

# =========================
# Sidebar cấu hình
# =========================
st.set_page_config(page_title="Fruit Classifier", page_icon="🍎", layout="wide")

st.sidebar.header("⚙️ Cấu hình")
model_path = st.sidebar.text_input(
    "Model file (.keras)",
    value="outputs_multi/fruit_model.keras",
)
class_json_path = st.sidebar.text_input(
    "class_indices.json",
    value="outputs_multi/class_indices.json",
)
img_size = st.sidebar.number_input("Kích thước ảnh (img_size)", min_value=64, max_value=512, value=224, step=4)
topk_view = st.sidebar.slider("Top-k hiển thị", 1, 10, 3)

enable_unknown = st.sidebar.checkbox("🚫 Phát hiện 'không phải trái cây'", value=True)
strict_mode = st.sidebar.checkbox("🔒 Bật chế độ nghiêm ngặt (khuyên nghị)", value=True)
conf_thresh = st.sidebar.slider("Ngưỡng tự tin (0–1)", 0.0, 1.0, 0.60 if strict_mode else 0.45, 0.01)
margin_thresh = st.sidebar.slider("Ngưỡng chênh lệch top1–top2", 0.0, 1.0, 0.25 if strict_mode else 0.15, 0.01)

st.title("🍎 Fruit Classifier (14 classes) + Unknown detector")

# =========================
# Cache helpers
# =========================
@st.cache_data(show_spinner=False)
def load_class_map(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cls2idx = json.load(f)
    # Đảm bảo key là str -> int
    idx2cls = {int(v): k for k, v in cls2idx.items()}
    return idx2cls

@st.cache_resource(show_spinner=False)
def load_keras_model(path: str):
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)

# =========================
# Utilities
# =========================
def _pil_to_bgr(pil: Image.Image):
    """Trả về ảnh BGR (np.uint8). Nếu không có cv2 thì vẫn trả về RGB dạng np.uint8."""
    arr = np.array(pil)  # RGB
    if CV2_OK:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr  # dùng RGB luôn

def preprocess_image(bgr_or_rgb: np.ndarray, size: int):
    """Resize + scale về [0,1]. Trả về tensor shape (1, size, size, 3)."""
    if CV2_OK and bgr_or_rgb.ndim == 3 and bgr_or_rgb.shape[2] == 3:
        # Nếu là BGR -> RGB để đồng nhất
        rgb = cv2.cvtColor(bgr_or_rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    else:
        # Không có cv2 hoặc đầu vào đã RGB
        pil = Image.fromarray(bgr_or_rgb)
        pil = pil.resize((size, size), Image.BILINEAR)
        rgb = np.array(pil)
    x = rgb.astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)
    return x

def softmax(x: np.ndarray, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

def entropy_from_probs(p: np.ndarray):
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p))

def predict_one(model, idx_to_class, bgr_img, size, topk, conf_thr, margin_thr):
    import tensorflow as tf

    x = preprocess_image(bgr_img, size)
    logits = model.predict(x, verbose=0)
    if isinstance(logits, (list, tuple)):  # đề phòng model nhiều head
        logits = logits[0]
    probs = softmax(logits, axis=-1)[0]  # (C,)
    n_cls = probs.shape[0]

    # Top-k
    k = int(np.clip(topk, 1, n_cls))
    top_idx = np.argsort(-probs)[:k]
    top_scores = [float(probs[i]) for i in top_idx]
    top_labels = [idx_to_class.get(int(i), str(i)) for i in top_idx]

    # Unknown decision
    top1 = float(np.max(probs))
    top2 = float(np.partition(probs, -2)[-2]) if n_cls >= 2 else 0.0
    margin = top1 - top2
    ent = float(entropy_from_probs(probs))

    is_unknown = False
    if conf_thr > 0 or margin_thr > 0:
        # nếu bật unknown detector
        if top1 < conf_thr or margin < margin_thr:
            is_unknown = True

    # Nhãn dự đoán
    pred_idx = int(np.argmax(probs))
    pred_label = idx_to_class.get(pred_idx, str(pred_idx))

    return {
        "pred_idx": pred_idx,
        "pred_label": pred_label,
        "probs": probs.tolist(),
        "top_labels": top_labels,
        "top_scores": top_scores,
        "max_conf": top1,
        "margin": margin,
        "entropy": ent,
        "is_unknown": (enable_unknown and is_unknown),
    }

def draw_topk_bar(labels, scores):
    # vẽ bar chart đơn giản
    st.bar_chart(
        data={lbl: [sc * 100.0] for lbl, sc in zip(labels, scores)},
        height=160,
    )

# =========================
# Nạp model + class map
# =========================
ok_cls = ok_model = False
idx_to_class = {}
try:
    idx_to_class = load_class_map(class_json_path)
    ok_cls = True
except Exception as e:
    st.error(f"Không đọc được class_indices.json: {e}")

try:
    t0 = time.time()
    model = load_keras_model(model_path)
    t1 = time.time()
    st.success(f"✅ Đã nạp model: `{model_path}` *(t={t1 - t0:.2f}s)*")
    ok_model = True
except Exception as e:
    st.error(f"Không nạp được model: {e}")

if not (ok_model and ok_cls):
    st.stop()

# =========================
# Uploader (luôn hiển thị)
# =========================
st.subheader("📤 Tải ảnh lên để phân loại")
files = st.file_uploader(
    "Chọn 1 hoặc nhiều ảnh (png/jpg/jpeg/webp/bmp)",
    type=["png", "jpg", "jpeg", "webp", "bmp"],
    accept_multiple_files=True,
    key="uploader_main",
)
if not files:
    st.info("Hãy chọn ít nhất một ảnh để dự đoán.")
    st.stop()

# =========================
# Dự đoán từng ảnh
# =========================
for upl in files:
    try:
        pil = Image.open(io.BytesIO(upl.read())).convert("RGB")
        bgr = _pil_to_bgr(pil)

        out = predict_one(
            model=model,
            idx_to_class=idx_to_class,
            bgr_img=bgr,
            size=img_size,
            topk=topk_view,
            conf_thr=conf_thresh if enable_unknown else 0.0,
            margin_thr=margin_thresh if enable_unknown else 0.0,
        )

        st.image(pil, caption=upl.name, use_container_width=True)

        if out["is_unknown"]:
            st.warning(
                f"⚠️ **Không phải trái cây (unknown)** — "
                f"max conf **{out['max_conf']:.3f}**, "
                f"margin **{out['margin']:.3f}**, entropy **{out['entropy']:.3f}**"
            )
        else:
            st.success(
                f"✅ **Pred: {out['pred_label']}** — "
                f"Conf **{out['max_conf']:.3f}** (margin {out['margin']:.3f})"
            )

        st.caption("Top-k:")
        for lbl, sc in zip(out["top_labels"], out["top_scores"]):
            st.write(f"• {lbl}: {sc:.3f}")
        draw_topk_bar(out["top_labels"], out["top_scores"])
        st.divider()

    except Exception as e:
        st.error(f"Ảnh `{upl.name}` lỗi: {e}")
