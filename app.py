# app.py
# 🍎 Fruit Classifier (14 classes) + Unknown Detector
# - Tự nhận diện format class_indices.json (label->idx, idx->label hoặc list)
# - Phát hiện "không phải trái cây" bằng ngưỡng xác suất & margin
# - Uploader luôn hiển thị (không đặt trong columns)

import io
import json
import time
import numpy as np
from PIL import Image
import streamlit as st

# cv2 là tùy chọn để resize nhanh; nếu thiếu vẫn chạy với PIL
try:
    import cv2
    CV2_OK = True
except Exception:
    CV2_OK = False

st.set_page_config(page_title="Fruit Classifier", page_icon="🍎", layout="wide")

# =========================
# Sidebar cấu hình
# =========================
st.sidebar.header("⚙️ Cấu hình")
model_path = st.sidebar.text_input("Model file (.keras)", "outputs_multi/fruit_model.keras")
class_json_path = st.sidebar.text_input("class_indices.json", "outputs_multi/class_indices.json")
img_size = st.sidebar.number_input("Kích thước ảnh (img_size)", 64, 512, 224, 4)
topk_view = st.sidebar.slider("Top-k hiển thị", 1, 10, 3)

enable_unknown = st.sidebar.checkbox("🚫 Phát hiện 'không phải trái cây'", True)
strict_mode = st.sidebar.checkbox("🔒 Bật chế độ nghiêm ngặt (khuyên nghị)", True)
conf_thresh = st.sidebar.slider(
    "Ngưỡng tự tin (0–1)", 0.0, 1.0, 0.60 if strict_mode else 0.45, 0.01
)
margin_thresh = st.sidebar.slider(
    "Ngưỡng chênh lệch top1–top2", 0.0, 1.0, 0.25 if strict_mode else 0.15, 0.01
)

st.title("🍎 Fruit Classifier (14 classes) + Unknown detector")

# =========================
# Helpers
# =========================
@st.cache_data(show_spinner=False)
def load_class_map_any_format(path: str):
    """
    Trả về mapping: idx(int) -> label(str)
    Hỗ trợ 3 format:
      1) dict label->idx (values là int)
      2) dict idx->label (keys là số hoặc chuỗi số)
      3) list [label0, label1, ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    idx_to_label = {}
    # Case 3: list
    if isinstance(obj, list):
        idx_to_label = {i: str(lbl) for i, lbl in enumerate(obj)}
        return idx_to_label

    if isinstance(obj, dict):
        # Thử phát hiện "idx->label" (key là số/chuỗi số)
        keys = list(obj.keys())
        vals = list(obj.values())

        def _is_all_digit(arr):
            try:
                return all(str(k).strip().lstrip("-").isdigit() for k in arr)
            except Exception:
                return False

        # idx -> label
        if _is_all_digit(keys) and all(isinstance(v, (str, int)) for v in vals):
            for k, v in obj.items():
                idx_to_label[int(k)] = str(v)
            return idx_to_label

        # label -> idx
        if all(isinstance(v, (int, np.integer)) or str(v).strip().lstrip("-").isdigit() for v in vals):
            for lbl, idx in obj.items():
                idx_to_label[int(idx)] = str(lbl)
            return idx_to_label

        # Nếu chưa khớp, thử trường hợp value lại là chuỗi số index
        if all(isinstance(v, str) for v in vals):
            if all(v.strip().lstrip("-").isdigit() for v in vals):
                for lbl, idx in obj.items():
                    idx_to_label[int(idx)] = str(lbl)
                return idx_to_label

    raise ValueError(
        "Không nhận diện được định dạng class_indices.json. "
        "Hỗ trợ: dict(label->idx), dict(idx->label) hoặc list(labels)."
    )

@st.cache_resource(show_spinner=False)
def load_keras_model(path: str):
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)

def _pil_to_bgr(pil: Image.Image):
    arr = np.array(pil)  # RGB
    if CV2_OK and arr.ndim == 3 and arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr  # giữ RGB nếu không có cv2

def preprocess_image(bgr_or_rgb: np.ndarray, size: int):
    if CV2_OK and bgr_or_rgb.ndim == 3 and bgr_or_rgb.shape[2] == 3:
        rgb = cv2.cvtColor(bgr_or_rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    else:
        pil = Image.fromarray(bgr_or_rgb)
        pil = pil.resize((size, size), Image.BILINEAR)
        rgb = np.array(pil)
    x = rgb.astype("float32") / 255.0
    return np.expand_dims(x, 0)

def softmax(x: np.ndarray, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

def entropy_from_probs(p: np.ndarray):
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p))

def predict_one(model, idx_to_label, bgr_img, size, topk, conf_thr, margin_thr):
    import tensorflow as tf
    x = preprocess_image(bgr_img, size)
    logits = model.predict(x, verbose=0)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    probs = softmax(logits, axis=-1)[0]  # (C,)

    k = int(np.clip(topk, 1, probs.shape[0]))
    top_idx = np.argsort(-probs)[:k]
    top_scores = [float(probs[i]) for i in top_idx]
    top_labels = [idx_to_label.get(int(i), str(i)) for i in top_idx]

    top1 = float(np.max(probs))
    top2 = float(np.partition(probs, -2)[-2]) if probs.shape[0] >= 2 else 0.0
    margin = top1 - top2
    ent = float(entropy_from_probs(probs))

    is_unknown = False
    if conf_thr > 0 or margin_thr > 0:
        if top1 < conf_thr or margin < margin_thr:
            is_unknown = True

    pred_idx = int(np.argmax(probs))
    pred_label = idx_to_label.get(pred_idx, str(pred_idx))

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
    st.bar_chart({lbl: [sc * 100.0] for lbl, sc in zip(labels, scores)}, height=160)

# =========================
# NẠP TÀI NGUYÊN
# =========================
ok_cls = ok_model = False
idx_to_class = {}
try:
    idx_to_class = load_class_map_any_format(class_json_path)
    # hiển thị danh sách lớp đã nạp
    ordered = [idx_to_class[i] for i in sorted(idx_to_class)]
    st.caption("**Classes ({}):** {}".format(len(ordered), ", ".join(ordered)))
    ok_cls = True
except Exception as e:
    st.error(f"Không đọc được class_indices.json: {e}")

try:
    t0 = time.time()
    model = load_keras_model(model_path)
    t1 = time.time()
    st.success(f"✅ Đã nạp model: `{model_path}` (t={t1 - t0:.2f}s)")
    ok_model = True
except Exception as e:
    st.error(f"Không nạp được model: {e}")

if not (ok_model and ok_cls):
    st.stop()

# =========================
# Uploader
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
# Dự đoán
# =========================
for upl in files:
    try:
        pil = Image.open(io.BytesIO(upl.read())).convert("RGB")
        bgr = _pil_to_bgr(pil)

        out = predict_one(
            model=model,
            idx_to_label=idx_to_class,
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
