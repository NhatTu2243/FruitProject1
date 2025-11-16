# app.py — Streamlit demo cho phân loại trái cây (có "unknown" bằng threshold)
import json
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# ===================== Cấu hình cơ bản =====================
BASE = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE / "outputs_multi" / "fruit_model.keras"
DEFAULT_CLASSMAP = BASE / "outputs_multi" / "class_indices.json"
DEFAULT_IMG_SIZE = 224

st.set_page_config(page_title="Fruit Classifier", page_icon="🍎", layout="wide")
st.title("🍎🍌🍊 Fruit Classifier – Streamlit App")

# ===================== Tiện ích =====================
@st.cache_resource(show_spinner=False)
def load_classes(class_map_path: Path):
    with open(class_map_path, "r", encoding="utf-8") as f:
        mp = json.load(f)  # {"0": "apple", ...}
    classes = [mp[str(i)] for i in range(len(mp))]
    return classes

@st.cache_resource(show_spinner=True)
def safe_load_model(model_path: Path):
    """
    Load model. Nếu model cũ có Lambda(preprocess_input) thì thêm custom_objects.
    Model hiện tại dùng chuẩn hóa trong graph nên thường load trực tiếp.
    """
    try:
        return tf.keras.models.load_model(model_path)
    except Exception:
        return tf.keras.models.load_model(
            model_path,
            custom_objects={"preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input},
        )

def prepare_image(pil_img: Image.Image, img_size: int = DEFAULT_IMG_SIZE) -> np.ndarray:
    """Resize đúng kích thước; KHÔNG /255 vì model đã có lớp chuẩn hóa."""
    img = pil_img.convert("RGB").resize((img_size, img_size))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr

def predict_one(model, classes, pil_img: Image.Image, topk: int, img_size: int):
    x = prepare_image(pil_img, img_size)
    probs = model.predict(x, verbose=0)[0]  # (C,)
    top_indices = np.argsort(probs)[::-1][:topk]
    top_labels = [classes[i] for i in top_indices]
    top_scores = [float(probs[i]) for i in top_indices]
    pred_idx = int(np.argmax(probs))
    return classes[pred_idx], float(probs[pred_idx]), list(zip(top_labels, top_scores)), probs

def apply_unknown(pred_label: str, pred_conf: float, threshold: float) -> str:
    """Nếu xác suất < threshold, trả về 'unknown'."""
    return pred_label if pred_conf >= threshold else "unknown"

# ===================== Sidebar =====================
st.sidebar.header("⚙️ Cấu hình")
model_path = Path(st.sidebar.text_input("Model file", str(DEFAULT_MODEL)))
classmap_path = Path(st.sidebar.text_input("class_indices.json", str(DEFAULT_CLASSMAP)))
img_size = st.sidebar.number_input("Kích thước ảnh (img_size)", 64, 512, DEFAULT_IMG_SIZE, step=32)
topk = st.sidebar.slider("Top-k", 1, 10, 3)

st.sidebar.subheader("🛡️ Phát hiện 'không phải trái cây'")
threshold = st.sidebar.slider("Ngưỡng tự tin (0–1)", 0.0, 1.0, 0.60, step=0.01)
st.sidebar.caption("Nếu xác suất dự đoán cao nhất < ngưỡng → gán 'unknown'.")

show_prob_table = st.sidebar.checkbox("Hiện bảng xác suất đầy đủ", value=False)

# Cache: load model & classes
try:
    classes = load_classes(classmap_path)
    model = safe_load_model(model_path)
    st.sidebar.success(f"Đã load model: {model_path.name}")
except Exception as e:
    st.sidebar.error(f"Không load được model/class map: {e}")
    st.stop()

st.sidebar.write(f"**Classes ({len(classes)}):**")
st.sidebar.write(", ".join(classes))

# ===================== Tabs giao diện =====================
tab1, tab2 = st.tabs(["📤 Upload ảnh", "📁 Dự đoán cả thư mục"])

# ---- Tab 1: Upload ảnh ----
with tab1:
    files = st.file_uploader(
        "Chọn 1 hoặc nhiều ảnh (jpg/png/webp/bmp...)",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=True
    )
    if files:
        cols = st.columns(3)
        for i, f in enumerate(files):
            try:
                pil = Image.open(f)
                pred, conf, top_list, all_probs = predict_one(
                    model, classes, pil, topk=topk, img_size=img_size
                )
                final_label = apply_unknown(pred, conf, threshold)

                with cols[i % 3]:
                    st.image(pil, caption=f.name, use_column_width=True)
                    if final_label == "unknown":
                        st.markdown(f"**⚠️ Không chắc (có thể không phải trái cây)** — max conf: `{conf:.3f}`")
                        st.markdown(f"*Gợi ý:* tăng ngưỡng, hoặc thu thập thêm dữ liệu 'non-fruit' để huấn luyện mở rộng.")
                    else:
                        st.markdown(f"**✅ Pred:** `{final_label}` — **Conf:** `{conf:.3f}`")

                    st.markdown("**Top-k:**")
                    for lbl, sc in top_list:
                        st.write(f"- {lbl}: {sc:.3f}")

                    if show_prob_table:
                        import pandas as pd
                        df_prob = pd.DataFrame({"class": classes, "probability": all_probs}).set_index("class")
                        st.bar_chart(df_prob["probability"])
                        st.caption("Xác suất theo lớp (theo class_indices.json)")
            except Exception as e:
                st.warning(f"Lỗi xử lý {f.name}: {e}")

# ---- Tab 2: Dự đoán thư mục ----
with tab2:
    st.info("Nhập đường dẫn thư mục ảnh trên máy (Windows): ví dụ `C:\\Users\\nhatt\\Pictures\\fruits_test`")
    folder = st.text_input("Đường dẫn thư mục")
    run = st.button("Quét & Dự đoán")
    if run:
        p = Path(folder)
        if not p.exists() or not p.is_dir():
            st.error("Thư mục không tồn tại.")
        else:
            exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            imgs = [fp for fp in p.rglob("*") if fp.suffix.lower() in exts]
            if not imgs:
                st.warning("Không tìm thấy ảnh hợp lệ trong thư mục.")
            else:
                rows = []
                prog = st.progress(0, text="Đang dự đoán...")
                for idx, fp in enumerate(imgs, start=1):
                    try:
                        pil = Image.open(fp)
                        pred, conf, top_list, _ = predict_one(model, classes, pil, topk=topk, img_size=img_size)
                        final_label = apply_unknown(pred, conf, threshold)
                        rows.append((fp.name, str(fp.parent.name), final_label, conf))
                    except Exception as e:
                        rows.append((fp.name, "", f"ERROR: {e}", 0.0))
                    prog.progress(idx / len(imgs), text=f"{idx}/{len(imgs)} ảnh")

                st.success(f"Đã xử lý {len(rows)} ảnh.")
                import pandas as pd
                df = pd.DataFrame(rows, columns=["filename", "folder", "pred_or_unknown", "confidence"])
                st.dataframe(df, use_container_width=True)
                st.download_button(
                    "Tải kết quả CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )

st.caption(
    "Tip: Nếu muốn phát hiện 'không phải trái cây' tốt hơn, hãy thêm dữ liệu lớp 'non-fruit' và huấn luyện lại (open-set). "
    "Hiện tại dùng ngưỡng xác suất để gán 'unknown'."
)

