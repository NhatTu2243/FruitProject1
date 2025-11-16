# app.py — Streamlit Fruit Classifier (bản tối giản, an toàn upload)
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

# ================= Cấu hình =================
st.set_page_config(page_title="Fruit Classifier", page_icon="🍎", layout="centered")

BASE = Path.cwd()
MODEL_PATH = BASE / "outputs_multi" / "fruit_model.keras"       # GIỮ ĐÚNG THƯ MỤC
CLASSMAP_PATH = BASE / "outputs_multi" / "class_indices.json"   # GIỮ ĐÚNG THƯ MỤC
IMG_SIZE = (224, 224)
ABSTAIN_THRESHOLD = 0.60  # nếu max prob < ngưỡng ⇒ coi là "không phải trái cây"

# ================= Tải model & class map (cache) =================
@st.cache_resource(show_spinner=False)
def load_model():
    # compile=False để tránh lỗi optimizer/metrics khi load
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

@st.cache_resource(show_spinner=False)
def load_classes():
    with open(CLASSMAP_PATH, "r", encoding="utf-8") as f:
        mp = json.load(f)  # hỗn hợp {"0": "apple"} hoặc list
    if isinstance(mp, list):
        return mp
    # dạng {"0": "apple", "1": "banana", ...}
    return [mp[str(i)] for i in range(len(mp))]

# ================= Tiện ích =================
def read_uploaded_image(uploaded_file) -> Image.Image:
    """Đọc ảnh từ st.file_uploader an toàn (bytes -> PIL RGB)."""
    data = uploaded_file.getvalue()  # an toàn hơn read/seek
    if not data:
        raise ValueError("File rỗng hoặc không đọc được bytes.")
    img = Image.open(BytesIO(data))
    return img.convert("RGB")

def predict_pil(pil_img: Image.Image, classes):
    # resize + chuẩn hóa đúng như khi train (đÃ /255.0)
    img_resized = pil_img.resize(IMG_SIZE, Image.BICUBIC)
    x = np.asarray(img_resized, dtype=np.float32)[None, ...] / 255.0
    logits = model.predict(x, verbose=0)
    probs = tf.nn.softmax(logits, axis=1).numpy()[0]
    idx = int(np.argmax(probs))
    return img_resized, probs, idx, float(probs[idx])

# ================= Khởi tạo =================
try:
    model = load_model()
    classes = load_classes()
except Exception as e:
    st.error(f"Không thể load model/class map: {e}")
    st.stop()

# ================= UI =================
st.title("🍎🍌🍊 Fruit Classifier Demo")
st.caption("Upload ảnh để mô hình dự đoán loại trái cây (MobileNetV2 fine-tune).")

with st.expander("🔧 Debug nhanh"):
    import platform, PIL
    st.write("Python:", platform.python_version())
    st.write("Streamlit:", st.__version__)
    st.write("TensorFlow:", tf.__version__)
    st.write("Pillow:", PIL.__version__)
    st.write("MODEL_PATH tồn tại:", MODEL_PATH.exists())
    st.write("CLASSMAP_PATH tồn tại:", CLASSMAP_PATH.exists())
    st.write("Classes:", classes)

files = st.file_uploader(
    "Chọn 1 hoặc nhiều ảnh (jpg/png/webp/bmp)",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True
)

if files:
    for uf in files:
        try:
            pil = read_uploaded_image(uf)
            img_resized, probs, idx, p = predict_pil(pil, classes)

            # CHỈ dùng use_column_width (tương thích mọi bản)
            st.image(img_resized, caption=getattr(uf, "name", "uploaded"), use_column_width=True)

            if p < ABSTAIN_THRESHOLD:
                st.markdown(
                    f"**Kết luận:** Không chắc là trái cây "
                    f"(max prob {p*100:.1f}% < {ABSTAIN_THRESHOLD*100:.0f}%)."
                )
            else:
                st.markdown(f"**Dự đoán:** `{classes[idx]}` — **Độ tự tin:** {p*100:.2f}%")

            # Top-3
            top3 = probs.argsort()[-3:][::-1]
            st.write("**Top-3:**")
            for k in top3:
                st.write(f"- {classes[int(k)]}: {probs[int(k)]*100:.2f}%")

            st.divider()

        except UnidentifiedImageError:
            st.warning(f"❌ `{getattr(uf, 'name', '')}` không phải file ảnh hợp lệ.")
        except Exception as e:
            st.error(f"❌ Lỗi xử lý `{getattr(uf, 'name', '')}`: {e}")
else:
    st.info("Hãy chọn ảnh để bắt đầu.")

# Nút dọn cache khi cần
col1, col2 = st.columns(2)
with col1:
    if st.button("♻️ Xóa cache model/classes"):
        st.cache_resource.clear()
        st.success("Đã xóa cache. Nhấn Rerun để tải lại.")
with col2:
    st.caption(f"Ngưỡng không-phải-trái-cây: {int(ABSTAIN_THRESHOLD*100)}% (chỉnh trong code).")
