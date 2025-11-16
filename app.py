# app.py — Streamlit Fruit Classifier (upload ảnh an toàn & debug rõ ràng)
import json
import inspect
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

# ================= Cấu hình =================
st.set_page_config(page_title="Fruit Classifier", page_icon="🍎", layout="centered")

BASE = Path.cwd()  # đúng như bạn muốn
MODEL_PATH = BASE / "outputs_multi" / "fruit_model.keras"
CLASSMAP_PATH = BASE / "outputs_multi" / "class_indices.json"
IMG_SIZE = (224, 224)
ABSTAIN_THRESHOLD = 0.60  # nếu max-prob < ngưỡng, coi là "không phải trái cây"

# ============== Tiện ích hiển thị ảnh (tương thích mọi bản Streamlit) ==============
def show_image(img, caption=None):
    params = inspect.signature(st.image).parameters
    if "use_container_width" in params:
        st.image(img, caption=caption, use_container_width=True)
    else:
        st.image(img, caption=caption, use_column_width=True)

# ============== Tải model & class map (cache) ==============
@st.cache_resource(show_spinner=False)
def load_model():
    # compile=False để tránh yêu cầu khớp optimizer/metrics khi load
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def _normalize_classes(obj):
    # hỗ trợ list hoặc dict {"0":"apple"} hoặc {"apple":0}
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # kiểu {"0":"apple"}
        if all(str(k).isdigit() for k in obj.keys()):
            return [obj[str(i)] for i in range(len(obj))]
        # kiểu {"apple": 0}
        ordered = sorted(((idx, name) for name, idx in obj.items()), key=lambda x: x[0])
        return [name for _, name in ordered]
    return [str(x) for x in obj]

@st.cache_resource(show_spinner=False)
def load_classes():
    with open(CLASSMAP_PATH, "r", encoding="utf-8") as f:
        mp = json.load(f)
    return _normalize_classes(mp)

# ============== Đọc ảnh upload an toàn ==============
def read_uploaded_image(uploaded_file) -> Image.Image:
    """
    Đọc st.uploaded_file an toàn:
      - Đọc bytes -> BytesIO (không phụ thuộc vị trí con trỏ)
      - Mở bằng PIL, convert RGB
    """
    data = uploaded_file.read()
    if not data:
        # có thể con trỏ đang ở cuối file do xem trước -> reset rồi đọc lại
        uploaded_file.seek(0)
        data = uploaded_file.read()
    bio = BytesIO(data)
    img = Image.open(bio)
    return img.convert("RGB")

# ============== Suy luận ==============
def predict_pil(pil_img: Image.Image, classes):
    # resize
    img_resized = pil_img.resize(IMG_SIZE, Image.BICUBIC)
    x = np.asarray(img_resized, dtype=np.float32)[None, ...] / 255.0
    logits = model.predict(x, verbose=0)
    probs = tf.nn.softmax(logits, axis=1).numpy()[0]
    idx = int(np.argmax(probs))
    return img_resized, probs, idx, float(probs[idx])

# ============== Khởi tạo model/lớp ==============
try:
    model = load_model()
    classes = load_classes()
except Exception as e:
    st.error(f"Không thể load model/class map: {e}")
    with st.expander("Debug paths"):
        st.write("MODEL_PATH:", str(MODEL_PATH))
        st.write("CLASSMAP_PATH:", str(CLASSMAP_PATH))
        st.write("Tồn tại model?", MODEL_PATH.exists())
        st.write("Tồn tại class map?", CLASSMAP_PATH.exists())
    st.stop()

# ============== UI ==============
st.title("🍎🍌🍊 Fruit Classifier Demo")
st.caption("Upload ảnh để mô hình dự đoán loại trái cây (MobileNetV2 fine-tune).")

# Debug panel
with st.expander("🔧 Debug môi trường"):
    import platform, PIL
    st.write("Python:", platform.python_version())
    st.write("Streamlit:", st.__version__)
    st.write("TensorFlow:", tf.__version__)
    st.write("Pillow:", PIL.__version__)
    st.write("Classes:", classes)
    st.write("MODEL_PATH tồn tại:", MODEL_PATH.exists())
    st.write("CLASSMAP_PATH tồn tại:", CLASSMAP_PATH.exists())

files = st.file_uploader(
    "Chọn 1 hoặc nhiều ảnh (jpg/png/webp/bmp)",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True
)

if files:
    for uf in files:
        try:
            pil = read_uploaded_image(uf)          # ⇐ cách đọc an toàn
            img_resized, probs, idx, p = predict_pil(pil, classes)

            # hiển thị ảnh bằng hàm tương thích
            show_image(img_resized, caption=getattr(uf, "name", "uploaded"))

            if p < ABSTAIN_THRESHOLD:
                st.markdown(
                    f"**Kết luận:** Không chắc là trái cây (max prob {p*100:.1f}% < {ABSTAIN_THRESHOLD*100:.0f}%)."
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
