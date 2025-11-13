import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="Hand Fracture Detector",
    page_icon="🩻",
    layout="wide"
)

# -------------------------------
# CUSTOM CSS STYLING
# -------------------------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
    }
    .main {
        background-color: rgba(255,255,255,0.05);
        border-radius: 15px;
        padding: 20px;
    }
    .stApp {
        background: linear-gradient(to bottom right, #141E30, #243B55);
    }
    h1, h2, h3 {
        color: #00BFFF !important;
        text-align: center;
        font-family: 'Poppins', sans-serif;
    }
    .stButton>button {
        background-color: #00BFFF;
        color: white;
        border-radius: 8px;
        font-size: 18px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #007ACC;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# APP HEADER
# -------------------------------
st.title(" AI-Powered Hand Fracture Detection")
st.markdown(
    "<p style='text-align:center;'>Upload an X-ray image of a hand, and our YOLO model will highlight fracture areas.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # Your trained YOLO model
    return model

model = load_model()

# Fixed confidence threshold
CONFIDENCE_THRESHOLD = 0.25  # Default 25%

# -------------------------------
# IMAGE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    # Show uploaded image
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption=" Uploaded X-ray", use_container_width=True)

    # Predict with model
    with st.spinner("🔍 Detecting fractures... Please wait."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            temp_path = tmp.name

        results = model.predict(temp_path, conf=CONFIDENCE_THRESHOLD)
        annotated_img = results[0].plot()

        # Show result image
        with col2:
            st.image(annotated_img, caption="✅ Detection Result", use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Detection Summary")

        if len(results[0].boxes) == 0:
            st.warning("No fractures detected in this image.")
        else:
            for box in results[0].boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                xyxy = box.xyxy.tolist()[0]
                st.success(f"**Detected:** {model.names[cls_id]} | Confidence: {conf:.2f}")
                st.caption(f"📍 Coordinates: {xyxy}")

        # Cleanup temp file
        os.remove(temp_path)

else:
    st.info("📸 Please upload an X-ray image to start detection.")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
# st.markdown(
#     "<p style='text-align:center; color: #aaa;'>🚀 Powered by YOLOv8 & Streamlit | Made with ❤️ by Laiba & Usama</p>",
#     unsafe_allow_html=True,
# )
