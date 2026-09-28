import streamlit as st
import hashlib
import io
from PIL import Image
import numpy as np
import cv2
import sys
import torch
import torchvision.transforms.functional as torchvision_functional

sys.modules.setdefault(
    "torchvision.transforms.functional_tensor",
    torchvision_functional,
)

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


def sharpen_image(image, amount):
    if amount == 0:
        return image

    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    luminance, chroma_a, chroma_b = cv2.split(lab_image)
    blurred = cv2.GaussianBlur(luminance, (0, 0), sigmaX=1.0)
    sharpened_luminance = cv2.addWeighted(
        luminance, 1.0 + amount, blurred, -amount, 0
    )
    sharpened_lab = cv2.merge((sharpened_luminance, chroma_a, chroma_b))
    return cv2.cvtColor(sharpened_lab, cv2.COLOR_LAB2RGB)


# -----------------------------
# Streamlit page configuration
# -----------------------------
st.set_page_config(
    page_title="Image Super Resolution",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Image Super Resolution")
st.write(
    "Upload a low-resolution image and enhance it using "
    "the pretrained Real-ESRGAN model."
)


# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    network = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )

    model = RealESRGANer(
        scale=4,
        model_path=(
            "https://github.com/xinntao/Real-ESRGAN/releases/download/"
            "v0.1.0/RealESRGAN_x4plus.pth"
        ),
        model=network,
        pre_pad=0,
        device=device,
        half=device.type == "cuda",
    )

    return model


# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a low-resolution image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    uploaded_bytes = uploaded_file.getvalue()
    upload_hash = hashlib.sha256(uploaded_bytes).hexdigest()
    input_sharpness = st.slider(
        "Input deblur",
        min_value=0.0,
        max_value=2.5,
        value=1.0,
        step=0.1,
        help="Sharpen edges before AI enhancement. Strong values can add halos or noise.",
    )
    enhancement_key = (upload_hash, input_sharpness)
    if st.session_state.get("enhancement_key") != enhancement_key:
        st.session_state.pop("enhanced_image", None)

    # Read image
    input_image = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")

    st.subheader("Input Image")

    st.image(
        input_image,
        caption="Low-Resolution Image",
        width="content",
    )

    output_sharpness = st.slider(
        "Output sharpness",
        min_value=0.0,
        max_value=3.0,
        value=1.8,
        step=0.1,
        help="Adjust final edge definition without rerunning AI enhancement.",
    )

    # -------------------------
    # Enhance button
    # -------------------------
    if st.button("Enhance Image"):

        with st.spinner("Enhancing image..."):

            try:
                # Load pretrained model
                model = load_model()

                # Convert PIL image to NumPy
                image_array = sharpen_image(
                    np.array(input_image), input_sharpness
                )

                # Super-resolution
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                output_bgr, _ = model.enhance(image_bgr)
                st.session_state["enhanced_image"] = cv2.cvtColor(
                    output_bgr, cv2.COLOR_BGR2RGB
                )
                st.session_state["enhancement_key"] = enhancement_key

            except Exception as e:
                st.error(f"Error while processing the image: {e}")

    if (
        st.session_state.get("enhancement_key") == enhancement_key
        and "enhanced_image" in st.session_state
    ):
        output_array = sharpen_image(
            st.session_state["enhanced_image"], output_sharpness
        )
        output_image = Image.fromarray(output_array)

        st.success("Image enhancement completed!")
        st.subheader("High-Resolution Image")
        st.image(
            output_image,
            caption="Enhanced Image",
            width="content",
        )

        image_bytes = io.BytesIO()
        output_image.save(image_bytes, format="PNG")
        st.download_button(
            label="Download High-Resolution Image",
            data=image_bytes.getvalue(),
            file_name="enhanced_image.png",
            mime="image/png",
        )