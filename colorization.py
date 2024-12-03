# Import statements
import numpy as np
import cv2
import os
import streamlit as st
from PIL import Image

# Paths to the model files
PROTOTXT = "model/colorization_deploy_v2.prototxt"
POINTS = "model/pts_in_hull.npy"
MODEL = "model/colorization_release_v2.caffemodel"

# Load the Model
@st.cache_resource
def load_colorization_model():
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    pts = np.load(POINTS)

    # Load centers for ab channel quantization used for rebalancing.
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    return net

# Function to preprocess and colorize the image
def colorize_image(net, image_file):
    try:
        image = Image.open(image_file)
        image = np.array(image.convert("RGB"))  # Ensure the image is in RGB mode
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to OpenCV format

        scaled = image.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50

        # Perform colorization
        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

        ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
        L = cv2.split(lab)[0]
        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = np.clip(colorized, 0, 1)
        colorized = (255 * colorized).astype("uint8")
        
        return colorized
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        return None

# Streamlit app
st.title("Image Colorization")
st.markdown("Upload a black-and-white image to colorize it.")

# Upload the image
uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    # Load the model
    net = load_colorization_model()

    if st.button("Colorize Image"):
        with st.spinner("Colorizing..."):
            colorized_image = colorize_image(net, uploaded_image)

        if colorized_image is not None:
            # Convert the colorized image to PIL format for display
            colorized_pil = Image.fromarray(cv2.cvtColor(colorized_image, cv2.COLOR_BGR2RGB))
            st.image(colorized_pil, caption="Colorized Image", use_container_width=True)

            # Provide option to download the colorized image
            colorized_pil.save("colorized_image.png")
            with open("colorized_image.png", "rb") as file:
                btn = st.download_button(
                    label="Download Colorized Image",
                    data=file,
                    file_name="colorized_image.png",
                    mime="image/png"
                )