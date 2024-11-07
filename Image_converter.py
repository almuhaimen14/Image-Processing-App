import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io

def load_image(uploaded_file):
    """Load an uploaded image file as a NumPy array in RGB or RGBA format."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if img.shape[-1] == 4:  # RGBA image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    else:  # RGB image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb, img

def convert_image_format(img, output_format, dpi=1200):
    """Convert image to specified format with maximal quality settings."""
    img_pil = Image.fromarray(img)
    buffer = io.BytesIO()

    if output_format.upper() in ['JPG', 'JPEG']:
        if img_pil.mode == "RGBA":
            background = Image.new("RGB", img_pil.size, (255, 255, 255))
            background.paste(img_pil, mask=img_pil.split()[3])  # Alpha channel as mask
            img_pil = background
        img_pil.save(buffer, format="JPEG", quality=100, optimize=True)

    elif output_format.upper() == 'PNG':
        if img_pil.mode != "RGBA":
            img_pil = img_pil.convert("RGBA")
        img_pil.save(buffer, format="PNG", compress_level=0)

    elif output_format.upper() == 'PDF':
        if img_pil.mode == "RGBA":
            background = Image.new("RGB", img_pil.size, (255, 255, 255))
            background.paste(img_pil, mask=img_pil.split()[3])
            img_pil = background
        # Use maximum DPI for PDF and convert to higher quality
        img_pil = img_pil.convert("RGB")
        img_pil.save(buffer, format="PDF", resolution=dpi)

    buffer.seek(0)
    return buffer

def main():
    st.title('Image Converter App')

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_rgb, image_bgr = load_image(uploaded_file)
        original_height, original_width = image_rgb.shape[:2]
        st.image(image_rgb, caption=f'Uploaded Image (Dimensions: {original_width}x{original_height} pixels)', use_column_width=True)

        action = st.selectbox("Select Action", ["Convert Format"])

        if action == "Convert Format":
            format_choice = st.selectbox("Select Output Format", ["JPEG", "PNG", "PDF", "JPG"])
            if st.button(f"Convert to {format_choice}"):
                # Set high DPI for PDF output to improve quality
                buffer = convert_image_format(image_rgb, format_choice, dpi=1200)
                st.download_button("Download Converted Image", buffer, f"converted_image.{format_choice.lower()}", f"image/{format_choice.lower()}")

if __name__ == "__main__":
    main()
