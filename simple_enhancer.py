import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import io

def load_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb, img

def resize_image(img, scale=2):
    height, width = img.shape[:2]
    img_resized = cv2.resize(img, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
    return img_resized

def selective_sharpening(img):
    # Apply a bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    # Enhance edges using the Laplacian operator
    laplacian = cv2.Laplacian(filtered, cv2.CV_64F)
    laplacian = np.clip(laplacian, 0, 255).astype('uint8')
    sharpened = cv2.subtract(filtered, laplacian)
    return sharpened

def enhance_colors(img):
    img_pil = Image.fromarray(img)
    enhancer_contrast = ImageEnhance.Contrast(img_pil)
    enhancer_color = ImageEnhance.Color(img_pil)
    img_pil = enhancer_contrast.enhance(1.2)
    img_pil = enhancer_color.enhance(1.2)
    img_final = np.array(img_pil)
    return img_final

def enhance_image(img):
    img_resized = resize_image(img, scale=2)
    img_sharpened = selective_sharpening(img_resized)
    img_color_enhanced = enhance_colors(img_sharpened)
    return img_color_enhanced

def main():
    st.title('Simple Image Quality Enhancer')
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image_rgb, image_bgr = load_image(uploaded_file)
        st.image(image_rgb, caption="Original Image", use_column_width=True)

        if st.button("Enhance Image Quality"):
            enhanced_image = enhance_image(image_rgb)
            st.image(enhanced_image, use_column_width=True, caption="Enhanced Image")
            _, buffer = cv2.imencode('.png', cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR))
            st.download_button("Download Enhanced Image", buffer.tobytes(), "enhanced_image.png", "image/png")

if __name__ == "__main__":
    main()
