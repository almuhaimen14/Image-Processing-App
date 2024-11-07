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

def apply_fourier_sharpening(img, radius=18, blend_ratio=0.2, notch_filter=True):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dft = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    rows, cols = img_gray.shape
    crow, ccol = rows // 2, cols // 2
    
    if notch_filter:
        n = 10
        mask = np.ones((rows, cols, 2), np.uint8)
        mask[crow-n:crow+n, ccol-n:ccol+n] = 0
    else:
        mask = np.zeros((rows, cols, 2), np.uint8)
        cv2.circle(mask, (ccol, crow), radius, (1, 1), -1)
    
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)
    img_enhanced = cv2.addWeighted(img, 1 - blend_ratio, cv2.cvtColor(img_back, cv2.COLOR_GRAY2RGB), blend_ratio, 0)
    return img_enhanced

def apply_high_frequency_boost(img, strength=1.0):
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    high_freq_detail = cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)
    return high_freq_detail

def apply_unsharp_mask(img, strength=1.0, blur_radius=0.2):
    blurred = cv2.GaussianBlur(img, (0, 0), blur_radius)
    img_sharpened = cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)
    return img_sharpened

def enhance_image(img):
    img_resized = resize_image(img, scale=2)
    img_fourier_sharpened = apply_fourier_sharpening(img_resized, radius=18, blend_ratio=0.2, notch_filter=True)

    img_denoised = cv2.fastNlMeansDenoisingColored(img_fourier_sharpened, None, h=3, templateWindowSize=7, searchWindowSize=21)
    img_detail_enhanced = apply_high_frequency_boost(img_denoised, strength=1.0)

    img_pil = Image.fromarray(img_detail_enhanced)
    enhancer_contrast = ImageEnhance.Contrast(img_pil)
    enhancer_saturation = ImageEnhance.Color(img_pil)
    img_pil = enhancer_contrast.enhance(1.3)  # Moderately increase contrast
    img_pil = enhancer_saturation.enhance(1.1)  # Slightly boost saturation

    img_final = np.array(img_pil)
    img_crisp_edges = apply_unsharp_mask(img_final, strength=1.0, blur_radius=0.2)

    return img_crisp_edges

def main():
    st.title('Complex Image Quality Enhancer')
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image_rgb, image_bgr = load_image(uploaded_file)
        st.image(image_rgb, caption="Original Image", use_column_width=True)

        if st.button("Complex Image Quality Enhancer"):
            enhanced_image = enhance_image(image_rgb)
            st.image(enhanced_image, use_column_width=True, caption="Enhanced Image")
            _, buffer = cv2.imencode('.png', cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR))
            st.download_button("Download Enhanced Image", buffer.tobytes(), "enhanced_image.png", "image/png")

if __name__ == "__main__":
    main()
