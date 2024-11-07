import streamlit as st
from PIL import Image, ImageFilter
import numpy as np
import cv2
import io

def load_image(uploaded_file):
    """Load an uploaded image file as a NumPy array in RGB format."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb, uploaded_file.type

def find_optimal_quality(img, original_size):
    """Find the threshold quality for JPEG compression where file size reduces with maximum quality retention."""
    buffer = io.BytesIO()
    img_pil = Image.fromarray(img)
    quality = 100
    img_pil.save(buffer, format="JPEG", quality=quality, optimize=True, subsampling=1)
    compressed_size = buffer.tell()

    # Iteratively decrease quality to find the point where compression is effective without major quality loss
    while compressed_size > original_size and quality > 30:
        quality -= 5
        buffer = io.BytesIO()
        img_pil.save(buffer, format="JPEG", quality=quality, optimize=True, subsampling=1)
        compressed_size = buffer.tell()

    # Final quality level where size reduction begins to retain maximum quality
    return max(quality, 30)  # Use at least 30 for minimal artifacts

def compress_image_jpeg(img, quality):
    """Compress JPEG image with a specified quality setting."""
    buffer = io.BytesIO()
    img_pil = Image.fromarray(img)
    if quality <= 30:
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(1))  # Mild blur to reduce noise at low quality
    img_pil.save(buffer, format="JPEG", quality=quality, optimize=True, subsampling=1)
    buffer.seek(0)
    return buffer

def compress_image_png(img, compression_level):
    """Compress PNG image without quality loss but considering file size."""
    buffer = io.BytesIO()
    img.convert("RGBA").save(buffer, format="PNG", compress_level=compression_level)
    buffer.seek(0)
    return buffer

def resize_image(img, width, height):
    """Resize image to specified width and height with high-quality resampling."""
    img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    return img_resized

def main():
    st.title('Advanced Image Compressor')
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_rgb, file_type = load_image(uploaded_file)
        original_size = uploaded_file.size
        original_height, original_width = image_rgb.shape[:2]
        st.image(image_rgb, caption=f'Original Image (Dimensions: {original_width}x{original_height} pixels)', use_column_width=True)

        action = st.selectbox("Choose Compression Action", ["Compress by Quality", "Compress by Dimensions"])
        if action == "Compress by Quality":
            format_choice = st.radio("Select Format", ["JPEG", "PNG"])
            if format_choice == "JPEG":
                # Determine optimal quality for this image to set as the baseline
                optimal_quality = find_optimal_quality(image_rgb, original_size)
                st.write("### JPEG Quality Slider")
                st.write("Controls the quality of the JPEG image. Lower values compress more but may reduce quality. Higher values compress less but may result in larger file sizes.")
                st.write("User Warning: Avoid setting too high for small images as it may increase file size.")
                st.write(f"Optimal Quality Level for this image: {optimal_quality}")
                quality = st.slider("Select JPEG Quality (higher = better quality)", 30, 100, optimal_quality)
                
                if st.button("Compress JPEG"):
                    compressed_buffer = compress_image_jpeg(image_rgb, quality)
                    compressed_size = len(compressed_buffer.getvalue())
                    st.image(compressed_buffer, caption='Compressed JPEG Image', use_column_width=True)
                    st.write(f"Selected Quality Level: {quality}")
                    st.write(f"Original Size: {original_size / 1024:.2f} KB, Compressed Size: {compressed_size / 1024:.2f} KB")
                    st.download_button("Download Compressed JPEG", compressed_buffer.getvalue(), "compressed_image.jpg", "image/jpeg")

            elif format_choice == "PNG":
                if format_choice == "PNG" and file_type.lower() not in ['png']:
                    st.warning("Warning: Converting JPEG to PNG might increase file size due to lossless compression.")
                st.write("### PNG Compression Level Slider")
                st.write("Controls the file size reduction for PNG files, which is lossless. Lower numbers mean less compression (larger file size and quick process time), higher numbers mean more compression (smaller file size but slower process time).")
                st.write("User Warning: For files smaller than 50KB, PNG compression may increase the file size instead of reducing it.")
                compression_level = st.slider("Select PNG Compression Level (higher=smaller file)", 1, 9, 6)

                if st.button("Compress PNG"):
                    compressed_buffer = compress_image_png(Image.fromarray(image_rgb), compression_level)
                    compressed_size = len(compressed_buffer.getvalue())
                    st.image(compressed_buffer, caption='Compressed PNG Image', use_column_width=True)
                    st.write(f"Original Size: {original_size / 1024:.2f} KB, Compressed Size: {compressed_size / 1024:.2f} KB")
                    st.download_button("Download Compressed PNG", compressed_buffer.getvalue(), "compressed_image.png", "image/png")

        elif action == "Compress by Dimensions":
            aspect_ratio_lock = st.checkbox("Maintain Aspect Ratio", value=True, help="When checked, adjusting the width will automatically adjust the height to keep the original proportions of the image.")
            width_input = st.number_input("Enter Width (pixels)", min_value=50, max_value=original_width, value=original_width)
            height_input = original_height

            if aspect_ratio_lock:
                height_input = int(width_input * original_height / original_width)
                st.write(f"Suggested Height (preserving aspect ratio): {height_input} pixels")
            else:
                height_input = st.number_input("Enter Height (pixels)", min_value=50, max_value=original_height, value=original_height)

            # Error handling for invalid dimension inputs
            invalid_dimensions = width_input < 50 or height_input < 50 or width_input > original_width or height_input > original_height
            if invalid_dimensions:
                st.warning("Insert proper values to compress the image.")
            else:
                if st.button("Compress Image"):
                    resized_image = resize_image(image_rgb, width_input, height_input)
                    if resized_image is not None:
                        st.image(resized_image, use_column_width=True, caption=f'Resized Image (Dimensions: {width_input}x{height_input} pixels)')
                        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))
                        st.download_button("Download Resized Image", buffer.tobytes(), "resized_image.jpg", "image/jpeg")

if __name__ == "__main__":
    main()
