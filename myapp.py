import streamlit as st
import complex_enhancer
import simple_enhancer
import Image_converter
import compressor

# Set up the main title for the app
st.title('Image Processing App')

# Sidebar for tool selection
st.sidebar.title("Select a Tool")
tool_choice = st.sidebar.radio(
    "Choose a processing tool:",
    ("Complex Enhancer", "Simple Enhancer", "Image Converter", "Image Compressor")
)

# Display the selected tool by calling the main function of each module
if tool_choice == "Complex Enhancer":
    st.subheader("Complex Enhancer")
    complex_enhancer.main()

elif tool_choice == "Simple Enhancer":
    st.subheader("Simple Enhancer")
    simple_enhancer.main()

elif tool_choice == "Image Converter":
    st.subheader("Image Converter")
    Image_converter.main()

elif tool_choice == "Image Compressor":  # Updated to match "Image Compressor"
    st.subheader("Image Compressor")
    compressor.main()

