
import streamlit as st
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import io
import base64
from pathlib import Path
import time

class AltTextGenerator:
    def __init__(self):
        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_alt_text(self, image, max_length=30, num_beams=4):
        if isinstance(image, str):  # If image is a file path
            image = Image.open(image)
        
        # Prepare image for model
        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        # Generate caption
        output_ids = self.model.generate(
            pixel_values,
            num_beams=num_beams,
            max_length=max_length,
            num_return_sequences=1
        )

        # Decode caption
        caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption

def main():
    st.set_page_config(
        page_title="Advanced Alt Text Generator",
        page_icon="🖼️",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .upload-box {
            border: 2px dashed #4CAF50;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }
        .success-msg {
            color: #4CAF50;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'generator' not in st.session_state:
        with st.spinner('Loading AI model... Please wait.'):
            st.session_state.generator = AltTextGenerator()

    # Header
    st.title("🖼️ Advanced Image Alt Text Generator")
    st.markdown("""
        Generate accurate and descriptive alt text for your images using advanced AI. 
        This tool helps improve web accessibility and SEO.
    """)

    # Sidebar configuration
    st.sidebar.header("Configuration")
    max_length = st.sidebar.slider("Maximum Text Length", 10, 100, 30)
    num_beams = st.sidebar.slider("Beam Search Size", 1, 10, 4)
    
    # File uploader
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop your images here or click to upload",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                with st.spinner('Analyzing image...'):
                    # Convert uploaded file to PIL Image
                    image = Image.open(uploaded_file)
                    
                    # Generate alt text
                    alt_text = st.session_state.generator.generate_alt_text(
                        image,
                        max_length=max_length,
                        num_beams=num_beams
                    )
                    
                    st.markdown("### Generated Alt Text:")
                    st.write(alt_text)
                    
                    # Copy button
                    if st.button(f"Copy Alt Text", key=f"copy_{uploaded_file.name}"):
                        st.write('<p class="success-msg">✓ Copied to clipboard!</p>', unsafe_allow_html=True)
                    
                    # Download as HTML
                    html_code = f'<img src="{uploaded_file.name}" alt="{alt_text}" />'
                    st.download_button(
                        label="Download HTML",
                        data=html_code,
                        file_name=f"{Path(uploaded_file.name).stem}_with_alt.html",
                        mime="text/html"
                    )

    # Usage tips
    with st.expander("📚 Usage Tips"):
        st.markdown("""
            1. **Upload Multiple Images**: You can upload multiple images at once for batch processing.
            2. **Adjust Settings**: Use the sidebar to customize the generation parameters:
                - Maximum Text Length: Controls the length of generated descriptions
                - Beam Search Size: Higher values generate more diverse descriptions
            3. **Copy & Download**: Each generated alt text can be copied or downloaded as HTML.
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        "Created with ❤️ using Streamlit and Hugging Face Transformers. "
        "Contribute on [GitHub](https://github.com/yourusername/alt-text-generator)"
    )

if __name__ == "__main__":
    main()
