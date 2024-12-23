import streamlit as st
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import io
import base64
from pathlib import Path

# Streamlit page configuration
st.set_page_config(
    page_title="Image Alt Text Generator",
    page_icon="🖼️",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the model and tokenizer with caching"""
    try:
        model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        return model, feature_extractor, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def generate_alt_text(image, model, feature_extractor, tokenizer, max_length=30, num_beams=4):
    """Generate alt text for an image"""
    try:
        if isinstance(image, str):
            image = Image.open(image)
        
        # Prepare image
        pixel_values = feature_extractor(image, return_tensors="pt").pixel_values
        
        # Generate caption
        output_ids = model.generate(
            pixel_values,
            num_beams=num_beams,
            max_length=max_length,
            num_return_sequences=1
        )
        
        # Decode caption
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        st.error(f"Error generating alt text: {str(e)}")
        return None

def main():
    st.title("🖼️ Image Alt Text Generator")
    st.markdown("""
        Generate descriptive alt text for your images using AI. 
        Upload an image to get started.
    """)
    
    # Load model
    with st.spinner('Loading AI model...'):
        model, feature_extractor, tokenizer = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please try again later.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file:
        try:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Generate alt text
            with st.spinner('Generating alt text...'):
                alt_text = generate_alt_text(
                    image,
                    model,
                    feature_extractor,
                    tokenizer
                )
            
            if alt_text:
                st.markdown("### Generated Alt Text:")
                st.write(alt_text)
                
                # Copy button
                st.code(f'<img src="{uploaded_file.name}" alt="{alt_text}" />', language='html')
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
