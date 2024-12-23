import streamlit as st
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

st.set_page_config(
    page_title="Image Alt Text Generator",
    page_icon="🖼️",
)

@st.cache_resource
def load_model():
    """Load the model with caching"""
    try:
        processor = AutoProcessor.from_pretrained("microsoft/git-base")
        model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")
        return processor, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def generate_alt_text(image, processor, model):
    """Generate alt text for an image"""
    try:
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values

        generated_ids = model.generate(
            pixel_values=pixel_values,
            max_length=50,
            num_beams=4,
        )
        
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_caption
    except Exception as e:
        st.error(f"Error generating alt text: {str(e)}")
        return None

def main():
    st.title("🖼️ Image Alt Text Generator")
    st.write("Upload an image to generate descriptive alt text.")
    
    # Initialize session state for model loading status
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    # Load model
    if not st.session_state.model_loaded:
        with st.spinner('Loading AI model (this may take a minute)...'):
            processor, model = load_model()
            if processor is not None and model is not None:
                st.session_state.model_loaded = True
                st.session_state.processor = processor
                st.session_state.model = model
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file and st.session_state.model_loaded:
        try:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Generate alt text
            with st.spinner('Generating alt text...'):
                alt_text = generate_alt_text(
                    image,
                    st.session_state.processor,
                    st.session_state.model
                )
            
            if alt_text:
                st.markdown("### Generated Alt Text:")
                st.write(alt_text)
                
                st.markdown("### HTML Code:")
                html_code = f'<img src="{uploaded_file.name}" alt="{alt_text}" />'
                st.code(html_code, language='html')
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.error("Please try uploading a different image.")

if __name__ == "__main__":
    main()
