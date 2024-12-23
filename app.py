import streamlit as st
from PIL import Image
from transformers import pipeline

# Streamlit page configuration
st.set_page_config(
    page_title="Image Alt Text Generator",
    page_icon="🖼️",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the image captioning pipeline"""
    try:
        caption_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        return caption_pipeline
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    st.title("🖼️ Image Alt Text Generator")
    st.markdown("""
        Generate descriptive alt text for your images using AI. 
        Upload an image to get started.
    """)
    
    # Load model
    with st.spinner('Loading AI model...'):
        caption_model = load_model()
    
    if caption_model is None:
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
                result = caption_model(image)
                alt_text = result[0]['generated_text']
            
            # Display results
            st.markdown("### Generated Alt Text:")
            st.write(alt_text)
            
            # Show HTML code
            st.markdown("### HTML Code:")
            html_code = f'<img src="{uploaded_file.name}" alt="{alt_text}" />'
            st.code(html_code, language='html')
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
