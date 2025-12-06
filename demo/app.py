"""
Streamlit demo app for sketch generation models.

This app provides an interactive interface for:
- Uploading images and generating sketches
- Comparing different model outputs
- Adjusting generation parameters
- Downloading generated sketches
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import create_model
from src.utils import load_checkpoint


# Page configuration
st.set_page_config(
    page_title="Sketch Generation Demo",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .model-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f9f9f9;
    }
    .stButton > button {
        width: 100%;
        background-color: #667eea;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #764ba2;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🎨 Sketch Generation Demo</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["pix2pix", "cyclegan", "edge_detection"],
    help="Choose the type of sketch generation model"
)

# Device selection
device_options = ["auto", "cpu"]
if torch.cuda.is_available():
    device_options.append("cuda")
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device_options.append("mps")

device = st.sidebar.selectbox("Device", device_options)

# Image size
image_size = st.sidebar.slider("Image Size", 128, 512, 256, step=32)

# Model parameters
st.sidebar.subheader("Model Parameters")

if model_type == "pix2pix":
    lambda_l1 = st.sidebar.slider("L1 Loss Weight", 10.0, 200.0, 100.0, step=10.0)
elif model_type == "cyclegan":
    lambda_cycle = st.sidebar.slider("Cycle Loss Weight", 5.0, 20.0, 10.0, step=1.0)
    lambda_identity = st.sidebar.slider("Identity Loss Weight", 1.0, 10.0, 5.0, step=0.5)

# Load model function
@st.cache_resource
def load_model(model_type: str, device: str):
    """Load the selected model."""
    try:
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        device = torch.device(device)
        
        if model_type == "edge_detection":
            # Edge detection doesn't need a checkpoint
            model = create_model("edge_detection")
            model.to(device)
            model.eval()
            return model, device
        
        # For other models, try to load from checkpoints
        checkpoint_dir = Path("checkpoints")
        if checkpoint_dir.exists():
            if model_type == "pix2pix":
                checkpoint_path = checkpoint_dir / "best_generator.pth"
            elif model_type == "cyclegan":
                checkpoint_path = checkpoint_dir / "best_g_image_to_sketch.pth"
            
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model = create_model(f"{model_type}_generator")
                
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                model.to(device)
                model.eval()
                return model, device
        
        # Fallback: create untrained model
        st.warning(f"No trained {model_type} model found. Using untrained model.")
        model = create_model(f"{model_type}_generator")
        model.to(device)
        model.eval()
        return model, device
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


# Preprocess image function
def preprocess_image(image: Image.Image, image_size: tuple) -> torch.Tensor:
    """Preprocess image for model input."""
    # Resize image
    image = image.resize(image_size, Image.LANCZOS)
    
    # Convert to tensor
    image_array = np.array(image)
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
    
    # Normalize to [-1, 1]
    image_tensor = image_tensor * 2.0 - 1.0
    
    return image_tensor


# Postprocess sketch function
def postprocess_sketch(sketch_tensor: torch.Tensor) -> np.ndarray:
    """Postprocess sketch tensor to image array."""
    # Denormalize from [-1, 1] to [0, 1]
    sketch_tensor = (sketch_tensor + 1) / 2
    sketch_tensor = torch.clamp(sketch_tensor, 0, 1)
    
    # Convert to numpy
    if sketch_tensor.dim() == 4:  # Batch dimension
        sketch_tensor = sketch_tensor.squeeze(0)
    
    if sketch_tensor.dim() == 3 and sketch_tensor.size(0) == 1:  # Single channel
        sketch_tensor = sketch_tensor.squeeze(0)
    
    sketch_array = sketch_tensor.cpu().numpy()
    
    # Convert to uint8
    sketch_array = (sketch_array * 255).astype(np.uint8)
    
    return sketch_array


# Generate sketch function
def generate_sketch(model: nn.Module, image_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """Generate sketch from image tensor."""
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        sketch_tensor = model(image_tensor)
        sketch_array = postprocess_sketch(sketch_tensor)
        return sketch_array


# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Upload Image")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image to convert to sketch"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Generate sketch button
        if st.button("🎨 Generate Sketch", type="primary"):
            with st.spinner("Generating sketch..."):
                # Load model
                model, device = load_model(model_type, device)
                
                if model is not None:
                    # Preprocess image
                    image_tensor = preprocess_image(image, (image_size, image_size))
                    
                    # Generate sketch
                    sketch_array = generate_sketch(model, image_tensor, device)
                    
                    # Display sketch
                    st.success("Sketch generated successfully!")
                    
                    with col2:
                        st.subheader("🎨 Generated Sketch")
                        st.image(sketch_array, caption="Generated Sketch", use_column_width=True)
                        
                        # Download button
                        sketch_image = Image.fromarray(sketch_array)
                        buf = io.BytesIO()
                        sketch_image.save(buf, format="PNG")
                        byte_im = buf.getvalue()
                        
                        st.download_button(
                            label="📥 Download Sketch",
                            data=byte_im,
                            file_name=f"sketch_{uploaded_file.name}",
                            mime="image/png"
                        )
                else:
                    st.error("Failed to load model")

with col2:
    if uploaded_file is None:
        st.subheader("🎨 Generated Sketch")
        st.info("Upload an image to generate a sketch")
        
        # Show example
        st.subheader("📋 Example")
        st.markdown("""
        **How to use:**
        1. Upload an image using the file uploader
        2. Select model type and parameters in the sidebar
        3. Click "Generate Sketch" to create a sketch
        4. Download the generated sketch
        
        **Model Types:**
        - **Pix2Pix**: Paired image-to-sketch translation
        - **CycleGAN**: Unpaired image-to-sketch translation  
        - **Edge Detection**: Simple edge detection baseline
        """)

# Model information
st.subheader("🤖 Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="model-card">
        <h4>Pix2Pix</h4>
        <p><strong>Type:</strong> Paired Image Translation</p>
        <p><strong>Architecture:</strong> U-Net Generator + PatchGAN Discriminator</p>
        <p><strong>Best for:</strong> High-quality paired image-to-sketch conversion</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="model-card">
        <h4>CycleGAN</h4>
        <p><strong>Type:</strong> Unpaired Image Translation</p>
        <p><strong>Architecture:</strong> ResNet Generator + PatchGAN Discriminator</p>
        <p><strong>Best for:</strong> Style transfer without paired data</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="model-card">
        <h4>Edge Detection</h4>
        <p><strong>Type:</strong> Traditional Computer Vision</p>
        <p><strong>Architecture:</strong> Sobel Edge Detection</p>
        <p><strong>Best for:</strong> Fast edge extraction baseline</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Sketch Generation Demo - Powered by PyTorch</p>
    <p>Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
