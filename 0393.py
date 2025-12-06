# Project 393. Sketch generation models - MODERNIZED
# Description:
# Sketch generation is the task of converting a real image into a sketch or line drawing that approximates the content of the original image. This process is useful for artistic applications, image-to-sketch conversion, and style transfer tasks. Generative models like GANs or Autoencoders can be used to learn the mapping from real images to their corresponding sketches. In this project, we will explore how to generate sketches from images using modern PyTorch implementations.

# 🚀 Modern PyTorch Implementation (Sketch Generation):
# This project now includes state-of-the-art implementations of:
# - Pix2Pix: Paired image-to-sketch translation using U-Net generator and PatchGAN discriminator
# - CycleGAN: Unpaired image-to-sketch translation using ResNet generator and cycle consistency
# - Edge Detection: Traditional computer vision baseline using Sobel operators

# Quick Start Guide:

# 1. Install Dependencies:
# pip install -r requirements.txt

# 2. Run Interactive Demo:
# streamlit run demo/app.py

# 3. Train a Model:
# python scripts/train.py --config configs/pix2pix.yaml

# 4. Generate Sketches:
# python scripts/sample.py --model-path checkpoints/best_generator.pth --model-type pix2pix --input image.jpg --output sketch.png

# Example Usage:

import torch
import torch.nn as nn
from src.models import create_model
from src.data import SyntheticSketchDataset
from src.utils.metrics import SketchMetrics

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create a Pix2Pix generator
generator = create_model('pix2pix_generator', in_channels=3, out_channels=1)
generator.to(device)

# Create synthetic dataset for testing
dataset = SyntheticSketchDataset(num_samples=100, image_size=(256, 256))

# Test the model
sample = dataset[0]
image = sample['image'].unsqueeze(0).to(device)
sketch = sample['sketch'].unsqueeze(0).to(device)

# Generate sketch
with torch.no_grad():
    generated_sketch = generator(image)

print(f"Input image shape: {image.shape}")
print(f"Generated sketch shape: {generated_sketch.shape}")
print(f"Real sketch shape: {sketch.shape}")

# Evaluate using metrics
metrics = SketchMetrics(device)
sketch_quality = metrics.calculate_sketch_quality(generated_sketch)
print(f"Sketch quality metrics: {sketch_quality}")

# ✅ What It Does:
# - Generates high-quality sketches from real images using modern deep learning models
# - Supports multiple model architectures (Pix2Pix, CycleGAN, Edge Detection)
# - Provides comprehensive evaluation metrics (FID, LPIPS, edge consistency)
# - Includes interactive web demo for real-time sketch generation
# - Features production-ready code with proper configuration management

# Key Features:
# - Modern PyTorch implementation with type hints and documentation
# - Comprehensive evaluation metrics for model comparison
# - Interactive Streamlit demo for easy testing
# - Support for both paired and unpaired datasets
# - Automatic device detection (CUDA/MPS/CPU)
# - Configurable training and evaluation pipelines
# - Production-ready code structure with tests and CI/CD

# Model Comparison:
# - Pix2Pix: Best for paired image-to-sketch conversion with high quality
# - CycleGAN: Best for unpaired style transfer without paired data
# - Edge Detection: Fast baseline using traditional computer vision

# This implementation provides a complete, modern framework for sketch generation
# that can be used for research, education, and production applications.