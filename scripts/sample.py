#!/usr/bin/env python3
"""
Sampling script for sketch generation models.

This script provides utilities for generating sketches from images
using trained models.
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import create_model
from src.utils import load_checkpoint
from src.configs import load_config, get_model_config


def load_trained_model(model_path: str, model_type: str, device: torch.device) -> nn.Module:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model
    if model_type == 'pix2pix':
        model = create_model('pix2pix_generator')
    elif model_type == 'cyclegan':
        model = create_model('cyclegan_generator')
    elif model_type == 'edge_detection':
        model = create_model('edge_detection')
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def preprocess_image(image_path: str, image_size: tuple = (256, 256)) -> torch.Tensor:
    """Preprocess an image for model input."""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Resize
    image = image.resize(image_size, Image.LANCZOS)
    
    # Convert to tensor
    image_array = np.array(image)
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
    
    # Normalize to [-1, 1]
    image_tensor = image_tensor * 2.0 - 1.0
    
    return image_tensor


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


def generate_sketch_from_image(
    model: nn.Module,
    image_path: str,
    device: torch.device,
    image_size: tuple = (256, 256)
) -> np.ndarray:
    """Generate sketch from a single image."""
    # Preprocess image
    image_tensor = preprocess_image(image_path, image_size)
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
    
    # Generate sketch
    with torch.no_grad():
        sketch_tensor = model(image_tensor)
    
    # Postprocess sketch
    sketch_array = postprocess_sketch(sketch_tensor)
    
    return sketch_array


def generate_sketches_from_directory(
    model: nn.Module,
    input_dir: str,
    output_dir: str,
    device: torch.device,
    image_size: tuple = (256, 256)
) -> None:
    """Generate sketches for all images in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for image_file in tqdm(image_files, desc="Generating sketches"):
        try:
            # Generate sketch
            sketch_array = generate_sketch_from_image(
                model, str(image_file), device, image_size
            )
            
            # Save sketch
            output_file = output_path / f"{image_file.stem}_sketch.png"
            Image.fromarray(sketch_array).save(output_file)
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")


def create_comparison_grid(
    image_paths: list,
    model: nn.Module,
    device: torch.device,
    image_size: tuple = (256, 256),
    num_images: int = 8
) -> plt.Figure:
    """Create a comparison grid of original images and generated sketches."""
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    
    for i in range(min(num_images, len(image_paths))):
        image_path = image_paths[i]
        
        # Load original image
        original_image = Image.open(image_path).convert('RGB')
        original_image = original_image.resize(image_size, Image.LANCZOS)
        
        # Generate sketch
        sketch_array = generate_sketch_from_image(model, image_path, device, image_size)
        
        # Display original image
        axes[0, i].imshow(original_image)
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Display generated sketch
        axes[1, i].imshow(sketch_array, cmap='gray')
        axes[1, i].set_title(f'Sketch {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    return fig


def main():
    """Main sampling function."""
    parser = argparse.ArgumentParser(description='Generate sketches from images')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--model-type', type=str, required=True, 
                       choices=['pix2pix', 'cyclegan', 'edge_detection'],
                       help='Type of model')
    parser.add_argument('--input', type=str, required=True, 
                       help='Input image or directory')
    parser.add_argument('--output', type=str, help='Output path or directory')
    parser.add_argument('--image-size', type=int, nargs=2, default=[256, 256],
                       help='Image size (width height)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device to use')
    parser.add_argument('--num-samples', type=int, default=8,
                       help='Number of samples for comparison grid')
    parser.add_argument('--save-comparison', action='store_true',
                       help='Save comparison grid')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = load_trained_model(args.model_path, args.model_type, device)
    print("Model loaded successfully")
    
    # Setup paths
    input_path = Path(args.input)
    image_size = tuple(args.image_size)
    
    if input_path.is_file():
        # Single image
        print(f"Processing single image: {input_path}")
        
        sketch_array = generate_sketch_from_image(
            model, str(input_path), device, image_size
        )
        
        # Save sketch
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path = input_path.parent / f"{input_path.stem}_sketch.png"
        
        Image.fromarray(sketch_array).save(output_path)
        print(f"Sketch saved to: {output_path}")
        
    elif input_path.is_dir():
        # Directory of images
        print(f"Processing directory: {input_path}")
        
        if args.output:
            output_dir = args.output
        else:
            output_dir = str(input_path.parent / f"{input_path.name}_sketches")
        
        generate_sketches_from_directory(
            model, str(input_path), output_dir, device, image_size
        )
        print(f"Sketches saved to: {output_dir}")
        
        # Create comparison grid if requested
        if args.save_comparison:
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_files = [f for f in input_path.iterdir() 
                          if f.suffix.lower() in image_extensions]
            
            if image_files:
                fig = create_comparison_grid(
                    [str(f) for f in image_files[:args.num_samples]],
                    model, device, image_size, args.num_samples
                )
                
                comparison_path = Path(output_dir) / "comparison_grid.png"
                fig.savefig(comparison_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"Comparison grid saved to: {comparison_path}")
    
    else:
        print(f"Invalid input path: {input_path}")
        return
    
    print("Sketch generation completed!")


if __name__ == '__main__':
    main()
