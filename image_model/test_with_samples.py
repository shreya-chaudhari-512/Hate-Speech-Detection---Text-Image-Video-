"""
Test All Models with Your Sample Images
SAVE AS: image_model/test_with_samples.py

Run this after you add some images to verify everything works.

Usage:
    python image_model/test_with_samples.py

This will test ResNet, ViT, and CLIP on your images.
"""

import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path

from models import create_model


def test_models_with_image(image_path):
    """
    Test all 3 models with one image
    
    Args:
        image_path: Path to an image file
    """
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"Testing with: {image_path}")
    print(f"{'='*70}\n")
    
    # Load image
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"✓ Image loaded: {image.size}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return
    
    # Preprocess (resize, normalize)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    print(f"  Preprocessed: {image_tensor.shape}\n")
    
    # Test each model
    models_to_test = ['resnet', 'vit', 'clip']
    
    for model_name in models_to_test:
        print(f"{'-'*70}")
        print(f"Testing {model_name.upper()}")
        print(f"{'-'*70}")
        
        try:
            # Create model
            print(f"  Loading {model_name}...")
            model = create_model(model_name, num_classes=2)
            model.eval()
            
            # Run prediction
            print(f"  Running prediction...")
            with torch.no_grad():
                output = model(image_tensor)
            
            # Get probabilities
            probs = torch.softmax(output, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            
            labels = ['Non-hate', 'Hate']
            
            print(f"\n  ✓ {model_name.upper()} Results:")
            print(f"    Prediction: {labels[pred_idx]}")
            print(f"    Confidence: {probs[pred_idx]:.2%}")
            print(f"    Scores: Non-hate={probs[0]:.2%}, Hate={probs[1]:.2%}\n")
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\n  ❌ Error with {model_name}: {e}\n")
    
    print(f"{'='*70}\n")


def find_sample_image():
    """Find a sample image from your dataset"""
    
    base_path = Path("dataset/images/raw_images")
    
    # Try to find any image
    folders = [
        base_path / "image_text" / "hate_text",
        base_path / "image_text" / "non_hate_text",
        base_path / "image_symbol" / "hate_symbol",
        base_path / "image_symbol" / "non_hate_symbol"
    ]
    
    for folder in folders:
        if folder.exists():
            for img in folder.glob("*"):
                if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    return str(img)
    
    return None


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MODEL TESTING SCRIPT")
    print("="*70)
    
    # Try to find an image
    sample_image = find_sample_image()
    
    if sample_image:
        print(f"\n✓ Found sample image")
        test_models_with_image(sample_image)
    else:
        print("\n⚠️  No images found in dataset!")
        print("\nTo test, add images to:")
        print("  dataset/images/raw_images/image_text/hate_text/")
        print("  dataset/images/raw_images/image_text/non_hate_text/")
        print("  dataset/images/raw_images/image_symbol/hate_symbol/")
        print("  dataset/images/raw_images/image_symbol/non_hate_symbol/")
        print("\nOr specify an image path:")
        print("  test_models_with_image('path/to/your/image.jpg')")
    
    print("\n" + "="*70)
    print("\nTo test with a specific image:")
    print("  from test_with_samples import test_models_with_image")
    print("  test_models_with_image('your_image.jpg')")
    print("="*70 + "\n")