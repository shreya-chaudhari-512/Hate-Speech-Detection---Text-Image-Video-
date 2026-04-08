"""
Interactive Image Tester with File Picker
SAVE AS: image_model/test_my_image.py

This lets you:
1. Click a button to browse and select an image
2. Test it with all 3 models
3. See predictions immediately

Usage:
    python image_model/test_my_image.py
"""

import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
import os
import tkinter as tk
from tkinter import filedialog

from models import create_model


def test_image(image_path):
    """
    Test one image with all 3 models
    
    Args:
        image_path: Path to image file
    """
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"\nERROR: File not found!")
        print(f"Path: {image_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"TESTING IMAGE")
    print(f"{'='*70}")
    print(f"File: {Path(image_path).name}")
    print(f"Path: {image_path}")
    print(f"{'='*70}\n")
    
    # Load image
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"OK Image loaded")
        print(f"  Size: {image.size[0]}x{image.size[1]} pixels")
        print(f"  Format: {Path(image_path).suffix}")
    except Exception as e:
        print(f"ERROR loading image: {e}")
        return
    
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    print(f"  Preprocessed: {image_tensor.shape}\n")
    
    # Test each model
    models_to_test = [
        ('resnet', 'ResNet50 (CNN)'),
        ('vit', 'Vision Transformer'),
        ('clip', 'CLIP (Multimodal)')
    ]
    
    results = []
    
    for model_type, model_display in models_to_test:
        print(f"{'-'*70}")
        print(f"{model_display.upper()}")
        print(f"{'-'*70}")
        
        try:
            # Load model
            print(f"  Loading...")
            model = create_model(model_type, num_classes=2)
            model.eval()
            
            # Predict
            with torch.no_grad():
                output = model(image_tensor)
            
            # Get probabilities
            probs = torch.softmax(output, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            
            labels = ['Non-hate', 'Hate']
            prediction = labels[pred_idx]
            confidence = probs[pred_idx].item()
            
            # Store result
            results.append({
                'model': model_display,
                'prediction': prediction,
                'confidence': confidence,
                'non_hate_score': probs[0].item(),
                'hate_score': probs[1].item()
            })
            
            # Print result
            print(f"\n  Prediction: {prediction}")
            print(f"  Confidence: {confidence:.1%}")
            print(f"  Scores: Non-hate={probs[0]:.1%}, Hate={probs[1]:.1%}\n")
            
            # Cleanup
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\n  ERROR: {e}\n")
    
    # Summary
    print(f"{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    
    for r in results:
        print(f"{r['model']:<25} -> {r['prediction']:<10} ({r['confidence']:.1%})")
    
    print(f"\n{'='*70}")
    print("NOTE: Models are UNTRAINED - predictions are random guesses!")
    print("After training, they will be accurate (85-90%).")
    print(f"{'='*70}\n")


def browse_and_select_image():
    """
    Open file picker dialog to select an image
    
    Returns:
        Path to selected image, or None if cancelled
    """
    
    # Create root window (hidden)
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front
    
    print("\nOpening file picker...")
    print("(A window should appear - if not, check your taskbar!)")
    
    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select an image to test",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.JPG *.JPEG *.PNG *.BMP *.WEBP"),
            ("JPEG files", "*.jpg *.jpeg *.JPG *.JPEG"),
            ("PNG files", "*.png *.PNG"),
            ("All files", "*.*")
        ],
        initialdir=os.path.expanduser("~")  # Start in user's home directory
    )
    
    root.destroy()  # Close the root window
    
    if file_path:
        return file_path
    else:
        print("No file selected.")
        return None


def interactive_mode():
    """Interactive mode with file picker"""
    
    print("\n" + "="*70)
    print("INTERACTIVE IMAGE TESTER")
    print("="*70)
    print("\nTest any image with ResNet, ViT, and CLIP!")
    print("="*70 + "\n")
    
    while True:
        print("\nOptions:")
        print("  1. Browse and select an image (FILE PICKER)")
        print("  2. Enter image path manually")
        print("  3. Test sample from dataset")
        print("  4. Quit")
        
        choice = input("\nYour choice (1/2/3/4): ").strip()
        
        if choice == '1':
            # Open file picker
            image_path = browse_and_select_image()
            
            if image_path:
                test_image(image_path)
            
        elif choice == '2':
            # Manual path entry
            print("\nEnter image path (or 'back' to return):")
            path = input("Path: ").strip()
            
            if path.lower() == 'back':
                continue
            
            # Clean up path (remove quotes if user copied with quotes)
            path = path.strip('"').strip("'")
            
            test_image(path)
        
        elif choice == '3':
            # Test sample from dataset
            base = Path('dataset/images/raw_images')
            folders = [
                base / 'image_text' / 'hate_text',
                base / 'image_text' / 'non_hate_text',
                base / 'image_symbol' / 'hate_symbol',
                base / 'image_symbol' / 'non_hate_symbol'
            ]
            
            sample_found = False
            for folder in folders:
                if folder.exists():
                    for img in folder.glob('*'):
                        if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                            print(f"\nUsing sample: {img}")
                            test_image(str(img))
                            sample_found = True
                            break
                if sample_found:
                    break
            
            if not sample_found:
                print("\nNo sample images found in dataset!")
                print("Add images to test this option.")
        
        elif choice == '4':
            print("\nGoodbye!")
            break
        
        else:
            print("\nInvalid choice! Please enter 1, 2, 3, or 4.")


def quick_test(image_path):
    """Quick test without interactive mode"""
    test_image(image_path)


if __name__ == "__main__":
    import sys
    
    # Check if path provided as argument
    if len(sys.argv) > 1:
        # Command line: python test_my_image.py "C:/path/to/image.jpg"
        image_path = sys.argv[1]
        quick_test(image_path)
    else:
        # Interactive mode
        interactive_mode()