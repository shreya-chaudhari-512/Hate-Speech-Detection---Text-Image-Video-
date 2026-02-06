"""
Model Factory - Easy model creation
SAVE AS: image_model/models/__init__.py

This makes it super easy to create any model.
Instead of complex code, just: create_model('resnet')
"""

from .resnet_model import ResNetHateDetector, create_resnet_model
from .vit_model import ViTHateDetector, create_vit_model
from .clip_model import CLIPHateDetector, create_clip_model


def create_model(model_type='resnet', num_classes=2, pretrained=True, freeze_backbone=False):
    """
    Create any model with one line of code!
    
    Args:
        model_type: 'resnet', 'vit', or 'clip'
        num_classes: 2 for binary (hate/non-hate)
        pretrained: Use pre-trained weights (always True for best results)
        freeze_backbone: Only train classifier (faster training)
    
    Returns:
        Model ready to train
    
    Example:
        model = create_model('resnet')  # That's it!
    """
    
    model_type = model_type.lower()
    
    if model_type in ['resnet', 'resnet50']:
        return create_resnet_model(num_classes, pretrained)
    
    elif model_type in ['vit', 'vit-base']:
        return create_vit_model(num_classes, pretrained, freeze_backbone)
    
    elif model_type == 'clip':
        return create_clip_model(num_classes, pretrained, freeze_backbone)
    
    else:
        raise ValueError(f"Unknown model: {model_type}. Choose 'resnet', 'vit', or 'clip'")


def compare_models_info():
    """Print comparison of all three models"""
    
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70 + "\n")
    
    models = {
        'ResNet50': create_resnet_model(),
        'ViT': create_vit_model(),
        'CLIP': create_clip_model()
    }
    
    for name, model in models.items():
        info = model.get_model_info()
        print(f"{name}:")
        print(f"  Type: {info['type']}")
        print(f"  Year: {info['year']}")
        print(f"  Parameters: {info['params']:,}")
        print()
    
    print("="*70)
    print("\n💡 Recommendation: CLIP is likely best for memes!")
    print("   But we'll train all 3 and compare results.\n")


# Export everything
__all__ = [
    'ResNetHateDetector',
    'ViTHateDetector',
    'CLIPHateDetector',
    'create_model',
    'create_resnet_model',
    'create_vit_model',
    'create_clip_model',
    'compare_models_info'
]


# Test
if __name__ == "__main__":
    print("Testing Model Factory...")
    
    # Create each model easily
    resnet = create_model('resnet')
    print("✓ Created ResNet")
    
    vit = create_model('vit')
    print("✓ Created ViT")
    
    clip = create_model('clip')
    print("✓ Created CLIP")
    
    print("\n✅ Model factory works!")
    
    # Show comparison
    compare_models_info()