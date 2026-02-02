"""
Model Factory - Easy access to all models
Location: image_model/models/__init__.py
"""

from .resnet_model import ResNetHateDetector, create_resnet_model
from .vit_model import ViTHateDetector, create_vit_model
from .clip_model import CLIPHateDetector, create_clip_model


def create_model(model_type='resnet', num_categories=3, pretrained=True, freeze_backbone=False):
    """
    Factory function to create any model
    
    Args:
        model_type: 'resnet', 'vit', or 'clip'
        num_categories: Number of hate categories (3)
        pretrained: Use pre-trained weights
        freeze_backbone: Only train classifier head
    
    Returns:
        Model instance
    """
    model_type = model_type.lower()
    
    if model_type == 'resnet' or model_type == 'resnet50':
        return create_resnet_model(num_categories, pretrained)
    
    elif model_type == 'vit' or model_type == 'vit-base':
        return create_vit_model(num_categories, pretrained, freeze_backbone)
    
    elif model_type == 'clip':
        return create_clip_model(num_categories, pretrained, freeze_backbone)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'resnet', 'vit', or 'clip'")


def compare_models_info():
    """
    Print comparison of all three models
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80 + "\n")
    
    models_info = []
    
    # Get info from each model
    for model_name in ['resnet', 'vit', 'clip']:
        print(f"Loading {model_name.upper()}...")
        try:
            model = create_model(model_name, pretrained=True)
            info = model.get_model_info()
            models_info.append(info)
            print(f"✓ {model_name.upper()} loaded\n")
        except Exception as e:
            print(f"✗ Error loading {model_name}: {e}\n")
    
    # Print comparison table
    print("\n" + "="*80)
    print("QUICK COMPARISON")
    print("="*80)
    
    for info in models_info:
        print(f"\n{info['name']} ({info['year']})")
        print(f"  Type: {info['type']}")
        print(f"  Parameters: {info['params']:,}")
        print(f"  Architecture: {info['architecture']}")
        print(f"  Pre-trained on: {info['pretrained_on']}")
        
        print(f"\n  ✅ Strengths:")
        for strength in info['strengths']:
            print(f"     • {strength}")
        
        print(f"\n  ⚠️ Weaknesses:")
        for weakness in info['weaknesses']:
            print(f"     • {weakness}")
        
        print("\n" + "-"*80)
    
    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION FOR HATE SPEECH DETECTION (MEMES)")
    print("="*80)
    print("""
🥇 BEST: CLIP
   - Specifically designed for image-text understanding
   - Pre-trained on 400M image-text pairs
   - Can use zero-shot classification with text prompts
   - Best for memes where context matters
   
🥈 GOOD: ViT (Vision Transformer)
   - Modern architecture with better context understanding
   - Good if you have enough training data
   - Better than ResNet for complex scenes
   
🥉 BASELINE: ResNet50
   - Fast and reliable baseline
   - Good for pure visual content
   - Easier to train with limited data
   
💡 Strategy: Test all three, compare results, justify choice with data!
    """)
    print("="*80 + "\n")


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