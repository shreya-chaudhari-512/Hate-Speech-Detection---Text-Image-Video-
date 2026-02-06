"""
Vision Transformer (ViT) Model for Binary Hate Speech Detection
SAVE AS: image_model/models/vit_model.py

This is a Transformer (like ChatGPT but for images) from 2021.
Better at understanding context than old CNNs.

Think of it as: Image → ViT → [0.1, 0.9] → "This is hate (90% confidence)"
"""

import torch
import torch.nn as nn
from transformers import ViTModel


class ViTHateDetector(nn.Module):
    """Vision Transformer for binary classification (hate/non-hate)"""
    
    def __init__(self, num_classes=2, dropout=0.3, model_name='google/vit-base-patch16-224'):
        """
        Initialize ViT model
        num_classes=2 means binary: [non-hate, hate]
        """
        super(ViTHateDetector, self).__init__()
        
        print(f"Loading ViT model: {model_name}")
        
        # Load pre-trained Vision Transformer
        self.vit = ViTModel.from_pretrained(model_name)
        
        # Get hidden size (768 for base model)
        self.hidden_size = self.vit.config.hidden_size
        
        # Add our custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 512),
            nn.GELU(),  # Activation function
            nn.LayerNorm(512),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)  # 2 outputs: [non-hate, hate]
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Input: x = [batch_size, 3, 224, 224]
        Output: [batch_size, 2] (scores for [non-hate, hate])
        """
        # ViT processes image and gives us features
        outputs = self.vit(pixel_values=x)
        
        # Use [CLS] token (first token) as image representation
        cls_token = outputs.last_hidden_state[:, 0, :]
        
        # Classify
        logits = self.classifier(cls_token)
        return logits
    
    def freeze_backbone(self):
        """Freeze ViT, only train classifier (faster training)"""
        for param in self.vit.parameters():
            param.requires_grad = False
        print("✓ ViT backbone frozen - only training classifier")
    
    def unfreeze_backbone(self):
        """Unfreeze ViT for full fine-tuning"""
        for param in self.vit.parameters():
            param.requires_grad = True
        print("✓ ViT backbone unfrozen - training all layers")
    
    def get_model_info(self):
        """Return info about this model"""
        return {
            'name': 'Vision Transformer (ViT)',
            'year': 2021,
            'type': 'Transformer',
            'params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


def create_vit_model(num_classes=2, pretrained=True, freeze_backbone=False):
    """
    Factory function to create ViT model
    
    Args:
        num_classes: 2 for binary
        pretrained: Use pre-trained weights
        freeze_backbone: If True, only train classifier (faster)
    
    Returns:
        ViTHateDetector model
    """
    model = ViTHateDetector(num_classes=num_classes)
    
    if freeze_backbone:
        model.freeze_backbone()
    
    return model


# Test if this file works
if __name__ == "__main__":
    print("Testing ViT Model...")
    
    model = create_vit_model(num_classes=2, freeze_backbone=False)
    
    # Create fake input
    dummy_input = torch.randn(4, 3, 224, 224)
    
    # Run through model
    output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # Should be [4, 2]
    print(f"Output example: {output[0]}")
    
    # Get model info
    info = model.get_model_info()
    print(f"\nModel: {info['name']}")
    print(f"Parameters: {info['params']:,}")
    
    print("\n✅ ViT model works!")