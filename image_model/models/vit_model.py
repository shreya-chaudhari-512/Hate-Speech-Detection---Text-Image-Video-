"""
Vision Transformer (ViT) Model for Image Hate Speech Detection
Location: image_model/models/vit_model.py

Architecture: Transformer-based (2021)
Pros: Better global context, state-of-the-art for many vision tasks
Cons: More computational resources, needs more data
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig


class ViTHateDetector(nn.Module):
    """
    Vision Transformer-based hate speech detector
    Pre-trained on ImageNet-21k, fine-tuned for hate detection
    """
    
    def __init__(self, num_categories=3, dropout=0.3, model_name='google/vit-base-patch16-224'):
        """
        Args:
            num_categories: 3 (gender, religion, caste)
            dropout: Dropout rate
            model_name: HuggingFace model identifier
        """
        super(ViTHateDetector, self).__init__()
        
        self.num_categories = num_categories
        self.model_name = model_name
        
        # Load pre-trained ViT
        print(f"Loading ViT model: {model_name}")
        self.vit = ViTModel.from_pretrained(model_name)
        
        # Get hidden size
        self.hidden_size = self.vit.config.hidden_size  # 768 for base
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 512),
            nn.GELU(),  # ViT uses GELU activation
            nn.LayerNorm(512),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_categories + 1)  # +1 for binary hate/not-hate
        )
        
        # Initialize classifier weights
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
        
        Args:
            x: Input tensor [batch_size, 3, 224, 224]
        
        Returns:
            logits: [batch_size, 4] (hate_binary, gender, religion, caste)
        """
        # ViT forward pass
        outputs = self.vit(pixel_values=x)
        
        # Use [CLS] token representation (first token)
        cls_token = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Classify
        logits = self.classifier(cls_token)  # [batch_size, 4]
        
        return logits
    
    def extract_features(self, x):
        """Extract features for visualization/analysis"""
        with torch.no_grad():
            outputs = self.vit(pixel_values=x)
            cls_token = outputs.last_hidden_state[:, 0, :]
        return cls_token
    
    def freeze_backbone(self):
        """Freeze ViT backbone, only train classifier"""
        for param in self.vit.parameters():
            param.requires_grad = False
        print("✓ ViT backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze ViT backbone for fine-tuning"""
        for param in self.vit.parameters():
            param.requires_grad = True
        print("✓ ViT backbone unfrozen")
    
    def get_model_info(self):
        """Return model information for comparison"""
        return {
            'name': 'Vision Transformer (ViT)',
            'year': 2021,
            'type': 'Transformer',
            'params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'architecture': 'Transformer encoder with patch embeddings',
            'pretrained_on': 'ImageNet-21k (14M images, 21k classes)',
            'model_variant': self.model_name,
            'hidden_size': self.hidden_size,
            'strengths': [
                'Better global context understanding',
                'State-of-the-art performance on many tasks',
                'Attention mechanism captures relationships',
                'Better for complex scenes (like memes)',
                'Can attend to both text and visual elements'
            ],
            'weaknesses': [
                'Requires more computational resources',
                'Slower inference than CNNs',
                'May need more training data',
                'Larger model size (~340MB vs ~100MB for ResNet)'
            ]
        }


def create_vit_model(num_categories=3, pretrained=True, freeze_backbone=False):
    """
    Factory function to create ViT model
    
    Args:
        num_categories: Number of hate categories
        pretrained: Use ImageNet pre-trained weights
        freeze_backbone: If True, only train classifier head
    
    Returns:
        ViTHateDetector model
    """
    if not pretrained:
        print("⚠ Creating ViT WITHOUT pre-trained weights (not recommended)")
        model = ViTHateDetector(
            num_categories=num_categories,
            model_name='google/vit-base-patch16-224'
        )
        # Reinitialize ViT weights
        model.vit.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    else:
        model = ViTHateDetector(num_categories=num_categories)
    
    if freeze_backbone:
        model.freeze_backbone()
    
    return model


# Test function
if __name__ == "__main__":
    print("Testing ViT model...")
    
    # Create model
    model = create_vit_model(num_categories=3, freeze_backbone=False)
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Model info
    info = model.get_model_info()
    print(f"\nModel: {info['name']}")
    print(f"Total parameters: {info['params']:,}")
    print(f"Trainable parameters: {info['trainable_params']:,}")
    
    # Test freezing
    print("\n--- Testing backbone freezing ---")
    model.freeze_backbone()
    frozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params after freezing: {frozen_trainable:,}")
    
    model.unfreeze_backbone()
    unfrozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params after unfreezing: {unfrozen_trainable:,}")
    
    print("\n✓ ViT model created successfully!")