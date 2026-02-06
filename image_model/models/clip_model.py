"""
CLIP Model for Binary Hate Speech Detection
SAVE AS: image_model/models/clip_model.py

CLIP understands BOTH images AND text - perfect for memes!
Trained on 400 million image-text pairs.

Think of it as: Meme (image+text) → CLIP → [0.05, 0.95] → "Hate!"
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor


class CLIPHateDetector(nn.Module):
    """CLIP for binary classification (hate/non-hate)"""
    
    def __init__(self, num_classes=2, dropout=0.3, model_name='openai/clip-vit-base-patch32'):
        """
        Initialize CLIP model
        num_classes=2 means binary: [non-hate, hate]
        """
        super(CLIPHateDetector, self).__init__()
        
        print(f"Loading CLIP model: {model_name}")
        
        # Load pre-trained CLIP
        self.clip = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Get embedding dimension (512 for base)
        self.embed_dim = self.clip.config.projection_dim
        
        # Add our custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, 512),
            nn.GELU(),
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
        # Get image embeddings from CLIP
        vision_outputs = self.clip.vision_model(pixel_values=x)
        image_embeds = vision_outputs.pooler_output
        
        # Project to shared space
        image_embeds = self.clip.visual_projection(image_embeds)
        
        # Classify
        logits = self.classifier(image_embeds)
        return logits
    
    def freeze_backbone(self):
        """Freeze CLIP, only train classifier"""
        for param in self.clip.parameters():
            param.requires_grad = False
        print("✓ CLIP backbone frozen - only training classifier")
    
    def unfreeze_backbone(self):
        """Unfreeze CLIP for full fine-tuning"""
        for param in self.clip.parameters():
            param.requires_grad = True
        print("✓ CLIP backbone unfrozen - training all layers")
    
    def get_model_info(self):
        """Return info about this model"""
        return {
            'name': 'CLIP',
            'year': 2021,
            'type': 'Dual-Encoder (Vision + Text)',
            'params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


def create_clip_model(num_classes=2, pretrained=True, freeze_backbone=False):
    """
    Factory function to create CLIP model
    
    Args:
        num_classes: 2 for binary
        pretrained: Use pre-trained weights
        freeze_backbone: If True, only train classifier
    
    Returns:
        CLIPHateDetector model
    """
    model = CLIPHateDetector(num_classes=num_classes)
    
    if freeze_backbone:
        model.freeze_backbone()
    
    return model


# Test if this file works
if __name__ == "__main__":
    print("Testing CLIP Model...")
    
    model = create_clip_model(num_classes=2, freeze_backbone=False)
    
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
    
    print("\n✅ CLIP model works!")
    print("\n💡 CLIP is likely best for meme detection!")