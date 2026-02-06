"""
ResNet50 Model for Binary Hate Speech Detection
SAVE AS: image_model/models/resnet_model.py

This is a CNN (Convolutional Neural Network) from 2015.
It looks at images and classifies them as hate (1) or non-hate (0).

Think of it as: Image → ResNet → [0.2, 0.8] → "This is hate (80% confidence)"
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNetHateDetector(nn.Module):
    """ResNet50 for binary classification (hate/non-hate)"""
    
    def __init__(self, num_classes=2, dropout=0.3):
        """
        Initialize model
        num_classes=2 means binary: [non-hate, hate]
        """
        super(ResNetHateDetector, self).__init__()
        
        # Load pre-trained ResNet50 (trained on ImageNet)
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Get size of last layer
        num_features = self.resnet.fc.in_features  # 2048
        
        # Remove ResNet's classification layer
        self.resnet.fc = nn.Identity()
        
        # Add our custom classification head for hate/non-hate
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)  # 2 outputs: [non-hate, hate]
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights randomly"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the model
        
        Input: x = [batch_size, 3, 224, 224] (batch of images)
        Output: [batch_size, 2] (scores for [non-hate, hate])
        """
        features = self.resnet(x)  # Extract features using ResNet
        logits = self.classifier(features)  # Classify as hate/non-hate
        return logits
    
    def get_model_info(self):
        """Return info about this model"""
        return {
            'name': 'ResNet50',
            'year': 2015,
            'type': 'CNN',
            'params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


def create_resnet_model(num_classes=2, pretrained=True):
    """
    Factory function to create ResNet model
    
    Args:
        num_classes: 2 for binary (hate/non-hate)
        pretrained: Use ImageNet weights (always True for best results)
    
    Returns:
        ResNetHateDetector model
    """
    model = ResNetHateDetector(num_classes=num_classes)
    return model


# Test if this file works
if __name__ == "__main__":
    print("Testing ResNet Model...")
    
    model = create_resnet_model(num_classes=2)
    
    # Create fake input (4 images, 3 channels, 224x224 pixels)
    dummy_input = torch.randn(4, 3, 224, 224)
    
    # Run through model
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # Should be [4, 2]
    print(f"Output example: {output[0]}")  # Two scores
    
    # Get model info
    info = model.get_model_info()
    print(f"\nModel: {info['name']}")
    print(f"Parameters: {info['params']:,}")
    
    print("\n✅ ResNet50 model works!")