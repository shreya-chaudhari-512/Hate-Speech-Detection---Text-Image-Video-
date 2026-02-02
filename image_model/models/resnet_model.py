"""
ResNet50 Model for Image Hate Speech Detection
Location: image_model/models/resnet_model.py

Architecture: CNN-based (2015)
Pros: Fast, proven, good baseline
Cons: Older architecture, limited context understanding
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNetHateDetector(nn.Module):
    """
    ResNet50-based hate speech detector
    Pre-trained on ImageNet, fine-tuned for hate detection
    """
    
    def __init__(self, num_classes=2, dropout=0.3):
        """
        Args:
            num_classes: 2 for binary (hate/non-hate), or 3 for categories (gender, religion, caste)
            dropout: Dropout rate for regularization
        """
        super(ResNetHateDetector, self).__init__()
        
        self.num_classes = num_classes
        
        # Load pre-trained ResNet50
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Get number of features from last layer
        num_features = self.resnet.fc.in_features
        
        # Remove original classification layer
        self.resnet.fc = nn.Identity()
        
        # Custom classification head
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
            nn.Linear(256, num_categories + 1)  # +1 for binary hate/not-hate
        )
        
        # Initialize weights
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
        # Extract features with ResNet
        features = self.resnet(x)  # [batch_size, 2048]
        
        # Classify
        logits = self.classifier(features)  # [batch_size, 4]
        
        return logits
    
    def extract_features(self, x):
        """Extract features for visualization/analysis"""
        with torch.no_grad():
            features = self.resnet(x)
        return features
    
    def get_model_info(self):
        """Return model information for comparison"""
        return {
            'name': 'ResNet50',
            'year': 2015,
            'type': 'CNN',
            'params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'architecture': 'Residual Convolutional Neural Network',
            'pretrained_on': 'ImageNet (1.2M images, 1000 classes)',
            'strengths': [
                'Fast inference',
                'Well-established baseline',
                'Good for detecting visual patterns',
                'Lower computational requirements'
            ],
            'weaknesses': [
                'Limited global context understanding',
                'Older architecture (2015)',
                'Not specifically designed for text in images',
                'May miss nuanced meme context'
            ]
        }


def create_resnet_model(num_categories=3, pretrained=True):
    """
    Factory function to create ResNet model
    
    Args:
        num_categories: Number of hate categories
        pretrained: Use ImageNet pre-trained weights
    
    Returns:
        ResNetHateDetector model
    """
    model = ResNetHateDetector(num_categories=num_categories)
    
    if not pretrained:
        # Reinitialize all weights (not recommended)
        print("⚠ Creating ResNet WITHOUT pre-trained weights")
        model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    
    return model


# Test function
if __name__ == "__main__":
    # Create model
    model = create_resnet_model(num_categories=3)
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Model info
    info = model.get_model_info()
    print(f"\nModel: {info['name']}")
    print(f"Total parameters: {info['params']:,}")
    print(f"Trainable parameters: {info['trainable_params']:,}")
    
    print("\n✓ ResNet50 model created successfully!")