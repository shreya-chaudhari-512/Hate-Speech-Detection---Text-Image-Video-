"""
CLIP (Contrastive Language-Image Pre-training) Model for Image Hate Speech Detection
Location: image_model/models/clip_model.py

Architecture: Dual-encoder (Vision + Text) Transformer (2021)
Pros: Understands both images AND text, perfect for memes
Cons: More complex, requires careful prompt engineering
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor


class CLIPHateDetector(nn.Module):
    """
    CLIP-based hate speech detector
    Pre-trained on 400M image-text pairs
    Perfect for detecting hate in memes (text + image context)
    """
    
    def __init__(self, num_categories=3, dropout=0.3, model_name='openai/clip-vit-base-patch32'):
        """
        Args:
            num_categories: 3 (gender, religion, caste)
            dropout: Dropout rate
            model_name: HuggingFace CLIP model identifier
        """
        super(CLIPHateDetector, self).__init__()
        
        self.num_categories = num_categories
        self.model_name = model_name
        
        # Load pre-trained CLIP
        print(f"Loading CLIP model: {model_name}")
        self.clip = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Get embedding dimension
        self.embed_dim = self.clip.config.projection_dim  # 512 for base
        
        # Custom classification head
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
            nn.Linear(256, num_categories + 1)  # +1 for binary hate/not-hate
        )
        
        # Hate-related text prompts for zero-shot classification
        self.hate_prompts = {
            'gender': [
                "an image with misogynistic content",
                "an image with sexist content",
                "an image showing gender-based hate",
                "a meme targeting women or LGBTQ+ people"
            ],
            'religion': [
                "an image with religious hate speech",
                "an image showing religious intolerance",
                "a meme targeting religious groups",
                "an image with anti-religious content"
            ],
            'caste': [
                "an image with caste-based discrimination",
                "an image showing casteist content",
                "a meme targeting caste groups",
                "an image with regional stereotypes"
            ],
            'normal': [
                "a normal image",
                "a neutral meme",
                "an image without hate speech",
                "a regular social media post"
            ]
        }
        
        # Initialize classifier
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
        Forward pass using vision encoder only
        
        Args:
            x: Input tensor [batch_size, 3, 224, 224]
        
        Returns:
            logits: [batch_size, 4] (hate_binary, gender, religion, caste)
        """
        # Get image embeddings
        vision_outputs = self.clip.vision_model(pixel_values=x)
        image_embeds = vision_outputs.pooler_output  # [batch_size, embed_dim]
        
        # Project to shared space
        image_embeds = self.clip.visual_projection(image_embeds)
        
        # Classify
        logits = self.classifier(image_embeds)  # [batch_size, 4]
        
        return logits
    
    def zero_shot_classify(self, x, text_prompts=None):
        """
        Zero-shot classification using text prompts
        This is CLIP's superpower - no training needed!
        
        Args:
            x: Input tensor [batch_size, 3, 224, 224]
            text_prompts: Optional custom prompts
        
        Returns:
            probs: [batch_size, num_classes] similarity scores
        """
        if text_prompts is None:
            # Use default hate prompts
            all_prompts = []
            for category, prompts in self.hate_prompts.items():
                all_prompts.extend(prompts)
            text_prompts = all_prompts
        
        # Get image embeddings
        with torch.no_grad():
            vision_outputs = self.clip.vision_model(pixel_values=x)
            image_embeds = vision_outputs.pooler_output
            image_embeds = self.clip.visual_projection(image_embeds)
            
            # Normalize
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        
        # Get text embeddings
        text_inputs = self.processor(
            text=text_prompts,
            return_tensors="pt",
            padding=True
        ).to(x.device)
        
        with torch.no_grad():
            text_outputs = self.clip.text_model(**text_inputs)
            text_embeds = text_outputs.pooler_output
            text_embeds = self.clip.text_projection(text_embeds)
            
            # Normalize
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        
        # Calculate similarity
        logits_per_image = torch.matmul(image_embeds, text_embeds.t())
        probs = logits_per_image.softmax(dim=-1)
        
        return probs
    
    def extract_features(self, x):
        """Extract features for visualization/analysis"""
        with torch.no_grad():
            vision_outputs = self.clip.vision_model(pixel_values=x)
            image_embeds = vision_outputs.pooler_output
            image_embeds = self.clip.visual_projection(image_embeds)
        return image_embeds
    
    def freeze_backbone(self):
        """Freeze CLIP backbone, only train classifier"""
        for param in self.clip.parameters():
            param.requires_grad = False
        print("✓ CLIP backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze CLIP backbone for fine-tuning"""
        for param in self.clip.parameters():
            param.requires_grad = True
        print("✓ CLIP backbone unfrozen")
    
    def get_model_info(self):
        """Return model information for comparison"""
        return {
            'name': 'CLIP (Contrastive Language-Image Pre-training)',
            'year': 2021,
            'type': 'Dual-Encoder Transformer (Vision + Text)',
            'params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'architecture': 'Vision Transformer + Text Transformer',
            'pretrained_on': '400M image-text pairs from the internet',
            'model_variant': self.model_name,
            'embed_dim': self.embed_dim,
            'strengths': [
                'BEST for memes (understands text + image context)',
                'Pre-trained on image-text pairs (perfect for our task)',
                'Zero-shot capability (can classify without training)',
                'Strong multimodal understanding',
                'Can use text prompts to guide classification',
                'Excellent at understanding context and nuance'
            ],
            'weaknesses': [
                'Largest model size (~600MB)',
                'Most computationally expensive',
                'More complex to fine-tune properly',
                'May require prompt engineering'
            ],
            'unique_features': [
                'Zero-shot classification',
                'Text-guided image understanding',
                'Can use natural language prompts'
            ]
        }


def create_clip_model(num_categories=3, pretrained=True, freeze_backbone=False):
    """
    Factory function to create CLIP model
    
    Args:
        num_categories: Number of hate categories
        pretrained: Use pre-trained weights
        freeze_backbone: If True, only train classifier head
    
    Returns:
        CLIPHateDetector model
    """
    if not pretrained:
        print("⚠ Creating CLIP WITHOUT pre-trained weights (NOT recommended)")
        print("⚠ CLIP's power comes from its pre-training on 400M image-text pairs")
    
    model = CLIPHateDetector(num_categories=num_categories)
    
    if freeze_backbone:
        model.freeze_backbone()
    
    return model


# Test function
if __name__ == "__main__":
    print("Testing CLIP model...")
    
    # Create model
    model = create_clip_model(num_categories=3, freeze_backbone=False)
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    print("\n--- Standard forward pass ---")
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test zero-shot classification
    print("\n--- Zero-shot classification ---")
    zero_shot_probs = model.zero_shot_classify(dummy_input)
    print(f"Zero-shot output shape: {zero_shot_probs.shape}")
    
    # Model info
    info = model.get_model_info()
    print(f"\nModel: {info['name']}")
    print(f"Total parameters: {info['params']:,}")
    print(f"Trainable parameters: {info['trainable_params']:,}")
    print(f"\nUnique features:")
    for feature in info['unique_features']:
        print(f"  • {feature}")
    
    print("\n✓ CLIP model created successfully!")
    print("\n💡 CLIP is likely the BEST choice for meme detection!")