# Image Hate Speech Detection - Model Comparison

## 📁 Folder Structure

```
image_model/
├── models/
│   ├── __init__.py              # Model factory
│   ├── resnet_model.py          # ResNet50 implementation
│   ├── vit_model.py             # Vision Transformer
│   └── clip_model.py            # CLIP model
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py           # Dataset loading (create next)
│   └── metrics.py               # Evaluation metrics (create next)
│
├── train_model.py               # Training script (create next)
├── evaluate_model.py            # Model comparison script
├── predict_image.py             # Inference script (create next)
│
├── saved_models/                # Trained model checkpoints
│   ├── resnet50_best.pth
│   ├── vit_best.pth
│   └── clip_best.pth
│
├── results/                     # Evaluation results
│   ├── model_comparison.csv
│   ├── model_comparison_report.txt
│   ├── detailed_results.json
│   └── confusion_matrices/
│       ├── RESNET_confusion_matrix.png
│       ├── VIT_confusion_matrix.png
│       └── CLIP_confusion_matrix.png
│
└── README.md                    # This file
```

---

## 🎯 Three Models Being Tested

### 1. **ResNet50** (Baseline)
- **Year**: 2015
- **Type**: Convolutional Neural Network (CNN)
- **Parameters**: ~25M
- **Pre-trained on**: ImageNet (1.2M images)

**Strengths**:
- Fast inference (~10ms per image)
- Well-established baseline
- Lower computational requirements
- Good for detecting visual patterns

**Weaknesses**:
- Limited global context understanding
- Older architecture
- Not designed for text in images
- May miss nuanced meme context

**Best for**: Pure visual hate symbols, quick baseline

---

### 2. **Vision Transformer (ViT)** (Modern Alternative)
- **Year**: 2021
- **Type**: Transformer
- **Parameters**: ~86M
- **Pre-trained on**: ImageNet-21k (14M images)

**Strengths**:
- Better global context understanding
- State-of-the-art on many vision tasks
- Attention mechanism captures relationships
- Better for complex scenes

**Weaknesses**:
- Requires more computational resources
- Slower inference (~30ms per image)
- May need more training data
- Larger model size

**Best for**: Complex visual scenes with multiple elements

---

### 3. **CLIP** (Recommended for Memes)
- **Year**: 2021
- **Type**: Dual-Encoder (Vision + Text Transformer)
- **Parameters**: ~151M
- **Pre-trained on**: 400M image-text pairs

**Strengths**:
- **BEST for memes** (understands text + image)
- Pre-trained on image-text pairs
- Zero-shot classification capability
- Strong multimodal understanding
- Can use text prompts for classification

**Weaknesses**:
- Largest model size (~600MB)
- Most computationally expensive (~40ms per image)
- More complex to fine-tune
- Requires prompt engineering

**Best for**: Memes, screenshots, any image with text + visual context

---

## 🔬 Evaluation Metrics

Models will be compared on:

1. **Accuracy**: Overall correctness
2. **F1-Score**: Balance of precision and recall (MOST IMPORTANT)
3. **Precision**: How many predicted hate images are actually hate?
4. **Recall**: How many actual hate images did we catch?
5. **Per-Category Performance**: Gender, Religion, Caste
6. **Inference Speed**: FPS (frames per second)
7. **Confusion Matrix**: Where does the model make mistakes?

---

## 📊 Expected Performance (Prediction)

Based on architecture and pre-training:

| Model | Expected F1 | Speed | Best Use Case |
|-------|-------------|-------|---------------|
| ResNet50 | 0.75-0.80 | Fast | Visual symbols |
| ViT | 0.80-0.85 | Medium | Complex scenes |
| CLIP | 0.85-0.90 | Slow | **Memes with text** |

**Note**: These are predictions. Actual results depend on your dataset!

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install torch torchvision
pip install transformers
pip install scikit-learn pandas matplotlib seaborn
pip install pillow numpy
```

### Step 2: Check Model Info

```python
from models import compare_models_info

# This will print detailed comparison
compare_models_info()
```

### Step 3: Train Models (Coming Next)

```bash
# Train ResNet50
python train_model.py --model resnet --epochs 10

# Train ViT
python train_model.py --model vit --epochs 10

# Train CLIP
python train_model.py --model clip --epochs 10
```

### Step 4: Compare Models

```python
from evaluate_model import ModelEvaluator, generate_comparison_report

# Create evaluator
evaluator = ModelEvaluator(test_loader)

# Compare all three models
model_configs = [
    ('resnet', 'saved_models/resnet50_best.pth'),
    ('vit', 'saved_models/vit_best.pth'),
    ('clip', 'saved_models/clip_best.pth')
]

comparison_df, all_results = evaluator.compare_models(model_configs)

# Generate report for your maam
generate_comparison_report(comparison_df, all_results)
```

---

## 📝 What to Tell Your Maam

### Justification for Model Choice:

**If CLIP performs best (likely):**

> "We tested three state-of-the-art models: ResNet50 (CNN baseline from 2015), Vision Transformer (modern transformer from 2021), and CLIP (multimodal model from 2021).
>
> **We selected CLIP as the final model because:**
> 
> 1. **Highest F1-Score** (X.XX): Best balance of precision and recall
> 2. **Multimodal Pre-training**: Pre-trained on 400M image-text pairs, making it ideal for meme detection where both visual and textual context matter
> 3. **Zero-Shot Capability**: Can classify new hate categories without additional training
> 4. **Superior Context Understanding**: Understands the relationship between text overlays and visual content in memes
> 5. **Production-Ready**: Despite being the largest model, the performance gain justifies the computational cost for a hate speech detection system where accuracy is critical
>
> While ResNet50 was faster (Xms vs Xms), the XX% improvement in F1-Score from CLIP significantly reduces false positives and false negatives, which is crucial for content moderation."

**If ViT performs best:**

> "Vision Transformer achieved the best F1-Score (X.XX) while maintaining reasonable inference speed. Its attention mechanism provides better global context understanding compared to CNNs, making it suitable for detecting hate in complex visual scenes."

**If ResNet50 performs best:**

> "Despite being the oldest architecture, ResNet50 achieved competitive results (F1: X.XX) with the fastest inference speed (Xms per image). Given the performance-to-speed ratio and our dataset size, ResNet50 provides the best balance for production deployment."

---

## 📈 Next Steps

1. ✅ Models created (ResNet, ViT, CLIP)
2. ✅ Evaluation script ready
3. ⏳ Need to create: Training script
4. ⏳ Need to create: Data loader
5. ⏳ Collect 500+ images for training
6. ⏳ Train all three models
7. ⏳ Run comparison
8. ⏳ Generate report

---

## 🤝 Integration with Text Model

Once you choose the best image model:

```python
# Your final pipeline
class MultimodalHateDetector:
    def __init__(self):
        # Image model (your choice)
        self.image_model = create_model('clip')  # or 'vit' or 'resnet'
        
        # Text model (teammate's BERT)
        self.text_model = BERTHateDetector()
        
        # OCR
        self.ocr = OCRExtractor()
        
        # Fusion
        self.fusion = FusionLayer()
    
    def predict(self, image_path):
        # Extract text
        text = self.ocr.extract_text(image_path)
        
        # Get predictions
        text_score = self.text_model.predict(text)
        image_score = self.image_model.predict(image_path)
        
        # Combine
        final_score = self.fusion.fuse(text_score, image_score)
        
        return final_score
```

---

## 📞 Support

If you need help:
1. Check this README
2. Look at code comments in each file
3. Run test functions in each model file
4. Ask me for help! 😊

---

## ✅ File Checklist

**Created ✓:**
- [x] `models/resnet_model.py`
- [x] `models/vit_model.py`
- [x] `models/clip_model.py`
- [x] `models/__init__.py`
- [x] `evaluate_model.py`
- [x] `README.md`

**To Create Next:**
- [ ] `utils/data_loader.py`
- [ ] `train_model.py`
- [ ] `predict_image.py`

**After Data Collection:**
- [ ] Train all three models
- [ ] Run comparison
- [ ] Generate report
- [ ] Choose best model
- [ ] Integrate with text model

---

**Good luck with your project! 🚀**