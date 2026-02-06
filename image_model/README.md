# Binary Hate Speech Detection - Quick Reference

## 📁 Your Folder Structure

```
dataset/images/raw_images/
├── image_text/
│   ├── hate_text/          ← 500 hateful memes
│   └── non_hate_text/      ← 500 normal memes
└── image_symbol/
    ├── hate_symbol/        ← 500 hate symbols
    └── non_hate_symbol/    ← 500 normal images

TOTAL TARGET: 2000 images
```

---

## ✅ Files You Have

| File | Purpose | When to Use |
|------|---------|-------------|
| `models/resnet_model.py` | CNN model (baseline) | Training/Testing |
| `models/vit_model.py` | Transformer model | Training/Testing |
| `models/clip_model.py` | Multimodal model (best for memes) | Training/Testing |
| `models/__init__.py` | Easy model creation | All scripts use this |
| `utils/data_loader.py` | Loads your images | Training |
| `utils/__init__.py` | Python package marker | Auto-import |
| `test_with_samples.py` | Tests models work | After collecting images |
| `evaluate_model.py` | Compares trained models | After training |
| `../utils/image_dataset_manager.py` | Organizes images | While collecting |
| `README.md` | This file! | Reference |

---

## 🎯 Workflow

### Step 1: Collect Images (1 week)

**Manual method (easiest):**
```
Download images → Drag to folders:
  - Hate memes → dataset/images/raw_images/image_text/hate_text/
  - Normal memes → dataset/images/raw_images/image_text/non_hate_text/
  - Hate symbols → dataset/images/raw_images/image_symbol/hate_symbol/
  - Normal images → dataset/images/raw_images/image_symbol/non_hate_symbol/
```

**Using manager (organized):**
```python
from utils.image_dataset_manager import ImageDatasetManager

manager = ImageDatasetManager("dataset/images/raw_images")

# Add images
manager.add_hate_meme("downloads/bad_meme.jpg")
manager.add_normal_meme("downloads/funny.jpg")
manager.add_hate_symbol("downloads/swastika.jpg")
manager.add_normal_image("downloads/landscape.jpg")

# Check progress
manager.print_stats()  # Shows: 450/2000 (22.5%)
```

### Step 2: Test Everything Works

```bash
# Test data loader
cd image_model
python utils/data_loader.py

# Test models with your images
python test_with_samples.py
```

### Step 3: Train Models (Coming Next)

```bash
# Train each model
python train_model.py --model resnet
python train_model.py --model vit
python train_model.py --model clip
```

### Step 4: Compare Models

```bash
# Compare all 3
python evaluate_model.py
```

Result: Report showing which model is best!

---

## 📊 Quick Commands

```bash
# Check how many images you have
python ../utils/image_dataset_manager.py

# Test if data loader works
python utils/data_loader.py

# Test if models work
python test_with_samples.py

# After training - compare models
python evaluate_model.py
```

---

## 🎯 Collection Target

| Folder | Target | Status |
|--------|--------|--------|
| `image_text/hate_text` | 500 | ⏳ |
| `image_text/non_hate_text` | 500 | ⏳ |
| `image_symbol/hate_symbol` | 500 | ⏳ |
| `image_symbol/non_hate_symbol` | 500 | ⏳ |
| **TOTAL** | **2000** | **⏳** |

---

## 💡 Where to Collect

**Hate memes (500):**
- Twitter: Search hate keywords in Hindi/English
- Reddit: r/IndiaSpeaks (controversial)

**Normal memes (500):**
- Reddit: r/memes, r/wholesomememes
- Google: "funny memes"

**Hate symbols (500):**
- Google Images: "hate symbols"
- News articles about hate crimes

**Normal images (500):** (EASIEST!)
- Unsplash.com (free stock photos)
- Pexels.com
- Your own photos

---

## 🔧 Troubleshooting

**"No images found":**
- Check your path: `dataset/images/raw_images/image_text/hate_text/`
- Make sure images are `.jpg`, `.png`, `.jpeg`, `.webp`

**"Module not found":**
```bash
# Make sure you're in the right directory
cd image_model
python test_with_samples.py
```

**"CUDA out of memory":**
- Reduce batch_size in data_loader (default is 32, try 16 or 8)

---

## 📞 Next Steps

1. ✅ All files created
2. ⏳ **YOU:** Collect 2000 images (~1 week)
3. ⏳ **ME:** Create training script
4. ⏳ Train all 3 models
5. ⏳ Compare and pick best

---

**Questions? Confused? Ask for help!** 😊