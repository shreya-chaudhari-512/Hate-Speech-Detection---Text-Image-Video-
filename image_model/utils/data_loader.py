"""
Data Loader for YOUR EXACT Folder Structure
SAVE AS: image_model/utils/data_loader.py

Your structure:
dataset/images/
├── image_symbol/
│   ├── hate_symbol/
│   └── non_hate_symbol/
└── image_text/
    ├── hate_text/
    └── non_hate_text/
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path


class HateSpeechImageDataset(Dataset):
    """Binary hate speech dataset - loads from YOUR folders"""
    
    def __init__(self, root_dir='dataset/images', transform=None, split='train'):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        
        self.image_paths = []
        self.labels = []
        
        self._load_images()
        
        print(f"\n{split.upper()} Dataset:")
        print(f"  Total: {len(self.image_paths)} images")
        print(f"  Hate: {sum(self.labels)}")
        print(f"  Non-hate: {len(self.labels) - sum(self.labels)}")
    
    def _load_images(self):
        """Load from YOUR exact folders"""
        
        # Map: (folder_path, label)
        folders = [
            ('image_text/hate_text', 1),
            ('image_text/non_hate_text', 0),
            ('image_symbol/hate_symbol', 1),
            ('image_symbol/non_hate_symbol', 0)
        ]
        
        for folder, label in folders:
            folder_path = self.root_dir / folder
            
            if folder_path.exists():
                count = 0
                for img in folder_path.glob('*'):
                    if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                        self.image_paths.append(str(img))
                        self.labels.append(label)
                        count += 1
                
                print(f"  ✓ {folder}: {count} images")
            else:
                print(f"  ✗ {folder}: not found")
        
        if len(self.image_paths) == 0:
            print(f"\n❌ No images found!")
            print(f"Add images to:")
            print(f"  {self.root_dir}/image_text/hate_text/")
            print(f"  {self.root_dir}/image_text/non_hate_text/")
            print(f"  {self.root_dir}/image_symbol/hate_symbol/")
            print(f"  {self.root_dir}/image_symbol/non_hate_symbol/")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), 'black')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label


def get_transforms(train=True):
    """Image transforms"""
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def create_data_loaders(root_dir='dataset/images', batch_size=32):
    """Create train/val/test loaders"""
    
    print(f"\n{'='*60}")
    print("CREATING DATA LOADERS")
    print(f"{'='*60}")
    
    dataset = HateSpeechImageDataset(root_dir, get_transforms(False), 'full')
    
    if len(dataset) == 0:
        print("\n❌ No images! Add images first!")
        return None, None, None
    
    # Split: 70% train, 15% val, 15% test
    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Train gets augmentation
    train_ds.dataset.transform = get_transforms(True)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"\n✓ Loaders created:")
    print(f"  Train: {train_size} images")
    print(f"  Val:   {val_size} images")
    print(f"  Test:  {test_size} images")
    print(f"  Batch: {batch_size}")
    print(f"{'='*60}\n")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Testing Data Loader...\n")
    
    dataset = HateSpeechImageDataset('dataset/images', get_transforms(False))
    
    if len(dataset) > 0:
        image, label = dataset[0]
        print(f"\n✓ Sample loaded:")
        print(f"  Shape: {image.shape}")
        print(f"  Label: {'Hate' if label == 1 else 'Non-hate'}")
        
        train, val, test = create_data_loaders(batch_size=4)
        
        if train:
            images, labels = next(iter(train))
            print(f"\n✓ Batch test:")
            print(f"  Batch shape: {images.shape}")
            print(f"  Labels: {labels.tolist()}")
            print("\n✅ DATA LOADER WORKS!")
    else:
        print("\n⚠️ Add images to test!")