"""
Data Loader for YOUR EXACT Folder Structure
SAVE AS: image_model/utils/data_loader.py

This loads images from YOUR folders for training.

YOUR PATH (example):
C:/Users/shrey/.../dataset/images/raw_images/image_symbol/hate_symbol

So structure is:
dataset/images/raw_images/
    image_text/
        hate_text/       (hate = 1)
        non_hate_text/   (hate = 0)
    image_symbol/
        hate_symbol/     (hate = 1)
        non_hate_symbol/ (hate = 0)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path


class HateSpeechImageDataset(Dataset):
    """
    Loads hate/non-hate images from YOUR folders
    
    Returns:
        image: Tensor of shape [3, 224, 224]
        label: 0 (non-hate) or 1 (hate)
    """
    
    def __init__(self, root_dir=r"C:\Users\shrey\FULL\PATH\TO\Hate-Speech-Detection\dataset\images\raw_images", transform=None, split='train'):
        """
        Args:
            root_dir: Path to raw_images folder
            transform: Image transforms (resize, normalize, etc.)
            split: 'train', 'val', 'test', or 'full'
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        
        self.image_paths = []
        self.labels = []
        
        self._load_images()
        
        if len(self.image_paths) > 0:
            hate_count = sum(self.labels)
            non_hate_count = len(self.labels) - hate_count
            print(f"\n{split.upper()} Dataset loaded:")
            print(f"  Total: {len(self.image_paths)} images")
            print(f"  Hate: {hate_count} ({hate_count/len(self.labels)*100:.1f}%)")
            print(f"  Non-hate: {non_hate_count} ({non_hate_count/len(self.labels)*100:.1f}%)")
    
    def _load_images(self):
        """Load images from YOUR folder structure"""
        
        # YOUR folder structure: raw_images/image_text/ and raw_images/image_symbol/
        folders = [
            ('image_text/hate_text', 1),        # Hate memes -> label=1
            ('image_text/non_hate_text', 0),    # Normal memes -> label=0
            ('image_symbol/hate_symbol', 1),    # Hate symbols -> label=1
            ('image_symbol/non_hate_symbol', 0) # Normal images -> label=0
        ]
        
        print(f"Loading from: {self.root_dir}")
        
        for folder_rel_path, label in folders:
            folder_full_path = self.root_dir / folder_rel_path
            
            if folder_full_path.exists():
                # Find all image files
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', 
                                   '.JPG', '.JPEG', '.PNG', '.BMP', '.WEBP']
                
                image_files = [
                    f for f in folder_full_path.iterdir() 
                    if f.is_file() and f.suffix in image_extensions
                ]
                
                # Add to dataset
                for img_file in image_files:
                    self.image_paths.append(str(img_file))
                    self.labels.append(label)
                
                folder_name = folder_rel_path.replace('/', ' > ')
                print(f"  OK {folder_name}: {len(image_files)} images")
            else:
                folder_name = folder_rel_path.replace('/', ' > ')
                print(f"  X  {folder_name}: NOT FOUND")
        
        if len(self.image_paths) == 0:
            print(f"\nNO IMAGES FOUND!")
            print(f"\nExpected folders at:")
            print(f"  {self.root_dir}/image_text/hate_text/")
            print(f"  {self.root_dir}/image_text/non_hate_text/")
            print(f"  {self.root_dir}/image_symbol/hate_symbol/")
            print(f"  {self.root_dir}/image_symbol/non_hate_symbol/")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get one image and its label
        
        Returns:
            image: Tensor [3, 224, 224]
            label: 0 or 1
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms (resize, normalize, etc.)
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(train=True):
    """
    Get image transforms
    
    Training transforms include data augmentation (flips, rotations)
    to create more training data from existing images.
    
    Test transforms only resize and normalize.
    """
    
    if train:
        # Training: WITH augmentation
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),       # Random crop
            transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip
            transforms.RandomRotation(degrees=15),   # Rotate +/- 15 degrees
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]    # ImageNet std
            )
        ])
    else:
        # Test/Val: NO augmentation
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def create_data_loaders(root_dir='dataset/images/raw_images',
                        batch_size=32,
                        train_split=0.7,
                        val_split=0.15,
                        test_split=0.15,
                        num_workers=0):
    """
    Create train, validation, and test data loaders
    
    Args:
        root_dir: Path to raw_images folder
        batch_size: How many images per batch (32 is good)
        train_split: 70% for training
        val_split: 15% for validation (checking during training)
        test_split: 15% for final testing
        num_workers: 0 for Windows, 4+ for Linux
    
    Returns:
        train_loader: For training
        val_loader: For checking during training
        test_loader: For final evaluation
    """
    
    print(f"\n{'='*70}")
    print("CREATING DATA LOADERS")
    print(f"{'='*70}\n")
    
    # Load all images
    full_dataset = HateSpeechImageDataset(
        root_dir=root_dir,
        transform=get_transforms(train=False),
        split='full'
    )
    
    if len(full_dataset) == 0:
        print("\nNo images found! Add images first!")
        return None, None, None
    
    # Split into train/val/test
    total = len(full_dataset)
    train_size = int(train_split * total)
    val_size = int(val_split * total)
    test_size = total - train_size - val_size
    
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Same split every time
    )
    
    # Apply training transforms to train set
    train_ds.dataset.transform = get_transforms(train=True)
    
    # Create loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False,  # Don't shuffle validation
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"\n{'='*70}")
    print("DATA LOADERS CREATED")
    print(f"{'='*70}")
    print(f"Train: {train_size:4d} images ({train_split*100:.0f}%)")
    print(f"Val:   {val_size:4d} images ({val_split*100:.0f}%)")
    print(f"Test:  {test_size:4d} images ({test_split*100:.0f}%)")
    print(f"\nBatch size: {batch_size}")
    print(f"{'='*70}\n")
    
    return train_loader, val_loader, test_loader


# Test this file
if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING DATA LOADER")
    print("="*70 + "\n")
    
    # Test loading dataset
    dataset = HateSpeechImageDataset(
        root_dir='dataset/images/raw_images',
        transform=get_transforms(train=False)
    )
    
    if len(dataset) > 0:
        # Get one sample
        image, label = dataset[0]
        print(f"\nOK Sample loaded:")
        print(f"  Image shape: {image.shape}")
        print(f"  Label: {label} ({'Hate' if label == 1 else 'Non-hate'})")
        
        # Create loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            batch_size=4,
            num_workers=0
        )
        
        if train_loader:
            # Get one batch
            images, labels = next(iter(train_loader))
            print(f"\nOK Batch loaded:")
            print(f"  Batch shape: {images.shape}")
            print(f"  Labels: {labels.tolist()}")
            
            print(f"\n{'='*70}")
            print("SUCCESS - DATA LOADER WORKS!")
            print(f"{'='*70}\n")
    else:
        print("\nNo images found!")
        print("Add images to test the data loader.")