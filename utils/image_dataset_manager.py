"""
Dataset Manager for YOUR EXACT Folder Structure
SAVE AS: utils/image_dataset_manager.py

Your structure:
dataset/images/
├── image_symbol/
│   ├── hate_symbol/
│   └── non_hate_symbol/
└── image_text/
    ├── hate_text/
    └── non_hate_text/
"""

import os
import shutil
from pathlib import Path
from datetime import datetime


class ImageDatasetManager:
    def __init__(self, base_path="dataset/images"):
        self.base_path = Path(base_path)
        self._create_folders()
    
    def _create_folders(self):
        """Create YOUR folder structure"""
        folders = [
            'image_text/hate_text',
            'image_text/non_hate_text',
            'image_symbol/hate_symbol',
            'image_symbol/non_hate_symbol',
            'metadata'
        ]
        
        for folder in folders:
            (self.base_path / folder).mkdir(parents=True, exist_ok=True)
    
    def add_image(self, source_path, label, image_type):
        """
        Add image to dataset
        
        Args:
            source_path: Path to your downloaded image
            label: 'hate' or 'non_hate'
            image_type: 'text' or 'symbol'
        
        Returns:
            Path where image was saved
        """
        
        # Validate
        if label not in ['hate', 'non_hate']:
            print(f"❌ Label must be 'hate' or 'non_hate', got '{label}'")
            return None
        
        if image_type not in ['text', 'symbol']:
            print(f"❌ Type must be 'text' or 'symbol', got '{image_type}'")
            return None
        
        if not Path(source_path).exists():
            print(f"❌ File not found: {source_path}")
            return None
        
        # Determine destination folder
        if image_type == 'text':
            if label == 'hate':
                dest_folder = self.base_path / 'image_text' / 'hate_text'
            else:
                dest_folder = self.base_path / 'image_text' / 'non_hate_text'
        else:  # symbol
            if label == 'hate':
                dest_folder = self.base_path / 'image_symbol' / 'hate_symbol'
            else:
                dest_folder = self.base_path / 'image_symbol' / 'non_hate_symbol'
        
        # Copy file
        filename = Path(source_path).name
        dest_path = dest_folder / filename
        
        # If file exists, add number
        counter = 1
        while dest_path.exists():
            stem = Path(source_path).stem
            ext = Path(source_path).suffix
            dest_path = dest_folder / f"{stem}_{counter}{ext}"
            counter += 1
        
        shutil.copy2(source_path, dest_path)
        print(f"✓ Saved to: {dest_path.relative_to(self.base_path)}")
        
        return dest_path
    
    def count_images(self):
        """Count images in each folder"""
        
        folders = {
            'hate_text': self.base_path / 'image_text' / 'hate_text',
            'non_hate_text': self.base_path / 'image_text' / 'non_hate_text',
            'hate_symbol': self.base_path / 'image_symbol' / 'hate_symbol',
            'non_hate_symbol': self.base_path / 'image_symbol' / 'non_hate_symbol'
        }
        
        counts = {}
        for name, folder in folders.items():
            if folder.exists():
                count = len(list(folder.glob('*.[jp][pn][g]'))) + len(list(folder.glob('*.jpeg'))) + len(list(folder.glob('*.webp')))
                counts[name] = count
            else:
                counts[name] = 0
        
        return counts
    
    def print_stats(self):
        """Print dataset statistics"""
        
        counts = self.count_images()
        
        total = sum(counts.values())
        hate_total = counts['hate_text'] + counts['hate_symbol']
        non_hate_total = counts['non_hate_text'] + counts['non_hate_symbol']
        
        print("\n" + "="*70)
        print("DATASET STATISTICS")
        print("="*70)
        print(f"\nTotal Images: {total}/2000")
        
        # Progress bar
        progress = min(100, (total / 2000) * 100)
        bar_len = 50
        filled = int(bar_len * total / 2000)
        bar = '█' * filled + '░' * (bar_len - filled)
        print(f"[{bar}] {progress:.1f}%")
        
        print("\n" + "-"*70)
        print("BY FOLDER:")
        print("-"*70)
        
        targets = {
            'hate_text': 500,
            'non_hate_text': 500,
            'hate_symbol': 500,
            'non_hate_symbol': 500
        }
        
        for name, count in counts.items():
            target = targets[name]
            pct = (count / target * 100) if target > 0 else 0
            status = '✓' if count >= target else '⏳'
            
            # Small progress bar
            bar_small = int(20 * count / target) if target > 0 else 0
            bar_str = '█' * bar_small + '░' * (20 - bar_small)
            
            # Format folder name
            folder_display = name.replace('_', ' ').title()
            
            print(f"{status} {folder_display:<20} [{bar_str}] {count:3d}/{target} ({pct:5.1f}%)")
        
        print("\n" + "-"*70)
        print("SUMMARY:")
        print("-"*70)
        print(f"Hate:     {hate_total:4d}/1000 ({hate_total/10:.1f}%)")
        print(f"Non-hate: {non_hate_total:4d}/1000 ({non_hate_total/10:.1f}%)")
        
        if total >= 2000:
            print("\n" + "="*70)
            print("🎉 TARGET REACHED! READY FOR TRAINING!")
            print("="*70)
        else:
            print(f"\n📝 Still need: {2000 - total} images")
        
        print("="*70 + "\n")
    
    # Quick add functions
    def add_hate_meme(self, source_path):
        """Quick: Add hateful meme"""
        return self.add_image(source_path, 'hate', 'text')
    
    def add_normal_meme(self, source_path):
        """Quick: Add normal meme"""
        return self.add_image(source_path, 'non_hate', 'text')
    
    def add_hate_symbol(self, source_path):
        """Quick: Add hate symbol"""
        return self.add_image(source_path, 'hate', 'symbol')
    
    def add_normal_image(self, source_path):
        """Quick: Add normal image"""
        return self.add_image(source_path, 'non_hate', 'symbol')


if __name__ == "__main__":
    manager = ImageDatasetManager("dataset/images")
    
    print("\n" + "="*70)
    print("IMAGE DATASET MANAGER")
    print("="*70)
    
    print("\nUSAGE:\n")
    
    print("# Add hateful meme:")
    print("manager.add_hate_meme('downloads/hate_meme.jpg')")
    
    print("\n# Add normal meme:")
    print("manager.add_normal_meme('downloads/funny_meme.jpg')")
    
    print("\n# Add hate symbol:")
    print("manager.add_hate_symbol('downloads/swastika.jpg')")
    
    print("\n# Add normal photo:")
    print("manager.add_normal_image('downloads/landscape.jpg')")
    
    print("\n# Check progress:")
    print("manager.print_stats()")
    
    # Show stats
    manager.print_stats()