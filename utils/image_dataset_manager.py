"""
Image Dataset Manager - Organizes your downloaded images
SAVE AS: utils/image_dataset_manager.py

This helps you organize images as you download them.

YOUR PATH (example):
C:/Users/shrey/.../dataset/images/raw_images/image_symbol/hate_symbol

Example usage:
    manager = ImageDatasetManager()
    manager.add_hate_meme('downloads/bad_meme.jpg')
    manager.print_stats()  # Shows: 450/2000 (22.5%)
"""

import shutil
from pathlib import Path


class ImageDatasetManager:
    """
    Manages your image dataset
    Makes it easy to add images to correct folders
    """
    
    def __init__(self, base_path="dataset/images/raw_images"):
        """
        Initialize manager
        
        Args:
            base_path: Path to raw_images folder
        """
        self.base_path = Path(base_path)
        self._create_folders()
        print(f"OK Dataset manager initialized")
        print(f"  Base path: {self.base_path}")
    
    def _create_folders(self):
        """Create folder structure if it doesn't exist"""
        
        folders = [
            'image_text/hate_text',
            'image_text/non_hate_text',
            'image_symbol/hate_symbol',
            'image_symbol/non_hate_symbol'
        ]
        
        for folder in folders:
            (self.base_path / folder).mkdir(parents=True, exist_ok=True)
    
    def add_image(self, source_path, label, image_type):
        """
        Add image to dataset
        
        Args:
            source_path: Path to your downloaded image
            label: 'hate' or 'non_hate'
            image_type: 'text' (memes) or 'symbol' (images)
        
        Returns:
            Path where image was saved
        
        Example:
            manager.add_image('downloads/meme.jpg', 'hate', 'text')
        """
        
        # Validate
        if label not in ['hate', 'non_hate']:
            print(f"ERROR Label must be 'hate' or 'non_hate', got '{label}'")
            return None
        
        if image_type not in ['text', 'symbol']:
            print(f"ERROR Type must be 'text' or 'symbol', got '{image_type}'")
            return None
        
        source_path = Path(source_path)
        if not source_path.exists():
            print(f"ERROR File not found: {source_path}")
            return None
        
        # Determine destination folder
        folder_map = {
            ('hate', 'text'): 'image_text/hate_text',
            ('non_hate', 'text'): 'image_text/non_hate_text',
            ('hate', 'symbol'): 'image_symbol/hate_symbol',
            ('non_hate', 'symbol'): 'image_symbol/non_hate_symbol'
        }
        
        dest_folder = self.base_path / folder_map[(label, image_type)]
        
        # Handle filename - avoid duplicates
        filename = source_path.name
        dest_path = dest_folder / filename
        
        if dest_path.exists():
            # Add number if file exists
            stem = source_path.stem
            ext = source_path.suffix
            counter = 1
            while dest_path.exists():
                dest_path = dest_folder / f"{stem}_{counter}{ext}"
                counter += 1
        
        # Copy image
        try:
            shutil.copy2(source_path, dest_path)
            rel_path = dest_path.relative_to(self.base_path)
            print(f"OK Saved: {rel_path}")
            return dest_path
        except Exception as e:
            print(f"ERROR: {e}")
            return None
    
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
                exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', 
                       '.JPG', '.JPEG', '.PNG', '.BMP', '.WEBP']
                count = sum(1 for f in folder.iterdir() 
                           if f.is_file() and f.suffix in exts)
                counts[name] = count
            else:
                counts[name] = 0
        
        return counts
    
    def print_stats(self):
        """
        Print dataset statistics with progress bars
        
        This shows:
        - How many images you've collected
        - Progress toward 2000 goal
        - Breakdown by folder
        """
        
        counts = self.count_images()
        
        total = sum(counts.values())
        hate_total = counts['hate_text'] + counts['hate_symbol']
        non_hate_total = counts['non_hate_text'] + counts['non_hate_symbol']
        
        print("\n" + "="*70)
        print("DATASET STATISTICS")
        print("="*70)
        
        # Overall progress
        target = 2000
        progress = min(100, (total / target) * 100)
        bar_len = 50
        filled = int(bar_len * min(total, target) / target)
        bar = chr(9608) * filled + chr(9617) * (bar_len - filled)
        
        print(f"\nTotal: {total}/{target}")
        print(f"[{bar}] {progress:.1f}%")
        
        # By folder
        print("\n" + "-"*70)
        print("BY FOLDER:")
        print("-"*70)
        
        targets = {'hate_text': 500, 'non_hate_text': 500, 
                  'hate_symbol': 500, 'non_hate_symbol': 500}
        
        names = {'hate_text': 'Hate Memes (image_text/hate_text)',
                'non_hate_text': 'Normal Memes (image_text/non_hate_text)',
                'hate_symbol': 'Hate Symbols (image_symbol/hate_symbol)',
                'non_hate_symbol': 'Normal Images (image_symbol/non_hate_symbol)'}
        
        for key in ['hate_text', 'non_hate_text', 'hate_symbol', 'non_hate_symbol']:
            count = counts[key]
            target_count = targets[key]
            pct = (count / target_count * 100) if target_count > 0 else 0
            status = 'OK' if count >= target_count else '..'
            
            bar_small = int(20 * min(count, target_count) / target_count)
            bar_str = chr(9608) * bar_small + chr(9617) * (20 - bar_small)
            
            print(f"{status} {names[key]:<45} [{bar_str}] {count:3d}/{target_count}")
        
        # Summary
        print("\n" + "-"*70)
        print("SUMMARY:")
        print("-"*70)
        print(f"Hate:     {hate_total:4d}/1000 ({hate_total/10:.1f}%)")
        print(f"Non-hate: {non_hate_total:4d}/1000 ({non_hate_total/10:.1f}%)")
        
        if total >= target:
            print("\n" + "="*70)
            print("SUCCESS TARGET REACHED! READY FOR TRAINING!")
            print("="*70)
        else:
            print(f"\nStill need: {target - total} images")
        
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


# Test and example
if __name__ == "__main__":
    print("\n" + "="*70)
    print("IMAGE DATASET MANAGER")
    print("="*70 + "\n")
    
    manager = ImageDatasetManager("dataset/images/raw_images")
    
    print("\nUSAGE EXAMPLES:\n")
    print("# Add hateful meme:")
    print("manager.add_hate_meme('downloads/hate_meme.jpg')")
    print("\n# Add normal meme:")
    print("manager.add_normal_meme('downloads/funny.jpg')")
    print("\n# Add hate symbol:")
    print("manager.add_hate_symbol('downloads/swastika.jpg')")
    print("\n# Add normal photo:")
    print("manager.add_normal_image('downloads/landscape.jpg')")
    print("\n# Check progress:")
    print("manager.print_stats()")
    
    manager.print_stats()