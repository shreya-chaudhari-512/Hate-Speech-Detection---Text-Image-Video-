"""
Model Evaluation and Comparison Script
SAVE AS: image_model/evaluate_model.py

After training all 3 models, this compares them and creates a report.

Usage:
    python image_model/evaluate_model.py

Outputs:
    - Accuracy, F1-score, precision, recall for each model
    - Confusion matrices
    - Comparison report for your maam
"""

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import pandas as pd
from pathlib import Path
import json

from models import create_model
from utils.data_loader import create_data_loaders


class ModelEvaluator:
    """Evaluate and compare models"""
    
    def __init__(self, test_loader, device='cuda'):
        self.test_loader = test_loader
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def evaluate_model(self, model, model_name='Model'):
        """
        Evaluate a single model
        
        Returns:
            Dictionary with all metrics
        """
        print(f"\n{'='*70}")
        print(f"EVALUATING: {model_name}")
        print(f"{'='*70}\n")
        
        model.to(self.device)
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        cm = confusion_matrix(all_labels, all_preds)
        
        # Print results
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  [[TN={cm[0,0]:3d}, FP={cm[0,1]:3d}]")
        print(f"   [FN={cm[1,0]:3d}, TP={cm[1,1]:3d}]]")
        
        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist()
        }
    
    def compare_models(self, model_configs):
        """
        Compare multiple models
        
        Args:
            model_configs: List of (model_type, model_path) tuples
        
        Returns:
            DataFrame with comparison
        """
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        
        all_results = []
        
        for model_type, model_path in model_configs:
            # Load model
            model = create_model(model_type, num_classes=2)
            
            if model_path and Path(model_path).exists():
                print(f"\nLoading weights from: {model_path}")
                model.load_state_dict(torch.load(model_path, map_location=self.device))
            else:
                print(f"\n⚠️  No trained weights for {model_type} - using random weights")
                print(f"   (Train the model first for real results)")
            
            # Evaluate
            results = self.evaluate_model(model, model_name=model_type.upper())
            all_results.append(results)
            
            del model
            torch.cuda.empty_cache()
        
        # Create comparison table
        comparison_df = pd.DataFrame([
            {
                'Model': r['model_name'],
                'Accuracy': r['accuracy'],
                'Precision': r['precision'],
                'Recall': r['recall'],
                'F1-Score': r['f1']
            }
            for r in all_results
        ])
        
        print("\n" + "="*70)
        print("COMPARISON TABLE")
        print("="*70)
        print(comparison_df.to_string(index=False))
        
        # Best model
        best_idx = comparison_df['F1-Score'].idxmax()
        best_model = comparison_df.loc[best_idx, 'Model']
        best_f1 = comparison_df.loc[best_idx, 'F1-Score']
        
        print("\n" + "="*70)
        print("RECOMMENDATION")
        print("="*70)
        print(f"\n🏆 Best Model: {best_model}")
        print(f"   F1-Score: {best_f1:.4f}")
        print(f"\n   Reason: Highest F1-score (balance of precision and recall)")
        print("="*70 + "\n")
        
        # Save results
        self._save_results(comparison_df, all_results)
        
        return comparison_df, all_results
    
    def _save_results(self, comparison_df, all_results):
        """Save comparison results"""
        results_dir = Path('image_model/results')
        results_dir.mkdir(exist_ok=True)
        
        # Save CSV
        comparison_df.to_csv(results_dir / 'model_comparison.csv', index=False)
        print(f"✓ Saved: {results_dir / 'model_comparison.csv'}")
        
        # Save detailed JSON
        with open(results_dir / 'detailed_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"✓ Saved: {results_dir / 'detailed_results.json'}")
        
        # Generate text report
        self._generate_report(comparison_df, results_dir / 'report.txt')
    
    def _generate_report(self, comparison_df, output_path):
        """Generate text report for your maam"""
        
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("IMAGE HATE SPEECH DETECTION - MODEL COMPARISON REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("MODELS TESTED:\n")
            f.write("-"*70 + "\n")
            f.write("1. ResNet50 (2015) - CNN baseline\n")
            f.write("2. Vision Transformer (2021) - Modern transformer\n")
            f.write("3. CLIP (2021) - Multimodal (image + text)\n\n")
            
            f.write("RESULTS:\n")
            f.write("-"*70 + "\n")
            f.write(comparison_df.to_string(index=False))
            f.write("\n\n")
            
            best_idx = comparison_df['F1-Score'].idxmax()
            best_model = comparison_df.loc[best_idx, 'Model']
            best_f1 = comparison_df.loc[best_idx, 'F1-Score']
            
            f.write("RECOMMENDATION:\n")
            f.write("-"*70 + "\n")
            f.write(f"Selected Model: {best_model}\n")
            f.write(f"F1-Score: {best_f1:.4f}\n\n")
            f.write("Justification:\n")
            f.write("- Highest F1-score among all tested models\n")
            f.write("- Best balance between precision and recall\n")
            f.write("- Suitable for production deployment\n")
            f.write("="*70 + "\n")
        
        print(f"✓ Saved: {output_path}")


# Main execution
if __name__ == "__main__":
    print("\n" + "="*70)
    print("MODEL EVALUATION SCRIPT")
    print("="*70)
    
    # Load test data
    print("\nLoading test data...")
    _, _, test_loader = create_data_loaders(
        root_dir='dataset/images/raw_images',
        batch_size=32
    )
    
    if test_loader is None:
        print("\n❌ No data found! Add images first.")
        exit()
    
    # Create evaluator
    evaluator = ModelEvaluator(test_loader)
    
    # Define model paths (update after training)
    model_configs = [
        ('resnet', 'image_model/saved_models/resnet_best.pth'),
        ('vit', 'image_model/saved_models/vit_best.pth'),
        ('clip', 'image_model/saved_models/clip_best.pth')
    ]
    
    # Compare models
    comparison_df, results = evaluator.compare_models(model_configs)
    
    print("\n✅ Evaluation complete!")
    print("Check image_model/results/ for detailed results.")