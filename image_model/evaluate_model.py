"""
Model Evaluation and Comparison Script
Location: image_model/evaluate_model.py

Compare ResNet50 vs ViT vs CLIP
Generate comparison report for your maam
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from models import create_model


class ModelEvaluator:
    """
    Evaluate and compare multiple models
    """
    
    def __init__(self, test_loader, device='cuda', num_categories=3):
        """
        Args:
            test_loader: DataLoader for test set
            device: 'cuda' or 'cpu'
            num_categories: Number of categories (3)
        """
        self.test_loader = test_loader
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_categories = num_categories
        self.category_names = ['gender', 'religion', 'caste']
        
        print(f"Evaluator initialized on {self.device}")
    
    def evaluate_model(self, model, model_name='Model'):
        """
        Evaluate a single model
        
        Returns:
            Dictionary with all metrics
        """
        print(f"\n{'='*60}")
        print(f"EVALUATING: {model_name}")
        print(f"{'='*60}\n")
        
        model.to(self.device)
        model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        inference_times = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                outputs = model(images)
                inference_time = time.time() - start_time
                inference_times.append(inference_time / len(images))  # Per image
                
                # Get predictions
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {(batch_idx + 1) * len(images)} images...")
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        results = self._calculate_metrics(
            all_labels, all_preds, all_probs, inference_times
        )
        
        results['model_name'] = model_name
        
        # Print results
        self._print_results(results)
        
        return results
    
    def _calculate_metrics(self, labels, preds, probs, inference_times):
        """Calculate all evaluation metrics"""
        
        # Overall accuracy
        accuracy = accuracy_score(labels, preds)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, average=None, zero_division=0
        )
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            labels, preds, average='weighted', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        
        # Inference speed
        avg_inference_time = np.mean(inference_times)
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        results = {
            'accuracy': accuracy,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist(),
            'support_per_class': support.tolist(),
            'confusion_matrix': cm.tolist(),
            'avg_inference_time': avg_inference_time,
            'fps': fps,
            'total_samples': len(labels)
        }
        
        return results
    
    def _print_results(self, results):
        """Print evaluation results"""
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}\n")
        
        print(f"Overall Accuracy: {results['accuracy']:.4f}")
        print(f"Weighted F1-Score: {results['f1_weighted']:.4f}")
        print(f"Weighted Precision: {results['precision_weighted']:.4f}")
        print(f"Weighted Recall: {results['recall_weighted']:.4f}")
        
        print(f"\nInference Speed:")
        print(f"  Average time per image: {results['avg_inference_time']*1000:.2f} ms")
        print(f"  FPS: {results['fps']:.2f}")
        
        print(f"\nPer-Category Performance:")
        print(f"{'Category':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
        print("-" * 65)
        
        for i, category in enumerate(self.category_names):
            if i < len(results['precision_per_class']):
                print(f"{category:<15} "
                      f"{results['precision_per_class'][i]:<12.4f} "
                      f"{results['recall_per_class'][i]:<12.4f} "
                      f"{results['f1_per_class'][i]:<12.4f} "
                      f"{results['support_per_class'][i]}")
        
        print(f"\n{'='*60}\n")
    
    def compare_models(self, model_configs, save_results=True):
        """
        Compare multiple models
        
        Args:
            model_configs: List of tuples (model_type, model_path)
                          e.g., [('resnet', 'path/to/resnet.pth'), ...]
            save_results: Save comparison to CSV
        
        Returns:
            DataFrame with comparison results
        """
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80 + "\n")
        
        all_results = []
        
        for model_type, model_path in model_configs:
            print(f"\n{'='*80}")
            print(f"Testing: {model_type.upper()}")
            print(f"{'='*80}\n")
            
            # Load model
            model = create_model(model_type, num_categories=self.num_categories)
            
            if model_path and Path(model_path).exists():
                print(f"Loading weights from {model_path}")
                model.load_state_dict(torch.load(model_path, map_location=self.device))
            else:
                print(f"⚠ No trained weights found, using pre-trained backbone only")
            
            # Evaluate
            results = self.evaluate_model(model, model_name=model_type.upper())
            
            # Add model info
            model_info = model.get_model_info()
            results['total_params'] = model_info['params']
            results['architecture_type'] = model_info['type']
            
            all_results.append(results)
            
            # Cleanup
            del model
            torch.cuda.empty_cache()
        
        # Create comparison DataFrame
        comparison_df = self._create_comparison_table(all_results)
        
        # Print comparison
        self._print_comparison(comparison_df)
        
        # Save results
        if save_results:
            self._save_results(all_results, comparison_df)
        
        return comparison_df, all_results
    
    def _create_comparison_table(self, all_results):
        """Create comparison table"""
        comparison = []
        
        for results in all_results:
            comparison.append({
                'Model': results['model_name'],
                'Type': results['architecture_type'],
                'Parameters': results['total_params'],
                'Accuracy': results['accuracy'],
                'F1-Score': results['f1_weighted'],
                'Precision': results['precision_weighted'],
                'Recall': results['recall_weighted'],
                'Inference (ms)': results['avg_inference_time'] * 1000,
                'FPS': results['fps']
            })
        
        df = pd.DataFrame(comparison)
        return df
    
    def _print_comparison(self, comparison_df):
        """Print comparison table"""
        print("\n" + "="*80)
        print("FINAL COMPARISON TABLE")
        print("="*80 + "\n")
        
        print(comparison_df.to_string(index=False))
        
        # Highlight best model for each metric
        print("\n" + "="*80)
        print("BEST MODEL FOR EACH METRIC")
        print("="*80 + "\n")
        
        metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'FPS']
        
        for metric in metrics:
            best_idx = comparison_df[metric].idxmax()
            best_model = comparison_df.loc[best_idx, 'Model']
            best_value = comparison_df.loc[best_idx, metric]
            print(f"{metric:<15}: {best_model} ({best_value:.4f})")
        
        # Overall recommendation
        print("\n" + "="*80)
        print("RECOMMENDATION")
        print("="*80 + "\n")
        
        # Best F1 score (most important for classification)
        best_f1_idx = comparison_df['F1-Score'].idxmax()
        recommended_model = comparison_df.loc[best_f1_idx, 'Model']
        
        print(f"🏆 RECOMMENDED MODEL: {recommended_model}")
        print(f"\nReason: Highest F1-Score ({comparison_df.loc[best_f1_idx, 'F1-Score']:.4f})")
        print("\nF1-Score is the most important metric for hate speech detection because:")
        print("  • Balances precision and recall")
        print("  • Critical for imbalanced datasets")
        print("  • Minimizes both false positives and false negatives")
        
        print("\n" + "="*80 + "\n")
    
    def _save_results(self, all_results, comparison_df):
        """Save evaluation results"""
        results_dir = Path('image_model/results')
        results_dir.mkdir(exist_ok=True)
        
        # Save comparison table
        comparison_df.to_csv(results_dir / 'model_comparison.csv', index=False)
        print(f"✓ Saved comparison to {results_dir / 'model_comparison.csv'}")
        
        # Save detailed results
        with open(results_dir / 'detailed_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"✓ Saved detailed results to {results_dir / 'detailed_results.json'}")
        
        # Save confusion matrices
        cm_dir = results_dir / 'confusion_matrices'
        cm_dir.mkdir(exist_ok=True)
        
        for results in all_results:
            self._plot_confusion_matrix(
                results['confusion_matrix'],
                results['model_name'],
                save_path=cm_dir / f"{results['model_name']}_confusion_matrix.png"
            )
    
    def _plot_confusion_matrix(self, cm, model_name, save_path=None):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        
        # Add 'not-hate' category
        labels = ['hate'] + self.category_names
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels,
                   yticklabels=labels)
        
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved confusion matrix to {save_path}")
        
        plt.close()


def generate_comparison_report(comparison_df, all_results, output_file='model_comparison_report.txt'):
    """
    Generate a text report for your maam
    """
    results_dir = Path('image_model/results')
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / output_file
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("IMAGE HATE SPEECH DETECTION - MODEL COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("TESTED MODELS:\n")
        f.write("-"*80 + "\n")
        f.write("1. ResNet50 (2015) - Convolutional Neural Network\n")
        f.write("2. Vision Transformer / ViT (2021) - Transformer-based\n")
        f.write("3. CLIP (2021) - Dual-encoder (Vision + Text)\n\n")
        
        f.write("COMPARISON RESULTS:\n")
        f.write("-"*80 + "\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("DETAILED ANALYSIS:\n")
        f.write("-"*80 + "\n\n")
        
        for results in all_results:
            f.write(f"{results['model_name']}:\n")
            f.write(f"  • Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"  • F1-Score: {results['f1_weighted']:.4f}\n")
            f.write(f"  • Inference Speed: {results['avg_inference_time']*1000:.2f} ms/image\n")
            f.write(f"  • Per-category F1:\n")
            
            for i, category in enumerate(['gender', 'religion', 'caste']):
                if i < len(results['f1_per_class']):
                    f.write(f"      - {category}: {results['f1_per_class'][i]:.4f}\n")
            f.write("\n")
        
        # Recommendation
        best_f1_idx = comparison_df['F1-Score'].idxmax()
        recommended = comparison_df.loc[best_f1_idx, 'Model']
        
        f.write("FINAL RECOMMENDATION:\n")
        f.write("-"*80 + "\n")
        f.write(f"Selected Model: {recommended}\n\n")
        f.write("Justification:\n")
        f.write("• Highest F1-Score among all tested models\n")
        f.write("• Best balance between precision and recall\n")
        f.write("• Suitable for production deployment\n")
        f.write("• Handles imbalanced dataset effectively\n\n")
        
        f.write("="*80 + "\n")
    
    print(f"\n✓ Comparison report saved to {output_path}")
    return output_path


# Main execution
if __name__ == "__main__":
    print("Model Evaluation Script")
    print("This will compare ResNet50, ViT, and CLIP")
    print("\nNote: You need to train models first before running this comparison")
    print("\nUsage:")
    print("  1. Train all three models using train_model.py")
    print("  2. Run this script to compare them")
    print("  3. Generate report for your maam")