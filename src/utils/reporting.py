"""
Unified reporting utilities for accommodation speech detection
Consolidates all report generation functionality
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate various types of reports from evaluation results"""
    
    def __init__(self, results: Dict):
        """Initialize with evaluation results"""
        self.results = results
        self.timestamp = datetime.now()
        
        # Set matplotlib style
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['font.size'] = 10
    
    def generate_all_reports(self, output_dir: str = "results"):
        """Generate all report types"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate different report types
        self.save_json_report(output_path / "detailed_results.json")
        self.save_markdown_report(output_path / "evaluation_report.md")
        self.save_visualizations(output_path)
        
        logger.info(f"All reports generated in {output_dir}")
    
    def save_json_report(self, filepath: Path):
        """Save detailed JSON report"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON report saved to {filepath}")
    
    def save_markdown_report(self, filepath: Path):
        """Generate and save markdown report"""
        metrics = self.results['aggregate_metrics']
        cv_config = self.results['cross_validation']
        dataset_stats = self.results['dataset_stats']
        errors = self.results['errors']
        
        report = f"""# Accommodation Speech Detection - Evaluation Report

Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Statistics
- **Total Samples**: {dataset_stats['total']}
- **Harmful Samples**: {dataset_stats['harmful']} ({dataset_stats['harmful']/dataset_stats['total']*100:.1f}%)
- **Safe Samples**: {dataset_stats['safe']} ({dataset_stats['safe']/dataset_stats['total']*100:.1f}%)

## Cross-Validation Configuration
- **Folds**: {cv_config['n_folds']}
- **Repeats**: {cv_config['n_repeats']}
- **Total Evaluations**: {cv_config['total_evaluations']}

## Performance Metrics

### Overall Performance
- **Accuracy**: {metrics['accuracy']['mean']:.2%} ± {metrics['accuracy']['std']:.2%}
  - Range: [{metrics['accuracy']['min']:.2%}, {metrics['accuracy']['max']:.2%}]
- **Precision (Harmful)**: {metrics['precision_harmful']['mean']:.3f} ± {metrics['precision_harmful']['std']:.3f}
- **Recall (Harmful)**: {metrics['recall_harmful']['mean']:.3f} ± {metrics['recall_harmful']['std']:.3f}
- **F1 Score (Harmful)**: {metrics['f1_harmful']['mean']:.3f} ± {metrics['f1_harmful']['std']:.3f}

### Confusion Matrix (Aggregated)
```
              Predicted
              Safe    Harmful
Actual Safe   {self.results['aggregate_confusion_matrix']['tn']:4d}    {self.results['aggregate_confusion_matrix']['fp']:4d}
     Harmful  {self.results['aggregate_confusion_matrix']['fn']:4d}    {self.results['aggregate_confusion_matrix']['tp']:4d}
```

## Error Analysis
- **Total Errors**: {errors['total']}
- **Error Rate**: {errors['total'] / (cv_config['total_evaluations'] * dataset_stats['total'] / cv_config['n_folds']):.2%}
- **False Positives**: {errors['by_type']['false_positives']} (Safe incorrectly marked as Harmful)
- **False Negatives**: {errors['by_type']['false_negatives']} (Harmful incorrectly marked as Safe)

## Sample Errors
"""
        
        # Add sample errors if available
        if errors.get('samples'):
            report += "\n### False Positives\n"
            fps = [e for e in errors['samples'] if e['true'] == '0'][:3]
            for i, error in enumerate(fps, 1):
                report += f"\n{i}. **Input**: {error['input']}\n"
                report += f"   - Fold: {error.get('fold', 'N/A')}, Repeat: {error.get('repeat', 'N/A')}\n"
            
            report += "\n### False Negatives\n"
            fns = [e for e in errors['samples'] if e['true'] == '1'][:3]
            for i, error in enumerate(fns, 1):
                report += f"\n{i}. **Input**: {error['input']}\n"
                report += f"   - Fold: {error.get('fold', 'N/A')}, Repeat: {error.get('repeat', 'N/A')}\n"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Markdown report saved to {filepath}")
    
    def save_visualizations(self, output_dir: Path):
        """Generate and save visualizations"""
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Confusion Matrix
        ax1 = plt.subplot(2, 3, 1)
        self._plot_confusion_matrix(ax1)
        
        # 2. Metrics Summary
        ax2 = plt.subplot(2, 3, 2)
        self._plot_metrics_summary(ax2)
        
        # 3. Performance Across Folds
        ax3 = plt.subplot(2, 3, 3)
        self._plot_fold_performance(ax3)
        
        # 4. Error Distribution
        ax4 = plt.subplot(2, 3, 4)
        self._plot_error_distribution(ax4)
        
        # 5. Metrics Distribution
        ax5 = plt.subplot(2, 3, 5)
        self._plot_metrics_distribution(ax5)
        
        # 6. Summary Statistics
        ax6 = plt.subplot(2, 3, 6)
        self._plot_summary_stats(ax6)
        
        plt.suptitle('Accommodation Speech Detection - Cross-Validation Results', fontsize=16)
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / "evaluation_results.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_path}")
    
    def _plot_confusion_matrix(self, ax):
        """Plot confusion matrix"""
        cm = self.results['aggregate_confusion_matrix']
        matrix = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
        
        im = ax.imshow(matrix, cmap='Blues', aspect='auto')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, f'{matrix[i, j]}', 
                              ha='center', va='center',
                              color='white' if matrix[i, j] > matrix.max()/2 else 'black',
                              fontsize=14, weight='bold')
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Predicted\nSafe', 'Predicted\nHarmful'])
        ax.set_yticklabels(['Actual\nSafe', 'Actual\nHarmful'])
        ax.set_title('Aggregate Confusion Matrix')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def _plot_metrics_summary(self, ax):
        """Plot metrics summary bars"""
        metrics = self.results['aggregate_metrics']
        
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        means = [
            metrics['accuracy']['mean'],
            metrics['precision_harmful']['mean'],
            metrics['recall_harmful']['mean'],
            metrics['f1_harmful']['mean']
        ]
        stds = [
            metrics['accuracy']['std'],
            metrics['precision_harmful']['std'],
            metrics['recall_harmful']['std'],
            metrics['f1_harmful']['std']
        ]
        
        x = np.arange(len(metric_names))
        bars = ax.bar(x, means, yerr=stds, capsize=5, 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        
        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel('Score')
        ax.set_title('Average Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_fold_performance(self, ax):
        """Plot performance across folds"""
        results = self.results['detailed_results']
        
        folds = [f"R{r['repeat']}-F{r['fold']}" for r in results]
        accuracies = [r['metrics']['accuracy'] for r in results]
        
        x = np.arange(len(folds))
        ax.plot(x, accuracies, 'o-', linewidth=2, markersize=8)
        
        # Add mean line
        mean_acc = np.mean(accuracies)
        ax.axhline(mean_acc, color='red', linestyle='--', alpha=0.5, 
                  label=f'Mean: {mean_acc:.3f}')
        
        ax.set_xlabel('Fold')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Across All Folds')
        ax.set_xticks(x)
        ax.set_xticklabels(folds, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
    
    def _plot_error_distribution(self, ax):
        """Plot error distribution"""
        errors = self.results['errors']
        
        if errors['total'] > 0:
            labels = ['False Positives', 'False Negatives']
            sizes = [errors['by_type']['false_positives'], 
                    errors['by_type']['false_negatives']]
            colors = ['#ff9999', '#66b3ff']
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                              autopct='%1.0f%%', startangle=90)
            ax.set_title(f"Error Distribution\n(Total: {errors['total']} errors)")
        else:
            ax.text(0.5, 0.5, 'No Errors!', ha='center', va='center',
                   fontsize=20, color='green', weight='bold')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
    
    def _plot_metrics_distribution(self, ax):
        """Plot distribution of metrics across folds"""
        results = self.results['detailed_results']
        
        # Extract metrics
        accuracies = [r['metrics']['accuracy'] for r in results]
        precisions = [r['metrics']['precision_harmful'] for r in results]
        recalls = [r['metrics']['recall_harmful'] for r in results]
        f1s = [r['metrics']['f1_harmful'] for r in results]
        
        # Create box plot
        data = [accuracies, precisions, recalls, f1s]
        ax.boxplot(data, labels=['Accuracy', 'Precision', 'Recall', 'F1'])
        ax.set_ylabel('Score')
        ax.set_title('Metrics Distribution Across Folds')
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_summary_stats(self, ax):
        """Plot summary statistics text"""
        ax.axis('off')
        
        summary_text = f"""Key Findings:

✓ High Consistency
  Accuracy std: ±{self.results['aggregate_metrics']['accuracy']['std']:.3f}
  
✓ Strong Performance
  Mean F1: {self.results['aggregate_metrics']['f1_harmful']['mean']:.3f}
  
✓ Error Analysis
  Total errors: {self.results['errors']['total']}
  Error rate: {self.results['errors']['total'] / (self.results['cross_validation']['total_evaluations'] * self.results['dataset_stats']['total'] / self.results['cross_validation']['n_folds']):.2%}
  
✓ Robustness
  {self.results['cross_validation']['total_evaluations']} evaluations
  Stratified folds"""
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))
        ax.set_title('Summary Statistics')