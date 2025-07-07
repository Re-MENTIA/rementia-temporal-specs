"""
Unified reporting utilities for multiple detection tasks
Supports both accommodation speech and episode detection with full visualization
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class UnifiedReportGenerator:
    """Generate comprehensive reports for any detection task"""
    
    def __init__(self, results: Dict):
        """Initialize with evaluation results"""
        self.results = results
        self.timestamp = datetime.now()
        self.task = results.get('task', 'unknown')
        
        # Set matplotlib style
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['font.size'] = 10
    
    def generate_all_reports(self, output_dir: str = "results"):
        """Generate all report types"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Generate different report types
        self.save_json_report(output_path / "detailed_results.json")
        self.save_markdown_report(output_path / "evaluation_report.md")
        self.save_visualizations(output_path / "evaluation_results.png")
        self.save_error_analysis(output_path / "error_analysis.md")
        
        logger.info(f"All reports generated in {output_dir}")
    
    def save_json_report(self, filepath: Path):
        """Save detailed JSON report"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON report saved to {filepath}")
    
    def save_markdown_report(self, filepath: Path):
        """Generate and save markdown report"""
        report = f"""# {self.task.replace('_', ' ').title()} - Evaluation Report

Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Statistics
"""
        
        # Dataset stats
        dataset_stats = self.results.get('dataset_stats', {})
        report += f"- **Total Samples**: {dataset_stats.get('total', 'N/A')}\n"
        
        # Label distribution
        if 'label_distribution' in dataset_stats:
            report += "\nLabel Distribution:\n"
            for label, count in dataset_stats['label_distribution'].items():
                percentage = (count / dataset_stats['total'] * 100) if dataset_stats['total'] else 0
                report += f"- **{label}**: {count} ({percentage:.1f}%)\n"
        
        # Cross-validation config
        cv_config = self.results.get('cross_validation_config', {})
        if cv_config:
            report += f"""
## Cross-Validation Configuration
- **Folds**: {cv_config.get('n_folds', 'N/A')}
- **Repeats**: {cv_config.get('n_repeats', 'N/A')}
- **Total Evaluations**: {cv_config.get('n_folds', 1) * cv_config.get('n_repeats', 1)}
- **Validation Split**: {cv_config.get('validation_split_ratio', 0) * 100:.0f}%
- **Stratified**: {cv_config.get('stratified', False)}
"""
        
        # Performance metrics
        report += "\n## Performance Metrics\n\n"
        
        if 'aggregate_metrics' in self.results:
            report += self._format_aggregate_metrics()
        elif 'metrics' in self.results:
            report += self._format_single_metrics()
        
        # Error analysis
        errors = self.results.get('errors', {})
        if errors:
            report += f"""
## Error Analysis
- **Total Errors**: {errors.get('total', 0)}
- **Error Rate**: {errors.get('error_rate', 0):.2%}
"""
            if 'by_type' in errors:
                report += "\nError Breakdown:\n"
                for error_type, count in errors['by_type'].items():
                    report += f"- **{error_type}**: {count}\n"
        
        # Save report
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Markdown report saved to {filepath}")
    
    def _format_aggregate_metrics(self) -> str:
        """Format aggregate metrics for cross-validation"""
        agg_metrics = self.results['aggregate_metrics']
        report = "### Overall Performance (Cross-Validation)\n"
        
        # Accuracy
        if 'accuracy' in agg_metrics:
            acc = agg_metrics['accuracy']
            report += f"- **Accuracy**: {acc['mean']:.2%} ± {acc['std']:.2%}\n"
            report += f"  - Range: [{acc['min']:.2%}, {acc['max']:.2%}]\n\n"
        
        # Task-specific metrics
        if 'precision_harmful' in agg_metrics:  # Accommodation speech
            report += "### Harmful Class Metrics\n"
            for metric in ['precision_harmful', 'recall_harmful', 'f1_harmful']:
                if metric in agg_metrics:
                    m = agg_metrics[metric]
                    name = metric.replace('_harmful', '').title()
                    report += f"- **{name}**: {m['mean']:.3f} ± {m['std']:.3f}\n"
        
        elif 'precision_1' in agg_metrics:  # Episode detection
            report += "### Episodic Memory Detection Metrics\n"
            for metric in ['precision_1', 'recall_1', 'f1_1']:
                if metric in agg_metrics:
                    m = agg_metrics[metric]
                    name = metric.replace('_1', '').title()
                    report += f"- **{name}**: {m['mean']:.3f} ± {m['std']:.3f}\n"
            
            report += "\n### Non-Episodic Class Metrics\n"
            for metric in ['precision_0', 'recall_0', 'f1_0']:
                if metric in agg_metrics:
                    m = agg_metrics[metric]
                    name = metric.replace('_0', '').title()
                    report += f"- **{name}**: {m['mean']:.3f} ± {m['std']:.3f}\n"
        
        return report
    
    def _format_single_metrics(self) -> str:
        """Format single evaluation metrics"""
        metrics = self.results['metrics']
        report = "### Performance Metrics\n"
        
        if 'accuracy' in metrics:
            report += f"- **Accuracy**: {metrics['accuracy']:.2%}\n"
        
        # Check for different metric formats
        if 'class_metrics' in metrics:
            # Multi-class metrics
            report += "\n### Per-Class Performance\n"
            for label, class_metrics in metrics['class_metrics'].items():
                report += f"\n**{label.upper()}**:\n"
                report += f"- Precision: {class_metrics['precision']:.3f}\n"
                report += f"- Recall: {class_metrics['recall']:.3f}\n"
                report += f"- F1 Score: {class_metrics['f1']:.3f}\n"
                report += f"- Support: {class_metrics['support']}\n"
        else:
            # Binary metrics
            for key in ['precision_harmful', 'recall_harmful', 'f1_harmful',
                       'precision_1', 'recall_1', 'f1_1']:
                if key in metrics:
                    name = key.replace('_harmful', ' (Harmful)').replace('_1', ' (Episodic)')
                    report += f"- **{name.replace('_', ' ').title()}**: {metrics[key]:.3f}\n"
        
        return report
    
    def save_error_analysis(self, filepath: Path):
        """Save detailed error analysis"""
        errors = self.results.get('errors', {})
        if not errors or errors.get('total', 0) == 0:
            return
        
        report = f"""# Error Analysis - {self.task.replace('_', ' ').title()}

Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Total Errors**: {errors.get('total', 0)}
- **Error Rate**: {errors.get('error_rate', 0):.2%}

## Error Breakdown
"""
        
        # Error types
        if 'by_type' in errors:
            for error_type, count in errors['by_type'].items():
                report += f"- **{error_type}**: {count}\n"
        
        # Sample errors
        if 'sample_errors' in errors:
            report += "\n## Sample Errors\n\n"
            
            # Group by error type
            error_groups = {}
            for error in errors['sample_errors']:
                true_label = error['true_label']
                pred_labels = error.get('predictions', {})
                
                # Find most common wrong prediction
                for pred, count in pred_labels.items():
                    if pred != true_label:
                        key = f"{true_label}_as_{pred}"
                        if key not in error_groups:
                            error_groups[key] = []
                        error_groups[key].append({
                            'text': error['text'],
                            'count': count
                        })
            
            # Show errors by type
            for error_type, examples in error_groups.items():
                report += f"\n### {error_type.replace('_', ' ').title()}\n\n"
                for i, example in enumerate(examples[:5], 1):  # Show top 5
                    report += f"{i}. **Text**: \"{example['text']}\"\n"
                    report += f"   - Misclassified {example['count']} times\n\n"
        
        # Save report
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Error analysis saved to {filepath}")
    
    def save_visualizations(self, filepath: Path):
        """Generate and save comprehensive visualizations"""
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Title
        task_name = self.task.replace('_', ' ').title()
        fig.suptitle(f'{task_name} - Cross-Validation Results', fontsize=20, y=0.98)
        
        # Layout depends on whether we have cross-validation results
        if 'fold_results' in self.results:
            # Cross-validation layout
            # Row 1: Confusion matrix, metrics summary, fold performance
            ax1 = plt.subplot(3, 3, 1)
            self._plot_aggregate_confusion_matrix(ax1)
            
            ax2 = plt.subplot(3, 3, 2)
            self._plot_metrics_summary(ax2)
            
            ax3 = plt.subplot(3, 3, 3)
            self._plot_fold_performance(ax3)
            
            # Row 2: Error distribution, metrics distribution, prompt changes
            ax4 = plt.subplot(3, 3, 4)
            self._plot_error_distribution(ax4)
            
            ax5 = plt.subplot(3, 3, 5)
            self._plot_metrics_distribution(ax5)
            
            ax6 = plt.subplot(3, 3, 6)
            self._plot_prompt_changes(ax6)
            
            # Row 3: Summary stats, label distribution, performance heatmap
            ax7 = plt.subplot(3, 3, 7)
            self._plot_summary_stats(ax7)
            
            ax8 = plt.subplot(3, 3, 8)
            self._plot_label_distribution(ax8)
            
            ax9 = plt.subplot(3, 3, 9)
            self._plot_performance_heatmap(ax9)
        else:
            # Single evaluation layout
            ax1 = plt.subplot(2, 2, 1)
            self._plot_single_confusion_matrix(ax1)
            
            ax2 = plt.subplot(2, 2, 2)
            self._plot_single_metrics(ax2)
            
            ax3 = plt.subplot(2, 2, 3)
            self._plot_error_distribution(ax3)
            
            ax4 = plt.subplot(2, 2, 4)
            self._plot_summary_stats(ax4)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {filepath}")
    
    def _plot_aggregate_confusion_matrix(self, ax):
        """Plot aggregate confusion matrix for cross-validation"""
        # Calculate aggregate confusion matrix from fold results
        cm = defaultdict(int)
        
        for fold_result in self.results.get('fold_results', []):
            fold_cm = fold_result['metrics'].get('confusion_matrix', {})
            
            # Handle different confusion matrix formats
            if 'tp' in fold_cm:
                # Binary format
                cm['tp'] += fold_cm.get('tp', 0)
                cm['tn'] += fold_cm.get('tn', 0)
                cm['fp'] += fold_cm.get('fp', 0)
                cm['fn'] += fold_cm.get('fn', 0)
            else:
                # String key format
                for key, value in fold_cm.items():
                    cm[key] += value
        
        if 'tp' in cm:
            # Binary confusion matrix
            matrix = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
            labels = ['Safe/Non-Episodic', 'Harmful/Episodic']
        else:
            # Convert string keys to matrix
            # This is more complex for multi-class, simplified for now
            matrix = np.array([[cm.get('0->0', 0), cm.get('0->1', 0)],
                              [cm.get('1->0', 0), cm.get('1->1', 0)]])
            labels = ['Class 0', 'Class 1']
        
        im = ax.imshow(matrix, cmap='Blues', aspect='auto')
        
        # Add text annotations
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                text = ax.text(j, i, f'{matrix[i, j]}', 
                              ha='center', va='center',
                              color='white' if matrix[i, j] > matrix.max()/2 else 'black',
                              fontsize=14, weight='bold')
        
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels([f'Pred\n{l}' for l in labels])
        ax.set_yticklabels([f'True\n{l}' for l in labels])
        ax.set_title('Aggregate Confusion Matrix')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def _plot_metrics_summary(self, ax):
        """Plot metrics summary"""
        agg_metrics = self.results.get('aggregate_metrics', {})
        
        # Determine which metrics to show based on task
        if 'precision_harmful' in agg_metrics:
            # Accommodation speech metrics
            metric_keys = ['accuracy', 'precision_harmful', 'recall_harmful', 'f1_harmful']
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        elif 'precision_1' in agg_metrics:
            # Episode detection metrics
            metric_keys = ['accuracy', 'precision_1', 'recall_1', 'f1_1']
            metric_names = ['Accuracy', 'Precision\n(Episodic)', 'Recall\n(Episodic)', 'F1\n(Episodic)']
        else:
            metric_keys = ['accuracy']
            metric_names = ['Accuracy']
        
        means = []
        stds = []
        
        for key in metric_keys:
            if key in agg_metrics:
                means.append(agg_metrics[key]['mean'])
                stds.append(agg_metrics[key]['std'])
            else:
                means.append(0)
                stds.append(0)
        
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
        fold_results = self.results.get('fold_results', [])
        
        if not fold_results:
            ax.text(0.5, 0.5, 'No fold results', ha='center', va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return
        
        folds = [f"R{r.get('repeat', 1)}-F{r['fold']}" for r in fold_results]
        accuracies = [r['test_accuracy'] for r in fold_results]
        
        x = np.arange(len(folds))
        ax.plot(x, accuracies, 'o-', linewidth=2, markersize=8)
        
        # Add mean line
        mean_acc = np.mean(accuracies)
        ax.axhline(mean_acc, color='red', linestyle='--', alpha=0.5, 
                  label=f'Mean: {mean_acc:.3f}')
        
        # Add shaded region for std
        std_acc = np.std(accuracies)
        ax.fill_between([-0.5, len(folds)-0.5], 
                       [mean_acc - std_acc]*2, 
                       [mean_acc + std_acc]*2,
                       alpha=0.2, color='red')
        
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
        errors = self.results.get('errors', {})
        
        if not errors or errors.get('total', 0) == 0:
            ax.text(0.5, 0.5, 'No Errors!', ha='center', va='center',
                   fontsize=20, color='green', weight='bold')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_title('Error Distribution')
            return
        
        # Get error types
        error_types = errors.get('by_type', {})
        if error_types:
            labels = list(error_types.keys())
            sizes = list(error_types.values())
            
            # Use different colors for different error types
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                              autopct='%1.0f%%', startangle=90)
            ax.set_title(f"Error Distribution\n(Total: {errors['total']} errors)")
    
    def _plot_metrics_distribution(self, ax):
        """Plot distribution of metrics across folds"""
        fold_results = self.results.get('fold_results', [])
        
        if not fold_results:
            ax.text(0.5, 0.5, 'No fold results', ha='center', va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return
        
        # Extract metrics based on task
        first_result = fold_results[0]['metrics']
        
        if 'precision_harmful' in first_result:
            # Accommodation speech
            metric_keys = ['accuracy', 'precision_harmful', 'recall_harmful', 'f1_harmful']
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
        elif 'precision_1' in first_result:
            # Episode detection
            metric_keys = ['accuracy', 'precision_1', 'recall_1', 'f1_1']
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
        else:
            metric_keys = ['accuracy']
            metric_names = ['Accuracy']
        
        # Collect data
        data = []
        for key in metric_keys:
            if key == 'accuracy':
                values = [r['test_accuracy'] for r in fold_results]
            else:
                values = [r['metrics'].get(key, 0) for r in fold_results]
            data.append(values)
        
        # Create box plot
        ax.boxplot(data, labels=metric_names)
        ax.set_ylabel('Score')
        ax.set_title('Metrics Distribution Across Folds')
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_prompt_changes(self, ax):
        """Plot prompt optimization information"""
        fold_results = self.results.get('fold_results', [])
        
        # Count prompt changes
        changes = sum(1 for r in fold_results if r.get('prompts', {}).get('changed', False))
        total = len(fold_results)
        
        if changes == 0:
            ax.text(0.5, 0.5, 'No Prompt\nOptimization\nPerformed', 
                   ha='center', va='center', fontsize=16, weight='bold')
        else:
            # Show optimization stats
            labels = ['Optimized', 'Original']
            sizes = [changes, total - changes]
            colors = ['#2ca02c', '#1f77b4']
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                              autopct='%1.0f%%', startangle=90)
            
            # Add validation accuracy info if available
            val_accs = [r.get('validation_accuracy', 0) for r in fold_results 
                       if r.get('prompts', {}).get('changed', False)]
            if val_accs:
                avg_val_acc = np.mean(val_accs)
                ax.text(0, -1.3, f'Avg Val Acc: {avg_val_acc:.2%}', 
                       ha='center', transform=ax.transAxes)
        
        ax.set_title('Prompt Optimization')
    
    def _plot_summary_stats(self, ax):
        """Plot summary statistics"""
        ax.axis('off')
        
        # Prepare summary text based on available data
        if 'aggregate_metrics' in self.results:
            accuracy = self.results['aggregate_metrics']['accuracy']
            summary_text = f"""Key Findings:

✓ High Consistency
  Accuracy std: ±{accuracy['std']:.3f}
  
✓ Strong Performance
  Mean Accuracy: {accuracy['mean']:.3f}
"""
            
            # Add task-specific metrics
            if 'f1_harmful' in self.results['aggregate_metrics']:
                f1 = self.results['aggregate_metrics']['f1_harmful']
                summary_text += f"  Mean F1 (Harmful): {f1['mean']:.3f}\n"
            elif 'f1_1' in self.results['aggregate_metrics']:
                f1 = self.results['aggregate_metrics']['f1_1']
                summary_text += f"  Mean F1 (Episodic): {f1['mean']:.3f}\n"
            
            # Error info
            errors = self.results.get('errors', {})
            summary_text += f"""  
✓ Error Analysis
  Total errors: {errors.get('total', 0)}
  Error rate: {errors.get('error_rate', 0):.2%}
"""
            
            # CV info
            cv_config = self.results.get('cross_validation_config', {})
            if cv_config:
                summary_text += f"""  
✓ Robustness
  {cv_config.get('n_folds', 1) * cv_config.get('n_repeats', 1)} evaluations
  Stratified folds"""
        else:
            # Single evaluation
            metrics = self.results.get('metrics', {})
            summary_text = f"""Results:

Accuracy: {metrics.get('accuracy', 0):.3f}
Total Samples: {self.results.get('dataset_stats', {}).get('total', 'N/A')}
"""
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))
        ax.set_title('Summary Statistics')
    
    def _plot_label_distribution(self, ax):
        """Plot label distribution"""
        dataset_stats = self.results.get('dataset_stats', {})
        label_dist = dataset_stats.get('label_distribution', {})
        
        if not label_dist:
            ax.text(0.5, 0.5, 'No label distribution data', ha='center', va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return
        
        labels = list(label_dist.keys())
        counts = list(label_dist.values())
        
        # Create bar plot
        x = np.arange(len(labels))
        bars = ax.bar(x, counts, color=plt.cm.Set3(np.linspace(0, 1, len(labels))))
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{count}', ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Label')
        ax.set_ylabel('Count')
        ax.set_title('Dataset Label Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_performance_heatmap(self, ax):
        """Plot performance heatmap across folds and metrics"""
        fold_results = self.results.get('fold_results', [])
        
        if not fold_results:
            ax.text(0.5, 0.5, 'No fold results', ha='center', va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return
        
        # Determine metrics to show
        first_result = fold_results[0]['metrics']
        if 'precision_harmful' in first_result:
            metric_keys = ['accuracy', 'precision_harmful', 'recall_harmful', 'f1_harmful']
            metric_names = ['Acc', 'Prec', 'Rec', 'F1']
        elif 'precision_1' in first_result:
            metric_keys = ['accuracy', 'precision_1', 'recall_1', 'f1_1']
            metric_names = ['Acc', 'Prec', 'Rec', 'F1']
        else:
            metric_keys = ['accuracy']
            metric_names = ['Acc']
        
        # Create matrix
        matrix = []
        for result in fold_results:
            row = []
            for key in metric_keys:
                if key == 'accuracy':
                    row.append(result['test_accuracy'])
                else:
                    row.append(result['metrics'].get(key, 0))
            matrix.append(row)
        
        matrix = np.array(matrix).T
        
        # Plot heatmap
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks
        ax.set_xticks(range(len(fold_results)))
        ax.set_xticklabels([f"R{r.get('repeat', 1)}-F{r['fold']}" for r in fold_results], 
                          rotation=45, ha='right')
        ax.set_yticks(range(len(metric_names)))
        ax.set_yticklabels(metric_names)
        
        # Add text annotations
        for i in range(len(metric_names)):
            for j in range(len(fold_results)):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}', 
                              ha='center', va='center',
                              color='white' if matrix[i, j] < 0.5 else 'black',
                              fontsize=8)
        
        ax.set_title('Performance Heatmap')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def _plot_single_confusion_matrix(self, ax):
        """Plot confusion matrix for single evaluation"""
        metrics = self.results.get('metrics', {})
        cm = metrics.get('confusion_matrix', {})
        
        if not cm:
            ax.text(0.5, 0.5, 'No confusion matrix', ha='center', va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return
        
        # Handle different formats
        if 'tp' in cm:
            # Binary format
            matrix = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
            labels = ['Safe/Non-Episodic', 'Harmful/Episodic']
        else:
            # String format - simplified for binary
            matrix = np.array([[cm.get('0->0', 0), cm.get('0->1', 0)],
                              [cm.get('1->0', 0), cm.get('1->1', 0)]])
            labels = ['Class 0', 'Class 1']
        
        im = ax.imshow(matrix, cmap='Blues', aspect='auto')
        
        # Add text
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                text = ax.text(j, i, f'{matrix[i, j]}', 
                              ha='center', va='center',
                              color='white' if matrix[i, j] > matrix.max()/2 else 'black',
                              fontsize=14, weight='bold')
        
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels([f'Pred\n{l}' for l in labels])
        ax.set_yticklabels([f'True\n{l}' for l in labels])
        ax.set_title('Confusion Matrix')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def _plot_single_metrics(self, ax):
        """Plot metrics for single evaluation"""
        metrics = self.results.get('metrics', {})
        
        # Collect available metrics
        metric_data = []
        metric_names = []
        
        if 'accuracy' in metrics:
            metric_data.append(metrics['accuracy'])
            metric_names.append('Accuracy')
        
        # Check for different metric types
        for key, name in [('precision_harmful', 'Precision'),
                         ('recall_harmful', 'Recall'),
                         ('f1_harmful', 'F1'),
                         ('precision_1', 'Precision'),
                         ('recall_1', 'Recall'),
                         ('f1_1', 'F1')]:
            if key in metrics:
                metric_data.append(metrics[key])
                if '1' in key:
                    metric_names.append(f'{name}\n(Episodic)')
                else:
                    metric_names.append(name)
        
        if not metric_data:
            ax.text(0.5, 0.5, 'No metrics', ha='center', va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return
        
        # Plot bars
        x = np.arange(len(metric_names))
        bars = ax.bar(x, metric_data, color=plt.cm.Set3(np.linspace(0, 1, len(metric_names))))
        
        # Add value labels
        for bar, value in zip(bars, metric_data):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)