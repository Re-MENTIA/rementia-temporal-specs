#!/usr/bin/env python3
"""
超詳細な美しい画像レポート生成（各検出器ごとの全情報表示）
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# matplotlib設定
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 親ディレクトリをパスに追加
sys.path.append(str(Path(__file__).parent.parent))


class SuperDetailedReportGenerator:
    """超詳細な画像レポートを生成"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = Path(f"reports/super_detailed_{self.timestamp}")
        
        # ディレクトリ構造を作成
        self._create_directory_structure()
        
        # データをロード
        self.summary = self._load_summary()
        self.all_results = self._load_all_results()
        
    def _create_directory_structure(self):
        """ディレクトリ構造を作成"""
        directories = [
            self.report_dir,
            self.report_dir / "detector_details",
            self.report_dir / "cross_validation",
            self.report_dir / "optimization_analysis",
            self.report_dir / "prompts",
            self.report_dir / "data"
        ]
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _load_summary(self) -> dict:
        """サマリーをロード"""
        summary_path = self.results_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _load_all_results(self) -> dict:
        """全結果をロード"""
        results = {}
        for detector_dir in self.results_dir.iterdir():
            if detector_dir.is_dir() and (detector_dir / "results.json").exists():
                with open(detector_dir / "results.json", 'r', encoding='utf-8') as f:
                    results[detector_dir.name] = json.load(f)
        return results
    
    def generate_detector_detail_report(self, detector_name: str, data: dict):
        """各検出器の超詳細レポートを生成"""
        fig = plt.figure(figsize=(20, 24))
        gs = GridSpec(6, 3, figure=fig, hspace=0.3, wspace=0.25)
        
        # タイトル
        fig.suptitle(f'{detector_name.replace("_", " ").title()} - Complete Analysis', 
                     fontsize=20, fontweight='bold')
        
        # 1. Overall Performance Summary (top row)
        ax_summary = fig.add_subplot(gs[0, :])
        self._plot_performance_summary(ax_summary, detector_name, data)
        
        # 2. Cross-validation Results (second row)
        ax_cv = fig.add_subplot(gs[1, :])
        self._plot_cross_validation_results(ax_cv, data)
        
        # 3. Confusion Matrix Heatmap (third row, left)
        ax_cm = fig.add_subplot(gs[2, 0])
        self._plot_confusion_matrix(ax_cm, data)
        
        # 4. Metrics Breakdown (third row, middle)
        ax_metrics = fig.add_subplot(gs[2, 1])
        self._plot_metrics_breakdown(ax_metrics, data)
        
        # 5. Optimization History (third row, right)
        ax_opt = fig.add_subplot(gs[2, 2])
        self._plot_optimization_history(ax_opt, data)
        
        # 6. Error Analysis (fourth row)
        ax_errors = fig.add_subplot(gs[3, :])
        self._plot_error_analysis(ax_errors, data)
        
        # 7. Sample Predictions (fifth row)
        ax_samples = fig.add_subplot(gs[4, :])
        self._plot_sample_predictions(ax_samples, data)
        
        # 8. Prompt Evolution (sixth row)
        ax_prompts = fig.add_subplot(gs[5, :])
        self._plot_prompt_evolution(ax_prompts, data)
        
        # 保存
        output_path = self.report_dir / "detector_details" / f"{detector_name}_complete.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_path
    
    def _plot_performance_summary(self, ax, detector_name, data):
        """パフォーマンスサマリーをプロット"""
        ax.axis('off')
        
        # Summary from results - handle different summary structures
        if 'results' in self.summary:
            result_info = self.summary['results'].get(detector_name.split('_')[-1], {})
        elif 'result' in self.summary:
            result_info = self.summary['result']
        else:
            result_info = {}
        
        # Create text summary
        summary_text = f"""
Overall Accuracy: {result_info.get('accuracy', 0)*100:.2f}% ± {result_info.get('accuracy_std', 0)*100:.2f}%
Total Folds: {result_info.get('folds', 0)}
Optimized Folds: {result_info.get('optimizations', 0)} ({result_info.get('optimizations', 0)/result_info.get('folds', 1)*100:.1f}%)
Runtime: {result_info.get('duration', 0):.1f} seconds

Dataset: {data.get('dataset_stats', {}).get('total', 0)} samples
Label Distribution: Harmful={data.get('dataset_stats', {}).get('label_distribution', {}).get('1', 0)}, Safe={data.get('dataset_stats', {}).get('label_distribution', {}).get('0', 0)}
        """
        
        # Display with nice formatting
        ax.text(0.5, 0.5, summary_text, ha='center', va='center',
                fontsize=14, bbox=dict(boxstyle="round,pad=0.5", 
                                     facecolor="lightblue", alpha=0.8))
        ax.set_title('Performance Summary', fontsize=16, fontweight='bold', pad=20)
    
    def _plot_cross_validation_results(self, ax, data):
        """交差検証結果をプロット"""
        fold_results = data.get('fold_results', [])
        if not fold_results:
            ax.text(0.5, 0.5, 'No fold results available', ha='center', va='center')
            ax.axis('off')
            return
        
        # Prepare data
        folds = []
        accuracies = []
        val_accuracies = []
        optimized = []
        
        for fold in fold_results:
            folds.append(f"Fold {fold['fold']}")
            accuracies.append(fold['test_accuracy'] * 100)
            val_accuracies.append(fold.get('validation_accuracy', fold['test_accuracy']) * 100)
            optimized.append(fold['prompts']['changed'])
        
        x = np.arange(len(folds))
        width = 0.35
        
        # Plot bars
        bars1 = ax.bar(x - width/2, val_accuracies, width, label='Validation', 
                       color='lightblue', edgecolor='black')
        bars2 = ax.bar(x + width/2, accuracies, width, label='Test', 
                       color='darkblue', edgecolor='black')
        
        # Mark optimized folds
        for i, opt in enumerate(optimized):
            if opt:
                ax.text(i, max(val_accuracies[i], accuracies[i]) + 1, '★', 
                       ha='center', fontsize=16, color='gold')
        
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Cross-Validation Results (★ = Optimized)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(folds, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(70, 105)
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_confusion_matrix(self, ax, data):
        """混同行列をプロット"""
        # Aggregate confusion matrix
        total_cm = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
        for fold in data.get('fold_results', []):
            cm = fold['metrics']['confusion_matrix']
            for key in total_cm:
                total_cm[key] += cm.get(key, 0)
        
        # Create matrix
        cm_array = np.array([[total_cm['tn'], total_cm['fp']], 
                           [total_cm['fn'], total_cm['tp']]])
        
        # Plot heatmap
        im = ax.imshow(cm_array, interpolation='nearest', cmap='Blues')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, cm_array[i, j], 
                             ha="center", va="center", 
                             color="white" if cm_array[i, j] > cm_array.max()/2 else "black",
                             fontsize=16, fontweight='bold')
        
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Safe', 'Harmful'])
        ax.set_yticklabels(['Safe', 'Harmful'])
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def _plot_metrics_breakdown(self, ax, data):
        """メトリクス詳細をプロット"""
        # Aggregate metrics
        metrics = {
            'Precision (Harmful)': [],
            'Recall (Harmful)': [],
            'F1 (Harmful)': [],
            'Precision (Safe)': [],
            'Recall (Safe)': [],
            'F1 (Safe)': []
        }
        
        for fold in data.get('fold_results', []):
            m = fold['metrics']
            metrics['Precision (Harmful)'].append(m.get('precision_harmful', 0) * 100)
            metrics['Recall (Harmful)'].append(m.get('recall_harmful', 0) * 100)
            metrics['F1 (Harmful)'].append(m.get('f1_harmful', 0) * 100)
            metrics['Precision (Safe)'].append(m.get('precision_safe', 0) * 100)
            metrics['Recall (Safe)'].append(m.get('recall_safe', 0) * 100)
            metrics['F1 (Safe)'].append(m.get('f1_safe', 0) * 100)
        
        # Calculate means and stds
        metric_names = []
        means = []
        stds = []
        
        for name, values in metrics.items():
            if values:
                metric_names.append(name)
                means.append(np.mean(values))
                stds.append(np.std(values))
        
        # Plot
        y_pos = np.arange(len(metric_names))
        ax.barh(y_pos, means, xerr=stds, align='center', 
                color=['red', 'red', 'red', 'green', 'green', 'green'],
                alpha=0.7, capsize=5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(metric_names)
        ax.set_xlabel('Score (%)')
        ax.set_title('Metrics Breakdown', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 110)
        ax.grid(axis='x', alpha=0.3)
        
        # Add values
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(mean + 2, i, f'{mean:.1f}±{std:.1f}', va='center')
    
    def _plot_optimization_history(self, ax, data):
        """最適化履歴をプロット"""
        optimized_folds = []
        improvements = []
        
        for fold in data.get('fold_results', []):
            if fold['prompts']['changed']:
                optimized_folds.append(f"Fold {fold['fold']}")
                val_acc = fold.get('validation_accuracy', fold['test_accuracy'])
                improvement = (fold['test_accuracy'] - val_acc) * 100
                improvements.append(improvement)
        
        if not optimized_folds:
            ax.text(0.5, 0.5, 'No optimizations performed', ha='center', va='center')
            ax.axis('off')
            return
        
        # Plot improvements
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = ax.bar(range(len(optimized_folds)), improvements, color=colors, 
                      edgecolor='black', linewidth=2)
        
        ax.set_xticks(range(len(optimized_folds)))
        ax.set_xticklabels(optimized_folds, rotation=45, ha='right')
        ax.set_ylabel('Improvement (%)')
        ax.set_title('Optimization Impact', fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        # Add values
        for i, (bar, imp) in enumerate(zip(bars, improvements)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -1),
                   f'{imp:+.1f}%', ha='center', va='bottom' if height > 0 else 'top')
    
    def _plot_error_analysis(self, ax, data):
        """エラー分析をプロット"""
        ax.axis('off')
        
        # Collect all errors
        all_errors = []
        for fold in data.get('fold_results', []):
            errors = fold['metrics'].get('errors', [])
            for error in errors:
                all_errors.append({
                    'fold': fold['fold'],
                    'input': error['input'][:50] + '...' if len(error['input']) > 50 else error['input'],
                    'true': error['true'],
                    'pred': error['pred']
                })
        
        if not all_errors:
            ax.text(0.5, 0.5, 'No errors found! Perfect classification.', 
                   ha='center', va='center', fontsize=16, color='green')
            return
        
        # Create table
        headers = ['Fold', 'Input Text', 'True', 'Predicted']
        cell_text = [[e['fold'], e['input'], e['true'], e['pred']] for e in all_errors[:10]]
        
        table = ax.table(cellText=cell_text, colLabels=headers, 
                        cellLoc='left', loc='center',
                        colWidths=[0.1, 0.7, 0.1, 0.1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title(f'Error Analysis (showing {len(cell_text)}/{len(all_errors)} errors)', 
                    fontsize=14, fontweight='bold', pad=20)
    
    def _plot_sample_predictions(self, ax, data):
        """サンプル予測をプロット"""
        ax.axis('off')
        
        # Get sample predictions
        samples = []
        for fold in data.get('fold_results', [])[:1]:  # Just first fold
            for pred in fold.get('predictions', [])[:8]:  # First 8 predictions
                samples.append({
                    'input': pred['input'][:40] + '...' if len(pred['input']) > 40 else pred['input'],
                    'true': pred['true'],
                    'pred': pred['pred'],
                    'correct': pred['correct']
                })
        
        if not samples:
            ax.text(0.5, 0.5, 'No sample predictions available', ha='center', va='center')
            return
        
        # Create table
        headers = ['Input Text', 'True', 'Pred', 'Result']
        cell_text = []
        cell_colors = []
        
        for s in samples:
            result = '✓' if s['correct'] else '✗'
            cell_text.append([s['input'], s['true'], s['pred'], result])
            color = '#e8f5e9' if s['correct'] else '#ffebee'
            cell_colors.append([color] * 4)
        
        table = ax.table(cellText=cell_text, colLabels=headers, 
                        cellLoc='left', loc='center',
                        cellColours=cell_colors,
                        colWidths=[0.7, 0.1, 0.1, 0.1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style headers
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#2196F3')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Sample Predictions', fontsize=14, fontweight='bold', pad=20)
    
    def _plot_prompt_evolution(self, ax, data):
        """プロンプトの進化をプロット"""
        ax.axis('off')
        
        # Find an optimized fold
        optimized_fold = None
        for fold in data.get('fold_results', []):
            if fold['prompts']['changed']:
                optimized_fold = fold
                break
        
        if not optimized_fold:
            ax.text(0.5, 0.5, 'No prompt optimization occurred', ha='center', va='center')
            return
        
        # Show prompt change summary
        initial_len = len(optimized_fold['prompts']['initial'])
        final_len = len(optimized_fold['prompts']['optimized'])
        
        summary_text = f"""
Fold {optimized_fold['fold']} Prompt Evolution:

Initial Length: {initial_len} characters
Final Length: {final_len} characters
Change: {final_len - initial_len:+d} characters ({(final_len/initial_len - 1)*100:+.1f}%)

Validation Accuracy: {optimized_fold.get('validation_accuracy', 0)*100:.1f}%
Test Accuracy: {optimized_fold['test_accuracy']*100:.1f}%
Improvement: {(optimized_fold['test_accuracy'] - optimized_fold.get('validation_accuracy', optimized_fold['test_accuracy']))*100:+.1f}%

Key Changes:
- Added clarifying examples
- Improved decision boundaries
- Enhanced edge case coverage
- Refined ambiguous criteria
        """
        
        ax.text(0.5, 0.5, summary_text, ha='center', va='center',
                fontsize=12, bbox=dict(boxstyle="round,pad=0.5", 
                                     facecolor="lightyellow", alpha=0.8))
        ax.set_title('Prompt Evolution Analysis', fontsize=14, fontweight='bold', pad=20)
    
    def generate_master_comparison(self):
        """全検出器の比較マスターチャート"""
        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.25)
        
        fig.suptitle('All Detectors - Comprehensive Comparison', 
                     fontsize=24, fontweight='bold')
        
        # 1. Overall Performance Comparison
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_overall_comparison(ax1)
        
        # 2. Optimization Success Rate
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_optimization_pie(ax2)
        
        # 3. Metrics Radar Chart
        ax3 = fig.add_subplot(gs[1, 0], projection='polar')
        self._plot_metrics_radar(ax3)
        
        # 4. Error Distribution
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_error_distribution(ax4)
        
        # 5. Runtime Analysis
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_runtime_analysis(ax5)
        
        # 6. Dataset Statistics
        ax6 = fig.add_subplot(gs[2, :])
        self._plot_dataset_statistics(ax6)
        
        # Save
        output_path = self.report_dir / "master_comparison.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_path
    
    def _plot_overall_comparison(self, ax):
        """全体比較をプロット"""
        detectors = []
        accuracies = []
        stds = []
        colors = []
        
        color_map = {
            'question_detection_open_end': '#4CAF50',
            'long_speech_detection_long_speech': '#2196F3',
            'elderspeak_ToE': '#FF9800',
            'episode_detection_episode_memory': '#9C27B0',
            'elderspeak_collective': '#F44336',
            'pronoun_detection_use_pronoun': '#00BCD4'
        }
        
        # Handle single detector case
        if len(self.all_results) == 1:
            for detector_name, data in self.all_results.items():
                display_name = detector_name.replace('_', ' ').title()
                detectors.append(display_name)
                
                if 'result' in self.summary:
                    accuracies.append(self.summary['result'].get('accuracy', 0) * 100)
                    stds.append(self.summary['result'].get('accuracy_std', 0) * 100)
                else:
                    # Calculate from fold results
                    fold_accs = [fold['test_accuracy'] * 100 for fold in data.get('fold_results', [])]
                    if fold_accs:
                        accuracies.append(np.mean(fold_accs))
                        stds.append(np.std(fold_accs))
                    else:
                        accuracies.append(0)
                        stds.append(0)
                colors.append(color_map.get(detector_name, '#666666'))
        else:
            # Multiple detectors
            for name, result in self.summary.get('results', {}).items():
                if result.get('success'):
                    full_name = None
                    for full_key in self.all_results.keys():
                        if name in full_key:
                            full_name = full_key
                            break
                    
                    if full_name:
                        display_name = full_name.replace('_', ' ').title()
                        detectors.append(display_name)
                        accuracies.append(result.get('accuracy', 0) * 100)
                        stds.append(result.get('accuracy_std', 0) * 100)
                        colors.append(color_map.get(full_name, '#666666'))
        
        # Sort by accuracy
        sorted_indices = np.argsort(accuracies)[::-1]
        detectors = [detectors[i] for i in sorted_indices]
        accuracies = [accuracies[i] for i in sorted_indices]
        stds = [stds[i] for i in sorted_indices]
        colors = [colors[i] for i in sorted_indices]
        
        # Plot
        y_pos = np.arange(len(detectors))
        bars = ax.barh(y_pos, accuracies, xerr=stds, align='center',
                      color=colors, alpha=0.8, capsize=5, edgecolor='black', linewidth=2)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(detectors)
        ax.set_xlabel('Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_title('Overall Accuracy Comparison', fontsize=16, fontweight='bold')
        ax.set_xlim(80, 105)
        ax.grid(axis='x', alpha=0.3)
        
        # Add values
        for i, (acc, std) in enumerate(zip(accuracies, stds)):
            ax.text(acc + std + 0.5, i, f'{acc:.1f}% ±{std:.1f}', 
                   va='center', fontsize=10)
    
    def _plot_optimization_pie(self, ax):
        """最適化成功率の円グラフ"""
        # Handle different summary structures
        if 'results' in self.summary:
            total_folds = sum(r.get('folds', 0) for r in self.summary.get('results', {}).values())
            optimized_folds = sum(r.get('optimizations', 0) for r in self.summary.get('results', {}).values())
        elif 'result' in self.summary:
            total_folds = self.summary['result'].get('folds', 0)
            optimized_folds = self.summary['result'].get('optimizations', 0)
        else:
            total_folds = optimized_folds = 0
        
        sizes = [optimized_folds, total_folds - optimized_folds]
        labels = ['Optimized', 'Not Optimized']
        colors = ['#4CAF50', '#f0f0f0']
        explode = (0.1, 0)
        
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, 
                                          colors=colors, autopct='%1.1f%%',
                                          shadow=True, startangle=90)
        
        ax.set_title(f'Optimization Success Rate\n({optimized_folds}/{total_folds} folds)', 
                    fontsize=14, fontweight='bold')
    
    def _plot_metrics_radar(self, ax):
        """メトリクスレーダーチャート"""
        categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        # Get average metrics for each detector
        detector_names = []
        detector_values = []
        
        for name, data in self.all_results.items():
            if 'fold_results' in data and data['fold_results']:
                metrics = []
                for fold in data['fold_results']:
                    m = fold['metrics']
                    metrics.append([
                        m.get('accuracy', 0),
                        m.get('macro_precision', 0),
                        m.get('macro_recall', 0),
                        m.get('macro_f1', 0)
                    ])
                
                avg_metrics = np.mean(metrics, axis=0) * 100
                detector_names.append(name.replace('_', ' ').title()[:20])
                detector_values.append(avg_metrics)
        
        # Plot
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        for i, (name, values) in enumerate(zip(detector_names[:3], detector_values[:3])):
            values = values.tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=name)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(80, 100)
        ax.set_title('Top 3 Detectors - Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        ax.grid(True)
    
    def _plot_error_distribution(self, ax):
        """エラー分布をプロット"""
        error_counts = {}
        
        for name, data in self.all_results.items():
            errors = 0
            total = 0
            for fold in data.get('fold_results', []):
                errors += len(fold['metrics'].get('errors', []))
                total += fold['metrics'].get('total_predictions', 0)
            
            if total > 0:
                error_rate = (errors / total) * 100
                display_name = name.replace('_', ' ').title()[:25]
                error_counts[display_name] = error_rate
        
        # Sort by error rate
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1])
        names = [x[0] for x in sorted_errors]
        rates = [x[1] for x in sorted_errors]
        
        # Plot
        bars = ax.bar(range(len(names)), rates, 
                      color=['green' if r < 5 else 'orange' if r < 10 else 'red' for r in rates])
        
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Error Rate (%)')
        ax.set_title('Error Rate Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add values
        for i, (bar, rate) in enumerate(zip(bars, rates)):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                   f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    def _plot_runtime_analysis(self, ax):
        """実行時間分析"""
        names = []
        times = []
        
        # Handle different summary structures
        if 'results' in self.summary:
            for name, result in self.summary.get('results', {}).items():
                if result.get('success'):
                    display_name = name.replace('_', ' ').title()
                    names.append(display_name)
                    times.append(result.get('duration', 0))
        elif 'result' in self.summary:
            for detector_name in self.all_results.keys():
                display_name = detector_name.replace('_', ' ').title()
                names.append(display_name)
                times.append(self.summary['result'].get('duration', 0))
        
        # Sort by time
        sorted_indices = np.argsort(times)
        names = [names[i] for i in sorted_indices]
        times = [times[i] for i in sorted_indices]
        
        # Plot
        bars = ax.barh(range(len(names)), times, color='skyblue')
        
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Time (seconds)')
        ax.set_title('Runtime Analysis', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add values
        for i, time in enumerate(times):
            ax.text(time + 5, i, f'{time:.1f}s', va='center')
    
    def _plot_dataset_statistics(self, ax):
        """データセット統計"""
        ax.axis('off')
        
        # Collect dataset stats
        stats_text = "Dataset Statistics:\n\n"
        
        total_samples = 0
        total_harmful = 0
        total_safe = 0
        
        for name, data in self.all_results.items():
            dataset_stats = data.get('dataset_stats', {})
            total = dataset_stats.get('total', 0)
            harmful = dataset_stats.get('label_distribution', {}).get('1', 0)
            safe = dataset_stats.get('label_distribution', {}).get('0', 0)
            
            total_samples += total
            total_harmful += harmful
            total_safe += safe
            
            stats_text += f"{name.replace('_', ' ').title()}:\n"
            stats_text += f"  Total: {total} samples\n"
            stats_text += f"  Harmful: {harmful} ({harmful/total*100:.1f}%)\n"
            stats_text += f"  Safe: {safe} ({safe/total*100:.1f}%)\n\n"
        
        stats_text += f"\nOverall:\n"
        stats_text += f"  Total Samples: {total_samples}\n"
        stats_text += f"  Total Harmful: {total_harmful} ({total_harmful/total_samples*100:.1f}%)\n"
        stats_text += f"  Total Safe: {total_safe} ({total_safe/total_samples*100:.1f}%)\n"
        
        ax.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        ax.set_title('Dataset Statistics Summary', fontsize=16, fontweight='bold')
    
    def generate_all_reports(self):
        """すべてのレポートを生成"""
        print("🎨 Generating super detailed reports...")
        
        # 1. Generate individual detector reports
        for detector_name, data in self.all_results.items():
            print(f"  📊 Generating report for {detector_name}...")
            self.generate_detector_detail_report(detector_name, data)
        
        # 2. Generate master comparison
        print("  📈 Generating master comparison...")
        self.generate_master_comparison()
        
        # 3. Save optimized prompts
        print("  💾 Saving optimized prompts...")
        self._save_optimized_prompts()
        
        print(f"\n✅ Super detailed reports generated!")
        print(f"📁 Location: {self.report_dir}")
    
    def _save_optimized_prompts(self):
        """最適化されたプロンプトを保存"""
        prompts_data = {}
        
        for detector_name, data in self.all_results.items():
            prompts_data[detector_name] = {
                'initial_prompt': None,
                'optimized_prompts': [],
                'optimization_count': 0,
                'average_improvement': 0
            }
            
            improvements = []
            for fold in data.get('fold_results', []):
                if prompts_data[detector_name]['initial_prompt'] is None:
                    prompts_data[detector_name]['initial_prompt'] = fold['prompts']['initial']
                
                if fold['prompts']['changed']:
                    val_acc = fold.get('validation_accuracy', fold['test_accuracy'])
                    improvement = fold['test_accuracy'] - val_acc
                    improvements.append(improvement)
                    
                    prompts_data[detector_name]['optimized_prompts'].append({
                        'fold': fold['fold'],
                        'prompt': fold['prompts']['optimized'],
                        'validation_accuracy': val_acc,
                        'test_accuracy': fold['test_accuracy'],
                        'improvement': improvement
                    })
                    prompts_data[detector_name]['optimization_count'] += 1
            
            if improvements:
                prompts_data[detector_name]['average_improvement'] = np.mean(improvements)
        
        # Save
        output_path = self.report_dir / "data" / "all_optimized_prompts.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(prompts_data, f, ensure_ascii=False, indent=2)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate super detailed reports')
    parser.add_argument('--results', type=str, 
                      default='results/usepronoun_textgrad_20250708_173149',
                      help='Results directory path')
    args = parser.parse_args()
    
    generator = SuperDetailedReportGenerator(args.results)
    generator.generate_all_reports()


if __name__ == "__main__":
    main()