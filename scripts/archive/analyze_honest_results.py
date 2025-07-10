#!/usr/bin/env python3
"""
Analyze honest evaluation results with REAL TextGrad optimization
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def analyze_results(results_dir: Path):
    """Analyze results from honest evaluation"""
    
    # Find latest results
    results_files = list(results_dir.glob("results/honest_evaluation_*/results.json"))
    if not results_files:
        print("No results found. Evaluation may still be running.")
        return
    
    latest_results = max(results_files, key=lambda p: p.stat().st_mtime)
    print(f"Analyzing: {latest_results}")
    
    with open(latest_results, 'r') as f:
        results = json.load(f)
    
    print("\n" + "="*80)
    print("HONEST TEXTGRAD OPTIMIZATION ANALYSIS")
    print("="*80)
    
    # Analyze optimization effectiveness
    fold_results = results.get('fold_results', [])
    
    optimization_stats = {
        'total_folds': len(fold_results),
        'optimized_folds': 0,
        'improved_folds': 0,
        'improvements': [],
        'no_change': 0,
        'degraded': 0
    }
    
    for fold in fold_results:
        if fold['prompts']['changed']:
            optimization_stats['optimized_folds'] += 1
            
            # Calculate improvement
            val_acc = fold.get('validation_accuracy', fold['test_accuracy'])
            test_acc = fold['test_accuracy']
            improvement = test_acc - val_acc
            
            optimization_stats['improvements'].append(improvement)
            
            if improvement > 0:
                optimization_stats['improved_folds'] += 1
            elif improvement == 0:
                optimization_stats['no_change'] += 1
            else:
                optimization_stats['degraded'] += 1
    
    # Print optimization summary
    print(f"\nOptimization Summary:")
    print(f"  Total folds: {optimization_stats['total_folds']}")
    print(f"  Folds with optimization: {optimization_stats['optimized_folds']} ({optimization_stats['optimized_folds']/optimization_stats['total_folds']*100:.1f}%)")
    
    if optimization_stats['improvements']:
        avg_improvement = sum(optimization_stats['improvements']) / len(optimization_stats['improvements'])
        print(f"\nOptimization Impact (when applied):")
        print(f"  Improved: {optimization_stats['improved_folds']}")
        print(f"  No change: {optimization_stats['no_change']}")
        print(f"  Degraded: {optimization_stats['degraded']}")
        print(f"  Average change: {avg_improvement*100:+.2f}%")
        print(f"  Best improvement: {max(optimization_stats['improvements'])*100:+.2f}%")
        print(f"  Worst change: {min(optimization_stats['improvements'])*100:+.2f}%")
    
    # Show aggregate metrics
    agg = results.get('aggregate_metrics', {})
    if agg:
        print(f"\nOverall Performance:")
        print(f"  Accuracy: {agg['accuracy']['mean']:.2%} ± {agg['accuracy']['std']:.2%}")
        if 'precision_harmful' in agg:
            print(f"  Precision (harmful): {agg['precision_harmful']['mean']:.3f} ± {agg['precision_harmful']['std']:.3f}")
            print(f"  Recall (harmful): {agg['recall_harmful']['mean']:.3f} ± {agg['recall_harmful']['std']:.3f}")
            print(f"  F1 (harmful): {agg['f1_harmful']['mean']:.3f} ± {agg['f1_harmful']['std']:.3f}")
    
    # Compare with fake optimization
    print("\n" + "="*80)
    print("COMPARISON WITH PREVIOUS (FAKE) OPTIMIZATION")
    print("="*80)
    
    # Look for previous results
    prev_results = results_dir.glob("results/20250707_*/detailed_results.json")
    for prev_file in prev_results:
        if "honest" not in str(prev_file):
            with open(prev_file, 'r') as f:
                prev_data = json.load(f)
            
            prev_changed = sum(1 for fold in prev_data.get('fold_results', []) 
                             if fold['prompts']['changed'])
            
            print(f"\nPrevious results ({prev_file.parent.name}):")
            print(f"  Claimed optimizations: {prev_changed}/{len(prev_data.get('fold_results', []))}")
            print(f"  REAL optimizations: {optimization_stats['optimized_folds']}/{optimization_stats['total_folds']}")
            print(f"  Honesty improvement: {optimization_stats['optimized_folds'] - prev_changed} more real optimizations")
            break
    
    # Show examples of optimized prompts
    print("\n" + "="*80)
    print("OPTIMIZATION EXAMPLES")
    print("="*80)
    
    for i, fold in enumerate(fold_results[:3]):  # Show first 3
        if fold['prompts']['changed']:
            print(f"\nFold {fold['fold']}:")
            print(f"  Validation accuracy: {fold.get('validation_accuracy', 'N/A')}")
            print(f"  Test accuracy: {fold['test_accuracy']:.2%}")
            print(f"  Improvement: {(fold['test_accuracy'] - fold.get('validation_accuracy', fold['test_accuracy']))*100:+.2f}%")
            print(f"  Prompt changed: YES")
            break

if __name__ == "__main__":
    analyze_results(Path.cwd())