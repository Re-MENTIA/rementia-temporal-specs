#!/usr/bin/env python3
"""
Run honest evaluation with REAL TextGrad optimization
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the evaluator
from scripts.run_unified_crossval import UnifiedCrossValidator, calculate_metrics

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/honest_evaluation_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation for collective instruction
    print("\n" + "="*80)
    print("HONEST EVALUATION WITH REAL TEXTGRAD OPTIMIZATION")
    print("="*80)
    
    evaluator = UnifiedCrossValidator(
        task="elderspeak",
        task_type="collective"
    )
    
    # Load dataset
    dataset_path = Path("datasets/Elderspeak/collective/collective_instruction_dataset.json")
    with open(dataset_path, 'r') as f:
        raw_dataset = json.load(f)
    
    # Transform dataset to expected format
    dataset = []
    for item in raw_dataset:
        dataset.append({
            'input': item['sentence'],
            'label': '1' if item['instruction_type'] == 'collective' else '0'
        })
    
    print(f"\nDataset: {dataset_path}")
    print(f"Total samples: {len(dataset)}")
    
    # Run cross-validation with optimization
    results = evaluator.run_cross_validation(dataset)
    
    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Count optimized prompts
    optimized_count = sum(1 for r in results['fold_results'] 
                         if r['prompts']['changed'])
    total_folds = len(results['fold_results'])
    
    print(f"\nPrompt Optimization:")
    print(f"  Folds with optimization: {optimized_count}/{total_folds}")
    print(f"  Optimization rate: {optimized_count/total_folds*100:.1f}%")
    
    # Show aggregate metrics
    agg = results['aggregate_metrics']
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy: {agg['accuracy']['mean']:.2%} ± {agg['accuracy']['std']:.2%}")
    print(f"  Precision (harmful): {agg['precision_harmful']['mean']:.3f} ± {agg['precision_harmful']['std']:.3f}")
    print(f"  Recall (harmful): {agg['recall_harmful']['mean']:.3f} ± {agg['recall_harmful']['std']:.3f}")
    print(f"  F1 (harmful): {agg['f1_harmful']['mean']:.3f} ± {agg['f1_harmful']['std']:.3f}")
    
    # Show improvements
    improvements = []
    for fold in results['fold_results']:
        if fold['prompts']['changed'] and fold.get('validation_accuracy'):
            # Calculate improvement from validation
            initial_acc = fold.get('validation_accuracy', fold['test_accuracy'])
            improvement = fold['test_accuracy'] - initial_acc
            improvements.append(improvement)
    
    if improvements:
        avg_improvement = sum(improvements) / len(improvements)
        print(f"\nOptimization Impact:")
        print(f"  Average improvement: {avg_improvement*100:+.2f}%")
        print(f"  Max improvement: {max(improvements)*100:+.2f}%")
        print(f"  Min improvement: {min(improvements)*100:+.2f}%")
    
    print(f"\nResults saved to: {results_file}")
    print("="*80)

if __name__ == "__main__":
    main()