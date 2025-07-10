#!/usr/bin/env python3
"""
Quick honest evaluation with REAL TextGrad - 1 repeat only for demo
"""

import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.run_unified_crossval import UnifiedCrossValidator

def main():
    print("\n" + "="*80)
    print("QUICK HONEST EVALUATION WITH REAL TEXTGRAD (1 REPEAT)")
    print("="*80)
    
    # Modify config for quick run
    config_path = Path("config/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set to 1 repeat for quick demo
    original_repeats = config['cross_validation']['n_repeats']
    config['cross_validation']['n_repeats'] = 1
    
    # Create temporary config
    temp_config = Path("config/temp_quick_config.yaml")
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
    
    try:
        # Create evaluator with modified config
        evaluator = UnifiedCrossValidator(
            task="elderspeak",
            task_type="collective"
        )
        # Override config
        evaluator.config = config
        
        # Load dataset
        dataset_path = Path("datasets/Elderspeak/collective/collective_instruction_dataset.json")
        with open(dataset_path, 'r') as f:
            raw_dataset = json.load(f)
        
        # Transform dataset
        dataset = []
        for item in raw_dataset:
            dataset.append({
                'input': item['sentence'],
                'label': '1' if item['instruction_type'] == 'collective' else '0'
            })
        
        print(f"\nDataset: {dataset_path}")
        print(f"Total samples: {len(dataset)}")
        print(f"Running with {config['cross_validation']['n_repeats']} repeat(s)")
        
        # Run evaluation
        results = evaluator.run_cross_validation(dataset)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/quick_honest_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("QUICK EVALUATION RESULTS")
        print("="*80)
        
        # Optimization summary
        optimized = sum(1 for r in results['fold_results'] if r['prompts']['changed'])
        total = len(results['fold_results'])
        
        print(f"\nOptimization Summary:")
        print(f"  Folds with REAL optimization: {optimized}/{total}")
        
        # Show examples
        for fold in results['fold_results']:
            if fold['prompts']['changed']:
                print(f"\nFold {fold['fold']} - REAL OPTIMIZATION:")
                print(f"  Initial validation accuracy: {fold.get('validation_accuracy', 'N/A')}")
                print(f"  Test accuracy: {fold['test_accuracy']:.2%}")
                print(f"  Prompt changed: YES (REAL TextGrad)")
                break
        
        # Performance
        agg = results['aggregate_metrics']
        print(f"\nOverall Performance:")
        print(f"  Accuracy: {agg['accuracy']['mean']:.2%}")
        
        print(f"\nResults saved to: {results_file}")
        
    finally:
        # Clean up temp config
        if temp_config.exists():
            temp_config.unlink()
    
    # Restore original config
    config['cross_validation']['n_repeats'] = original_repeats
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

if __name__ == "__main__":
    main()