#!/usr/bin/env python3
"""
usePronoun検出器のテスト（修正版）
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.run_unified_crossval import UnifiedCrossValidator

def test_usepronoun():
    """usePronoun検出器をテスト"""
    print("🔧 Testing usePronoun detector with fixed configuration...")
    
    # Initialize evaluator
    evaluator = UnifiedCrossValidator(
        task="pronoun_detection",
        task_type="use_pronoun"
    )
    
    # Load dataset using evaluator's method to ensure proper normalization
    dataset = evaluator.load_dataset()
    print(f"📊 Dataset loaded: {len(dataset)} samples")
    
    # Count labels
    label_counts = {}
    for item in dataset:
        label = str(item['label'])
        label_counts[label] = label_counts.get(label, 0) + 1
    print(f"   Label distribution: {label_counts}")
    
    # Run evaluation with just 1 fold for testing
    evaluator.config['cross_validation'] = evaluator.config.get('cross_validation', {})
    evaluator.config['cross_validation']['standard'] = {
        'n_folds': 1,
        'n_repeats': 1,
        'stratified': True,
        'random_seed_base': 42
    }
    evaluator.config['models'] = evaluator.config.get('models', {})
    evaluator.config['models']['default'] = 'gpt-4.1-mini'
    
    try:
        results = evaluator.run_cross_validation(dataset)
        
        # Display results
        print("\n✅ Evaluation completed successfully!")
        print(f"Accuracy: {results['aggregate_metrics']['accuracy']['mean']:.2%}")
        print(f"Optimizations: {sum(1 for f in results['fold_results'] if f['prompts']['changed'])}")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    test_usepronoun()