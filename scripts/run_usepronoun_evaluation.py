#!/usr/bin/env python3
"""
Run evaluation for usePronoun detector with TextGrad optimization
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.run_unified_crossval import UnifiedCrossValidator


def setup_logging():
    """Setup logging configuration"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/usepronoun_evaluation_{timestamp}.log'
    
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def load_dataset(dataset_path: str):
    """Load and format the usePronoun dataset"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Format for unified cross-validator
    formatted_data = []
    for item in raw_data:
        formatted_data.append({
            'sentence': item['sentence'],
            'label': str(item['label']),  # Convert to string
            'language': item.get('language', 'Unknown'),
            'context': item.get('context', '')
        })
    
    return formatted_data


def run_evaluation():
    """Run the evaluation"""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Starting usePronoun evaluation with TextGrad optimization")
    logger.info("=" * 60)
    
    try:
        # Initialize evaluator
        evaluator = UnifiedCrossValidator(
            task="pronoun_detection",
            task_type="use_pronoun"
        )
        
        # Load dataset using evaluator's method to ensure proper normalization
        dataset = evaluator.load_dataset()
        
        logger.info(f"Dataset loaded: {len(dataset)} samples")
        
        # Count labels
        label_counts = {}
        for item in dataset:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        logger.info(f"Label distribution: {label_counts}")
        
        # Run cross-validation
        results = evaluator.run_cross_validation(dataset)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f'results/usepronoun_evaluation_{timestamp}')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full results
        with open(output_dir / 'full_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Generate summary
        summary = {
            'timestamp': timestamp,
            'dataset_size': len(dataset),
            'label_distribution': label_counts,
            'accuracy': {
                'mean': results['aggregate_metrics']['accuracy']['mean'],
                'std': results['aggregate_metrics']['accuracy']['std']
            },
            'optimizations': sum(1 for f in results['fold_results'] if f['prompts']['changed']),
            'total_folds': len(results['fold_results'])
        }
        
        with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # Log summary
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Overall Accuracy: {summary['accuracy']['mean']:.2%} ± {summary['accuracy']['std']:.2%}")
        logger.info(f"Optimizations: {summary['optimizations']}/{summary['total_folds']} folds")
        logger.info(f"Results saved to: {output_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        logger.error("Traceback:", exc_info=True)
        raise


if __name__ == "__main__":
    run_evaluation()