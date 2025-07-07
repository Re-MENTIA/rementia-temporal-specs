#!/usr/bin/env python3
"""
Optimize prompts for all detectors and save results
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.detectors import (
    AccommodationSpeechDetector,
    EpisodeMemoryDetector,
    OpenEndQuestionDetector,
    LongSpeechDetector
)
from src.optimization import PromptOptimizer
from src.utils.data_loader import DataLoader


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_dataset_for_detector(detector_name: str, config: Dict) -> List[Dict]:
    """Load dataset for a specific detector"""
    loader = DataLoader(config)
    
    if detector_name == 'accommodation':
        return loader.load_accommodation_speech_data()
    elif detector_name == 'episode_memory':
        return loader.load_episode_memory_data()
    elif detector_name == 'open_end_question':
        return loader.load_open_end_question_data()
    elif detector_name == 'long_speech':
        return loader.load_long_speech_data()
    else:
        raise ValueError(f"Unknown detector: {detector_name}")


def split_dataset(dataset: List[Dict], val_ratio: float = 0.2) -> tuple:
    """Split dataset into train and validation sets"""
    val_size = int(len(dataset) * val_ratio)
    return dataset[val_size:], dataset[:val_size]


def optimize_detector(detector, dataset: List[Dict], optimizer: PromptOptimizer) -> Dict:
    """Optimize a single detector"""
    # Split dataset
    train_data, val_data = split_dataset(dataset)
    
    # Optimize prompt
    optimized_prompt, val_accuracy = optimizer.optimize(detector, train_data, val_data)
    
    # Update detector with optimized prompt
    detector.prompt = optimized_prompt
    
    # Evaluate on full dataset
    full_results = detector.evaluate(dataset)
    
    return {
        'detector_name': detector.__class__.__name__,
        'prompt': optimized_prompt,
        'validation_accuracy': val_accuracy,
        'full_dataset_accuracy': full_results['accuracy'],
        'metrics': full_results['metrics']
    }


def main():
    """Main optimization routine"""
    parser = argparse.ArgumentParser(description='Optimize prompts for all detectors')
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output',
        default='src/prompts/optimized_prompts.json',
        help='Output path for optimized prompts'
    )
    
    args = parser.parse_args()
    logger = setup_logging()
    
    # Load configuration
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize optimizer
    optimizer = PromptOptimizer(args.config)
    
    # Initialize detectors
    detectors = {
        'accommodation': AccommodationSpeechDetector(args.config),
        'episode_memory': EpisodeMemoryDetector(args.config),
        'open_end_question': OpenEndQuestionDetector(args.config),
        'long_speech': LongSpeechDetector(args.config)
    }
    
    # Optimize each detector
    results = {}
    
    for name, detector in detectors.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Optimizing {name} detector")
        logger.info('='*60)
        
        try:
            # Load dataset
            dataset = load_dataset_for_detector(name, config)
            logger.info(f"Loaded {len(dataset)} samples")
            
            # Optimize
            result = optimize_detector(detector, dataset, optimizer)
            results[name] = result
            
            logger.info(f"Optimization complete:")
            logger.info(f"  Validation accuracy: {result['validation_accuracy']:.2%}")
            logger.info(f"  Full dataset accuracy: {result['full_dataset_accuracy']:.2%}")
            
        except Exception as e:
            logger.error(f"Error optimizing {name}: {e}")
            results[name] = {'error': str(e)}
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\nSaved optimized prompts to {output_path}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("OPTIMIZATION SUMMARY")
    logger.info("="*60)
    
    for name, result in results.items():
        if 'error' in result:
            logger.info(f"{name}: ERROR - {result['error']}")
        else:
            logger.info(f"{name}: {result['full_dataset_accuracy']:.2%} accuracy")


if __name__ == "__main__":
    main()