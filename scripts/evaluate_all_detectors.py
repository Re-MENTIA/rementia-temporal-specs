#!/usr/bin/env python3
"""
Evaluate all detectors with cross-validation
"""

import argparse
import json
import logging
import sys
from datetime import datetime
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
from src.utils.data_loader import DataLoader
from src.utils.cross_validator import CrossValidator
from src.utils.unified_reporting import UnifiedReportGenerator


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def evaluate_detector(detector, dataset: List[Dict], config: Dict) -> Dict:
    """Evaluate a single detector with cross-validation"""
    # Initialize cross-validator
    cv = CrossValidator(config)
    
    # Run cross-validation
    results = cv.run_cross_validation(detector, dataset)
    
    # Add detector info
    results['detector_info'] = {
        'name': detector.__class__.__name__,
        'description': detector.detector_config.description,
        'labels': detector.detector_config.labels
    }
    
    return results


def main():
    """Main evaluation routine"""
    parser = argparse.ArgumentParser(description='Evaluate all detectors')
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--prompts',
        help='Path to optimized prompts (optional)'
    )
    parser.add_argument(
        '--output-dir',
        default='results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    logger = setup_logging()
    
    # Load configuration
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize data loader
    data_loader = DataLoader(config)
    
    # Initialize detectors
    detectors = {
        'accommodation': AccommodationSpeechDetector(args.config),
        'episode_memory': EpisodeMemoryDetector(args.config),
        'open_end_question': OpenEndQuestionDetector(args.config),
        'long_speech': LongSpeechDetector(args.config)
    }
    
    # Load optimized prompts if provided
    if args.prompts:
        logger.info(f"Loading optimized prompts from {args.prompts}")
        with open(args.prompts, 'r') as f:
            prompts = json.load(f)
            
        for name, detector in detectors.items():
            if name in prompts and 'prompt' in prompts[name]:
                detector.prompt = prompts[name]['prompt']
                logger.info(f"Loaded optimized prompt for {name}")
    
    # Evaluate each detector
    all_results = {}
    
    for name, detector in detectors.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {name} detector")
        logger.info('='*60)
        
        try:
            # Load dataset
            if name == 'accommodation':
                dataset = data_loader.load_accommodation_speech_data()
            elif name == 'episode_memory':
                dataset = data_loader.load_episode_memory_data()
            elif name == 'open_end_question':
                dataset = data_loader.load_open_end_question_data()
            elif name == 'long_speech':
                dataset = data_loader.load_long_speech_data()
            
            logger.info(f"Loaded {len(dataset)} samples")
            
            # Evaluate
            results = evaluate_detector(detector, dataset, config)
            all_results[name] = results
            
            # Print summary
            logger.info(f"\nResults for {name}:")
            logger.info(f"  Overall Accuracy: {results['aggregate_metrics']['accuracy']['mean']:.2%} "
                       f"± {results['aggregate_metrics']['accuracy']['std']:.2%}")
            
        except Exception as e:
            logger.error(f"Error evaluating {name}: {e}")
            all_results[name] = {'error': str(e)}
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"evaluation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw results
    with open(output_dir / 'all_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # Generate reports for each detector
    for name, results in all_results.items():
        if 'error' not in results:
            detector_dir = output_dir / name
            detector_dir.mkdir(exist_ok=True)
            
            reporter = UnifiedReportGenerator(results)
            reporter.generate_all_reports(str(detector_dir))
    
    # Generate summary report
    generate_summary_report(all_results, output_dir)
    
    logger.info(f"\nResults saved to {output_dir}")


def generate_summary_report(all_results: Dict, output_dir: Path):
    """Generate summary report for all detectors"""
    with open(output_dir / 'summary.md', 'w', encoding='utf-8') as f:
        f.write("# Detector Evaluation Summary\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overall Results\n\n")
        f.write("| Detector | Accuracy | Precision | Recall | F1 Score |\n")
        f.write("|----------|----------|-----------|--------|----------|\n")
        
        for name, results in all_results.items():
            if 'error' in results:
                f.write(f"| {name} | ERROR | - | - | - |\n")
            else:
                metrics = results['aggregate_metrics']
                f.write(f"| {name} | "
                       f"{metrics['accuracy']['mean']:.2%} ± {metrics['accuracy']['std']:.2%} | "
                       f"{metrics.get('precision_1', metrics.get('precision_harmful', {})).get('mean', 0):.3f} | "
                       f"{metrics.get('recall_1', metrics.get('recall_harmful', {})).get('mean', 0):.3f} | "
                       f"{metrics.get('f1_1', metrics.get('f1_harmful', {})).get('mean', 0):.3f} |\n")
        
        f.write("\n## Detector Details\n\n")
        
        for name, results in all_results.items():
            if 'error' not in results:
                info = results.get('detector_info', {})
                f.write(f"### {name}\n")
                f.write(f"- **Description**: {info.get('description', 'N/A')}\n")
                f.write(f"- **Labels**: {info.get('labels', [])}\n")
                f.write(f"- **Total Samples**: {results.get('dataset_stats', {}).get('total_samples', 0)}\n")
                f.write(f"- **Cross-validation**: {results.get('cv_config', {}).get('n_folds', 0)} folds × "
                       f"{results.get('cv_config', {}).get('n_repeats', 0)} repeats\n\n")


if __name__ == "__main__":
    main()