#!/usr/bin/env python3
"""
Main evaluation script for accommodation speech detection
Clean, configurable implementation based on best practices
"""

import argparse
import json
import logging
import os
import random
import sys
import time
import yaml
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.openai_client import OpenAIClient
from src.utils.metrics import calculate_metrics


class EvaluationRunner:
    """Main evaluation runner with configuration support"""
    
    def __init__(self, config_path: str = "config/config.yaml", 
                 accommodation_type: str = "ToE"):
        """Initialize with configuration"""
        self.config = self._load_config(config_path)
        self.accommodation_type = accommodation_type
        self.client = OpenAIClient()
        self._setup_logging()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format=log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s'),
            handlers=[
                logging.FileHandler(log_config.get('file', 'evaluation.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_dataset(self, dataset_path: Optional[str] = None, 
                    accommodation_type: str = "ToE") -> List[Dict]:
        """Load dataset from file"""
        if dataset_path:
            path = dataset_path
        else:
            base_path = self.config['dataset']['base_path']
            type_config = self.config['dataset']['accommodation_types'][accommodation_type]
            path = os.path.join(base_path, type_config['path'])
        
        self.logger.info(f"Loading dataset from {path}")
        
        with open(path, 'r', encoding=self.config['dataset']['encoding']) as f:
            dataset = json.load(f)
        
        self.logger.info(f"Loaded {len(dataset)} samples")
        return dataset
    
    def evaluate_example(self, text: str, prompt: str) -> str:
        """Evaluate a single example"""
        try:
            return self.client.predict(
                text=text,
                prompt=prompt,
                model=self.config['models']['default']
            )
        except Exception as e:
            self.logger.error(f"Error evaluating example: {e}")
            return "0"
    
    def optimize_prompt(self, train_data: List[Dict], val_data: List[Dict], 
                       initial_prompt: str) -> Tuple[str, bool]:
        """Optimize prompt based on validation performance"""
        if not self.config['prompt_optimization']['enabled']:
            return initial_prompt, False
        
        self.logger.info("Optimizing prompt on validation set")
        
        # Sample validation subset
        n_samples = self.config['prompt_optimization']['validation_samples']
        val_subset = random.sample(val_data, min(n_samples, len(val_data)))
        
        # Evaluate initial prompt
        val_predictions = []
        for example in val_subset:
            pred = self.evaluate_example(example['input'], initial_prompt)
            val_predictions.append({
                'input': example['input'],
                'true': example['label'],
                'pred': pred,
                'correct': example['label'] == pred
            })
            time.sleep(self.config['api']['rate_limit_delay'])
        
        # Calculate metrics
        metrics = calculate_metrics(val_predictions)
        self.logger.info(f"Initial validation accuracy: {metrics['accuracy']:.2%}")
        
        # Optimization logic
        optimized = initial_prompt
        changed = False
        
        opt_config = self.config['prompt_optimization']
        
        if metrics['accuracy'] < opt_config['accuracy_threshold']:
            if metrics['recall_harmful'] < opt_config['recall_threshold']:
                # Improve recall
                optimized += """

CRITICAL: Pay special attention to:
- ANY use of -chan suffix (ちゃん) with elderly people
- Words like いい子, えらい, よしよし when addressing adults
- Baby-talk patterns even if mixed with polite language"""
                changed = True
                
            elif metrics['precision_harmful'] < opt_config['precision_threshold']:
                # Improve precision
                optimized += """

IMPORTANT: Only flag CLEAR infantilization:
- Respectful -san/-sama is always appropriate
- Professional language is safe even if warm
- Focus on patronizing tone, not just friendly language"""
                changed = True
        
        if changed:
            self.logger.info("Prompt optimized based on validation performance")
        
        return optimized, changed
    
    def run_single_fold(self, train_data: List[Dict], val_data: List[Dict], 
                       test_data: List[Dict], fold_num: int, repeat_num: int) -> Dict:
        """Run evaluation for a single fold"""
        self.logger.info(f"Running Fold {fold_num} (Repeat {repeat_num})")
        self.logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Load initial prompt
        initial_prompt = self._get_initial_prompt()
        
        # Optimize prompt
        optimized_prompt, prompt_changed = self.optimize_prompt(
            train_data, val_data, initial_prompt
        )
        
        # Evaluate on test set
        self.logger.info("Evaluating on test set")
        test_predictions = []
        
        for i, example in enumerate(test_data):
            if i % 10 == 0:
                self.logger.debug(f"Progress: {i}/{len(test_data)}")
            
            pred = self.evaluate_example(example['input'], optimized_prompt)
            test_predictions.append({
                'input': example['input'],
                'true': example['label'],
                'pred': pred,
                'correct': example['label'] == pred
            })
            
            # Rate limiting
            time.sleep(self.config['api']['rate_limit_delay'])
        
        # Calculate metrics
        metrics = calculate_metrics(test_predictions)
        
        # Prepare results
        results = {
            'fold': fold_num,
            'repeat': repeat_num,
            'dataset_split': {
                'train_size': len(train_data),
                'val_size': len(val_data),
                'test_size': len(test_data),
                'test_harmful': sum(1 for x in test_data if x['label'] == '1'),
                'test_safe': sum(1 for x in test_data if x['label'] == '0')
            },
            'prompt_optimized': prompt_changed,
            'metrics': metrics,
            'predictions': test_predictions[:self.config['output']['max_predictions_to_save']]
        }
        
        self.logger.info(f"Fold {fold_num} - Accuracy: {metrics['accuracy']:.2%}, "
                        f"Precision: {metrics['precision_harmful']:.3f}, "
                        f"Recall: {metrics['recall_harmful']:.3f}")
        
        return results
    
    def _get_initial_prompt(self) -> str:
        """Get initial prompt from config or default"""
        prompts = self.config.get('prompts', {})
        if self.accommodation_type in prompts:
            return prompts[self.accommodation_type]['initial']
        else:
            # Default prompt if type not configured
            return """You are an expert in analyzing eldercare communication.
Analyze the text and respond with ONLY "1" (harmful) or "0" (safe)."""
    
    def create_stratified_folds(self, dataset: List[Dict]) -> List[List[Dict]]:
        """Create stratified k-folds"""
        cv_config = self.config['cross_validation']
        n_folds = cv_config['n_folds']
        
        # Separate by class
        harmful = [d for d in dataset if d['label'] == '1']
        safe = [d for d in dataset if d['label'] == '0']
        
        # Shuffle
        random.shuffle(harmful)
        random.shuffle(safe)
        
        # Create folds
        folds = [[] for _ in range(n_folds)]
        
        # Distribute samples
        for i, sample in enumerate(harmful):
            folds[i % n_folds].append(sample)
        
        for i, sample in enumerate(safe):
            folds[i % n_folds].append(sample)
        
        # Shuffle each fold
        for fold in folds:
            random.shuffle(fold)
        
        return folds
    
    def run_cross_validation(self, dataset: List[Dict]) -> Dict:
        """Run complete cross-validation"""
        cv_config = self.config['cross_validation']
        n_folds = cv_config['n_folds']
        n_repeats = cv_config['n_repeats']
        
        self.logger.info(f"Starting {n_folds}-fold cross-validation with {n_repeats} repeats")
        
        all_results = []
        
        for repeat in range(n_repeats):
            self.logger.info(f"\n=== REPEAT {repeat + 1}/{n_repeats} ===")
            
            # Set random seed
            random.seed(cv_config['random_seed_base'] + repeat * 100)
            
            # Create folds
            folds = self.create_stratified_folds(dataset)
            
            for fold_idx in range(n_folds):
                # Prepare data
                test_fold = folds[fold_idx]
                train_folds = [folds[j] for j in range(n_folds) if j != fold_idx]
                all_train = [item for fold in train_folds for item in fold]
                
                # Split train/val
                random.shuffle(all_train)
                val_size = int(len(all_train) * cv_config['validation_split_ratio'])
                val_data = all_train[:val_size]
                train_data = all_train[val_size:]
                
                # Run fold
                fold_results = self.run_single_fold(
                    train_data, val_data, test_fold,
                    fold_idx + 1, repeat + 1
                )
                
                all_results.append(fold_results)
                
                # Save checkpoint if enabled
                if self.config['output']['checkpoint_enabled']:
                    self._save_checkpoint(all_results)
        
        # Aggregate results
        return self._aggregate_results(all_results, dataset)
    
    def _aggregate_results(self, results: List[Dict], dataset: List[Dict]) -> Dict:
        """Aggregate cross-validation results"""
        import numpy as np
        
        # Extract metrics
        accuracies = [r['metrics']['accuracy'] for r in results]
        precisions = [r['metrics']['precision_harmful'] for r in results]
        recalls = [r['metrics']['recall_harmful'] for r in results]
        f1s = [r['metrics']['f1_harmful'] for r in results]
        
        # Aggregate confusion matrix
        total_tp = sum(r['metrics']['confusion_matrix']['tp'] for r in results)
        total_tn = sum(r['metrics']['confusion_matrix']['tn'] for r in results)
        total_fp = sum(r['metrics']['confusion_matrix']['fp'] for r in results)
        total_fn = sum(r['metrics']['confusion_matrix']['fn'] for r in results)
        
        # Collect all errors
        all_errors = []
        for r in results:
            for error in r['metrics'].get('errors', []):
                error['fold'] = r['fold']
                error['repeat'] = r['repeat']
                all_errors.append(error)
        
        aggregated = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'dataset_stats': {
                'total': len(dataset),
                'harmful': sum(1 for d in dataset if d['label'] == '1'),
                'safe': sum(1 for d in dataset if d['label'] == '0')
            },
            'cross_validation': {
                'n_folds': self.config['cross_validation']['n_folds'],
                'n_repeats': self.config['cross_validation']['n_repeats'],
                'total_evaluations': len(results)
            },
            'aggregate_metrics': {
                'accuracy': {
                    'mean': float(np.mean(accuracies)),
                    'std': float(np.std(accuracies)),
                    'min': float(np.min(accuracies)),
                    'max': float(np.max(accuracies))
                },
                'precision_harmful': {
                    'mean': float(np.mean(precisions)),
                    'std': float(np.std(precisions))
                },
                'recall_harmful': {
                    'mean': float(np.mean(recalls)),
                    'std': float(np.std(recalls))
                },
                'f1_harmful': {
                    'mean': float(np.mean(f1s)),
                    'std': float(np.std(f1s))
                }
            },
            'aggregate_confusion_matrix': {
                'tp': int(total_tp),
                'tn': int(total_tn),
                'fp': int(total_fp),
                'fn': int(total_fn)
            },
            'errors': {
                'total': len(all_errors),
                'by_type': {
                    'false_positives': len([e for e in all_errors if e['true'] == '0']),
                    'false_negatives': len([e for e in all_errors if e['true'] == '1'])
                },
                'samples': all_errors[:10]  # Save sample errors
            },
            'detailed_results': results
        }
        
        return aggregated
    
    def _save_checkpoint(self, results: List[Dict]):
        """Save checkpoint for recovery"""
        checkpoint_file = self.config['output']['checkpoint_file']
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        self.logger.debug(f"Checkpoint saved to {checkpoint_file}")
    
    def save_results(self, results: Dict, output_file: str):
        """Save results to file in timestamped directory"""
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(self.config['output']['results_dir'], timestamp)
        os.makedirs(results_dir, exist_ok=True)
        
        # Save main results JSON
        output_path = os.path.join(results_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Results saved to {output_path}")
        
        # Generate reports using ReportGenerator
        from src.utils.reporting import ReportGenerator
        reporter = ReportGenerator(results)
        reporter.generate_all_reports(results_dir)
        
        # Clean up checkpoint
        checkpoint_file = self.config['output']['checkpoint_file']
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            self.logger.debug("Checkpoint file removed")
        
        return results_dir


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run cross-validation evaluation for accommodation speech detection'
    )
    parser.add_argument(
        '--config', 
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--dataset',
        help='Path to dataset (overrides config)'
    )
    parser.add_argument(
        '--type',
        default='ToE',
        help='Accommodation type (e.g., ToE, BT, etc.)'
    )
    parser.add_argument(
        '--output',
        default='crossvalidation_results.json',
        help='Output filename'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint'
    )
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = EvaluationRunner(args.config, args.type)
    
    # Load dataset
    dataset = runner.load_dataset(args.dataset, args.type)
    
    # Check for checkpoint
    if args.resume and os.path.exists(runner.config['output']['checkpoint_file']):
        runner.logger.info("Resuming from checkpoint...")
        # TODO: Implement checkpoint recovery
    
    # Run cross-validation
    results = runner.run_cross_validation(dataset)
    
    # Save results
    results_dir = runner.save_results(results, args.output)
    
    # Print summary
    print("\n" + "="*60)
    print("CROSS-VALIDATION COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {results_dir}")
    print(f"\nOverall Performance:")
    print(f"  Accuracy: {results['aggregate_metrics']['accuracy']['mean']:.2%} ± "
          f"{results['aggregate_metrics']['accuracy']['std']:.2%}")
    print(f"  Precision: {results['aggregate_metrics']['precision_harmful']['mean']:.3f} ± "
          f"{results['aggregate_metrics']['precision_harmful']['std']:.3f}")
    print(f"  Recall: {results['aggregate_metrics']['recall_harmful']['mean']:.3f} ± "
          f"{results['aggregate_metrics']['recall_harmful']['std']:.3f}")
    print(f"  F1 Score: {results['aggregate_metrics']['f1_harmful']['mean']:.3f} ± "
          f"{results['aggregate_metrics']['f1_harmful']['std']:.3f}")
    print(f"\nTotal Errors: {results['errors']['total']}")
    print(f"  False Positives: {results['errors']['by_type']['false_positives']}")
    print(f"  False Negatives: {results['errors']['by_type']['false_negatives']}")


if __name__ == "__main__":
    main()