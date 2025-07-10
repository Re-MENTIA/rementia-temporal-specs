#!/usr/bin/env python3
"""
Unified cross-validation evaluation script with full reporting
Supports both accommodation speech and episode detection
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
# from src.utils.unified_reporting import UnifiedReportGenerator  # Temporarily disabled due to matplotlib


class UnifiedCrossValidator:
    """Unified cross-validation evaluator for multiple detection tasks"""
    
    def __init__(self, config_path: str = "config/config.yaml", 
                 task: str = "elderspeak", 
                 task_type: str = "ToE"):
        """Initialize with configuration"""
        self.config = self._load_config(config_path)
        self.task = task
        self.task_type = task_type
        self.client = OpenAIClient()
        self._setup_logging()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        log_file = f"{self.task}_{self.task_type}_crossval.log"
        
        # Ensure log file can be created
        try:
            # Create log file if it doesn't exist
            with open(log_file, 'a'):
                pass
        except:
            # If we can't create in current dir, use temp dir
            import tempfile
            log_file = os.path.join(tempfile.gettempdir(), log_file)
        
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_dataset(self, dataset_path: Optional[str] = None) -> List[Dict]:
        """Load dataset from file"""
        if dataset_path:
            path = dataset_path
        else:
            task_config = self.config['tasks'][self.task]['types'][self.task_type]
            base_path = self.config['tasks'][self.task]['base_path']
            path = os.path.join(base_path, task_config['path'])
        
        self.logger.info(f"Loading dataset from {path}")
        
        with open(path, 'r', encoding=self.config['dataset']['encoding']) as f:
            dataset = json.load(f)
        
        # Normalize dataset format
        task_config = self.config['tasks'][self.task]['types'][self.task_type]
        input_field = task_config.get('input_field', 'input')
        label_field = task_config.get('label_field', 'label')
        label_mapping = task_config.get('label_mapping', {})
        
        normalized = []
        for item in dataset:
            label = str(item.get(label_field, item.get('label')))
            # Apply label mapping if exists
            if label_mapping:
                label = label_mapping.get(label, label)
            
            normalized.append({
                'input': item.get(input_field, item.get('input')),
                'label': label
            })
        
        self.logger.info(f"Loaded {len(normalized)} samples")
        
        # Count label distribution
        label_counts = defaultdict(int)
        for item in normalized:
            label_counts[item['label']] += 1
        
        self.logger.info(f"Label distribution: {dict(label_counts)}")
        
        return normalized
    
    def evaluate_single(self, text: str, prompt: str) -> str:
        """Evaluate single text"""
        model = self.config['models']['default']
        response = self.client.predict(text=text, prompt=prompt, model=model)
        
        # Normalize response based on task type
        task_config = self.config['tasks'][self.task]['types'][self.task_type]
        labels = task_config['labels']
        
        response = response.strip()
        
        # For episode memory, handle the response mapping
        if self.task == "episode_detection":
            response = response.lower()
            if response in ["true", "1"]:
                response = "1"
            elif response in ["false", "neutral", "0"]:
                response = "0"
            else:
                self.logger.warning(f"Unexpected response: {response}, defaulting to 0")
                response = "0"
        
        if response not in labels:
            self.logger.warning(f"Unexpected response: {response}, defaulting to {labels[0]}")
            return labels[0]
        
        return response
    
    def evaluate_dataset(self, dataset: List[Dict], prompt: str, 
                        show_progress: bool = True) -> Tuple[float, List[Dict]]:
        """Evaluate entire dataset"""
        predictions = []
        correct = 0
        
        for i, item in enumerate(dataset):
            if show_progress and i % 10 == 0:
                self.logger.info(f"Progress: {i}/{len(dataset)}")
            
            pred = self.evaluate_single(item['input'], prompt)
            is_correct = pred == item['label']
            
            predictions.append({
                'input': item['input'],
                'true': item['label'],
                'pred': pred,
                'correct': is_correct
            })
            
            if is_correct:
                correct += 1
            
            time.sleep(self.config.get('api', {}).get('openai', {}).get('rate_limit_delay', 0.1))
        
        accuracy = correct / len(dataset) if dataset else 0
        return accuracy, predictions
    
    def optimize_prompt(self, train_data: List[Dict], val_data: List[Dict], 
                       initial_prompt: str) -> Tuple[str, float]:
        """REAL gradient-based prompt optimization"""
        self.logger.info(f"Starting REAL TextGrad optimization on {len(val_data)} validation samples")
        
        # Evaluate with initial prompt
        initial_accuracy, initial_predictions = self.evaluate_dataset(val_data, initial_prompt, show_progress=False)
        self.logger.info(f"Initial validation accuracy: {initial_accuracy:.2%}")
        
        # Calculate initial metrics
        initial_metrics = calculate_metrics(initial_predictions)
        
        # Check if optimization is needed
        opt_thresholds = self.config.get('textgrad_optimization', {}).get('thresholds', {})
        accuracy_threshold = opt_thresholds.get('accuracy', 0.95)
        recall_threshold = opt_thresholds.get('recall', 0.80)
        precision_threshold = opt_thresholds.get('precision', 0.90)
        
        if (initial_accuracy >= accuracy_threshold and
            initial_metrics.get('recall_harmful', initial_metrics.get('recall_1', 1.0)) >= recall_threshold and
            initial_metrics.get('precision_harmful', initial_metrics.get('precision_1', 1.0)) >= precision_threshold):
            self.logger.info("Initial prompt meets all thresholds, no optimization needed")
            return initial_prompt, initial_accuracy
        
        # Create improved prompt based on errors
        errors = [p for p in initial_predictions if not p['correct']]
        if not errors:
            self.logger.info("No errors found, returning initial prompt")
            return initial_prompt, initial_accuracy
        
        # Check prompt length
        prompt_tokens_estimate = len(initial_prompt) / 4
        if prompt_tokens_estimate > 2500:  # Leave room for gradient generation
            self.logger.info(f"Prompt too long for optimization (~{int(prompt_tokens_estimate)} tokens)")
            return initial_prompt, initial_accuracy
        
        self.logger.info(f"Found {len(errors)} errors, performing gradient-based optimization...")
        
        # Implement REAL gradient-based optimization
        current_prompt = initial_prompt
        best_prompt = initial_prompt
        best_accuracy = initial_accuracy
        
        # Optimization loop - 3 iterations
        for iteration in range(3):
            self.logger.info(f"\n--- Optimization Iteration {iteration + 1}/3 ---")
            
            # Sample errors for gradient computation (max 10)
            error_sample = errors[:10] if len(errors) > 10 else errors
            
            # Generate gradient feedback using GPT-4
            gradient_prompt = f"""You are an expert prompt engineer optimizing a classification prompt.

Current prompt:
{current_prompt}

The prompt had {len(errors)} errors out of {len(val_data)} validation samples (accuracy: {initial_accuracy:.2%}).

Here are examples of misclassifications:
"""
            for i, err in enumerate(error_sample[:5]):
                gradient_prompt += f"\n{i+1}. Input: \"{err['input']}\"\n   Expected: {err['true']}, Got: {err['pred']}\n"
            
            gradient_prompt += f"""
Analyze why these errors occurred and provide specific improvements to the prompt.
Focus on:
1. Clarifying ambiguous criteria
2. Adding missing edge cases
3. Improving examples if present
4. Making decision boundaries clearer

Provide the COMPLETE improved prompt that addresses these issues.
The improved prompt should maintain the same structure and end with the same instruction format."""
            
            # Get gradient (improvement suggestions)
            gradient_response = self.client.generate(
                text=gradient_prompt,
                system_prompt="You are an expert at improving classification prompts based on error analysis.",
                model="gpt-4o-mini",
                max_tokens=4000
            )
            
            # Extract improved prompt
            improved_prompt = gradient_response.strip()
            
            # Validate improved prompt
            if len(improved_prompt) < 100:
                self.logger.warning(f"Invalid gradient response (too short: {len(improved_prompt)} chars), keeping current prompt")
                continue
            
            # Check if it contains instruction format (more flexible)
            if not any(keyword in improved_prompt.lower() for keyword in ["analyze", "respond with", "classify", "detect"]):
                self.logger.warning("Invalid gradient response (no instruction found), keeping current prompt")
                self.logger.debug(f"Response preview: {improved_prompt[:200]}...")
                continue
            
            # Evaluate improved prompt
            new_accuracy, new_predictions = self.evaluate_dataset(val_data, improved_prompt, show_progress=False)
            self.logger.info(f"Iteration {iteration + 1} accuracy: {new_accuracy:.2%} (Δ = {(new_accuracy - initial_accuracy)*100:+.2f}%)")
            
            # Update best prompt if improved
            if new_accuracy > best_accuracy:
                best_prompt = improved_prompt
                best_accuracy = new_accuracy
                errors = [p for p in new_predictions if not p['correct']]
                self.logger.info(f"✓ Improvement found! New best accuracy: {best_accuracy:.2%}")
            else:
                self.logger.info("✗ No improvement, keeping previous best")
            
            # Early stopping if we reach target performance
            new_metrics = calculate_metrics(new_predictions)
            if (new_accuracy >= accuracy_threshold and
                new_metrics.get('recall_harmful', new_metrics.get('recall_1', 1.0)) >= recall_threshold and
                new_metrics.get('precision_harmful', new_metrics.get('precision_1', 1.0)) >= precision_threshold):
                self.logger.info("Target performance reached, stopping optimization")
                break
            
            current_prompt = improved_prompt
        
        # Log optimization results
        improvement = best_accuracy - initial_accuracy
        self.logger.info(f"\nOptimization complete:")
        self.logger.info(f"  Initial accuracy: {initial_accuracy:.2%}")
        self.logger.info(f"  Final accuracy: {best_accuracy:.2%}")
        self.logger.info(f"  Improvement: {improvement*100:+.2f}%")
        self.logger.info(f"  Prompt changed: {best_prompt != initial_prompt}")
        
        return best_prompt, best_accuracy
    
    def run_fold(self, fold_num: int, train_data: List[Dict], 
                 val_data: List[Dict], test_data: List[Dict]) -> Dict:
        """Run evaluation for one fold"""
        self.logger.info(f"Running Fold {fold_num}")
        self.logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Get initial prompt
        initial_prompt = self.get_prompt()
        
        # Optimize prompt if enabled
        textgrad_config = self.config.get('textgrad_optimization', {})
        if textgrad_config.get('enabled', True) and val_data:
            optimized_prompt, val_accuracy = self.optimize_prompt(train_data, val_data, initial_prompt)
        else:
            optimized_prompt = initial_prompt
            val_accuracy = None
        
        # Evaluate on test set
        self.logger.info("Evaluating on test set")
        test_accuracy, predictions = self.evaluate_dataset(test_data, optimized_prompt)
        
        # Calculate metrics
        metrics = calculate_metrics(predictions)
        
        results = {
            'fold': fold_num,
            'data_split': {
                'train': len(train_data),
                'val': len(val_data),
                'test': len(test_data)
            },
            'prompts': {
                'initial': initial_prompt,
                'optimized': optimized_prompt,
                'changed': initial_prompt != optimized_prompt
            },
            'validation_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
            'metrics': metrics,
            'predictions': predictions[:self.config.get('output', {}).get('predictions', {}).get('max_to_save', 50)]
        }
        
        self.logger.info(f"Fold {fold_num} - Accuracy: {test_accuracy:.2%}, "
                        f"Precision: {metrics.get('precision_harmful', metrics.get('precision_1', 0)):.3f}, "
                        f"Recall: {metrics.get('recall_harmful', metrics.get('recall_1', 0)):.3f}")
        
        return results
    
    def get_prompt(self) -> str:
        """Get prompt for current task and type"""
        prompt_key = self.task_type
        if self.task == "episode_detection":
            prompt_key = "episode_memory"  # Map to prompt key
        elif self.task == "question_detection":
            prompt_key = "open_end"  # Map to prompt key
        elif self.task == "long_speech_detection":
            prompt_key = "long_speech"  # Map to prompt key
        elif self.task == "pronoun_detection":
            prompt_key = "use_pronoun"  # Map to prompt key
        
        prompts = self.config.get('prompts', {})
        if prompt_key in prompts:
            return prompts[prompt_key]['initial']
        else:
            # Use the prompt from the detector if available
            if self.task == "pronoun_detection" and self.task_type == "use_pronoun":
                from src.detectors.use_pronoun import UsePronounDetector
                detector = UsePronounDetector()
                return detector.prompt
            self.logger.warning(f"No prompt configured for {prompt_key}")
            return "Classify the text."
    
    def create_stratified_folds(self, dataset: List[Dict]) -> List[List[Dict]]:
        """Create stratified k-folds"""
        cv_config = self.config.get('cross_validation', {})
        standard_config = cv_config.get('standard', {})
        n_folds = standard_config.get('n_folds', 3)
        
        # Group by label
        label_groups = defaultdict(list)
        for item in dataset:
            label_groups[item['label']].append(item)
        
        # Create folds
        folds = [[] for _ in range(n_folds)]
        
        # Distribute each label group across folds
        for label, items in label_groups.items():
            random.shuffle(items)
            for i, item in enumerate(items):
                folds[i % n_folds].append(item)
        
        # Shuffle each fold
        for fold in folds:
            random.shuffle(fold)
        
        return folds
    
    def run_cross_validation(self, dataset: List[Dict]) -> Dict:
        """Run full cross-validation"""
        cv_config = self.config.get('cross_validation', {})
        
        # Use standard config by default
        standard_config = cv_config.get('standard', {})
        n_folds = standard_config.get('n_folds', 3)
        n_repeats = standard_config.get('n_repeats', 3)
        val_split = cv_config.get('validation_split_ratio', 0.2)
        
        self.logger.info(f"Starting {n_folds}-fold cross-validation with {n_repeats} repeats")
        
        all_results = []
        all_predictions = []
        
        for repeat in range(1, n_repeats + 1):
            self.logger.info(f"\n=== REPEAT {repeat}/{n_repeats} ===")
            
            # Set random seed for reproducibility
            random_seed_base = standard_config.get('random_seed_base', 42)
            random.seed(random_seed_base + repeat)
            
            # Create stratified folds
            folds = self.create_stratified_folds(dataset)
            
            for fold_idx in range(n_folds):
                # Prepare train/test split
                test_fold = folds[fold_idx]
                train_folds = [f for i, f in enumerate(folds) if i != fold_idx]
                train_data = [item for fold in train_folds for item in fold]
                
                # Split train into train/val
                random.shuffle(train_data)
                val_size = int(len(train_data) * val_split)
                val_data = train_data[:val_size]
                train_data_final = train_data[val_size:]
                
                # Run fold
                fold_results = self.run_fold(
                    fold_num=fold_idx + 1,
                    train_data=train_data_final,
                    val_data=val_data,
                    test_data=test_fold
                )
                fold_results['repeat'] = repeat
                
                all_results.append(fold_results)
                all_predictions.extend(fold_results['predictions'])
        
        # Aggregate results
        aggregate_metrics = self._aggregate_metrics(all_results)
        
        # Analyze errors
        errors = self._analyze_errors(all_predictions)
        
        # Prepare final results
        results = {
            'task': f"{self.task}_{self.task_type}",
            'dataset_stats': {
                'total': len(dataset),
                'label_distribution': dict(self._count_labels(dataset))
            },
            'cross_validation_config': cv_config,
            'fold_results': all_results,
            'aggregate_metrics': aggregate_metrics,
            'errors': errors,
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def _count_labels(self, dataset: List[Dict]) -> Dict[str, int]:
        """Count label distribution"""
        counts = defaultdict(int)
        for item in dataset:
            counts[item['label']] += 1
        return counts
    
    def _aggregate_metrics(self, fold_results: List[Dict]) -> Dict:
        """Aggregate metrics across folds"""
        # Collect all metrics
        metric_names = ['accuracy', 'precision_harmful', 'recall_harmful', 'f1_harmful',
                       'precision_safe', 'recall_safe', 'f1_safe',
                       'precision_1', 'recall_1', 'f1_1',  # For episode memory
                       'precision_0', 'recall_0', 'f1_0']
        
        aggregated = {}
        
        for metric in metric_names:
            values = []
            for result in fold_results:
                # Try test_accuracy first, then metrics
                if metric == 'accuracy':
                    values.append(result['test_accuracy'])
                elif metric in result['metrics']:
                    values.append(result['metrics'][metric])
            
            if values:
                aggregated[metric] = {
                    'mean': sum(values) / len(values),
                    'std': (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5,
                    'min': min(values),
                    'max': max(values),
                    'values': values
                }
        
        return aggregated
    
    def _analyze_errors(self, all_predictions: List[Dict]) -> Dict:
        """Analyze errors across all predictions"""
        errors = [p for p in all_predictions if not p['correct']]
        
        # Group errors by type
        error_types = defaultdict(list)
        for error in errors:
            error_type = f"{error['true']}_as_{error['pred']}"
            error_types[error_type].append(error)
        
        # Count unique error texts
        unique_errors = {}
        for error in errors:
            text = error['input']
            if text not in unique_errors:
                unique_errors[text] = {
                    'text': text,
                    'true_label': error['true'],
                    'predictions': defaultdict(int)
                }
            unique_errors[text]['predictions'][error['pred']] += 1
        
        return {
            'total': len(errors),
            'by_type': {k: len(v) for k, v in error_types.items()},
            'sample_errors': list(unique_errors.values())[:20],  # Top 20 unique errors
            'error_rate': len(errors) / len(all_predictions) if all_predictions else 0
        }
    
    def save_results(self, results: Dict, output_file: str):
        """Save results to file in timestamped directory"""
        # Create timestamped directory (no task name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(self.config['output']['results_dir'], timestamp)
        os.makedirs(results_dir, exist_ok=True)
        
        # Save main results JSON
        output_path = os.path.join(results_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Results saved to {output_path}")
        
        # Generate reports using UnifiedReportGenerator
        # reporter = UnifiedReportGenerator(results)
        # reporter.generate_all_reports(results_dir)
        self.logger.info("Report generation skipped (matplotlib not available)")
        
        # Save prompts
        self._save_prompts(results, results_dir)
        
        return results_dir
    
    def _save_prompts(self, results: Dict, results_dir: str):
        """Save prompt information"""
        prompts_info = {
            'initial_prompt': None,
            'optimized_prompts': [],
            'optimization_history': []
        }
        
        # Extract prompt info from fold results
        for fold_result in results.get('fold_results', []):
            if prompts_info['initial_prompt'] is None:
                prompts_info['initial_prompt'] = fold_result['prompts']['initial']
            
            if fold_result['prompts']['changed']:
                prompts_info['optimized_prompts'].append({
                    'fold': fold_result['fold'],
                    'repeat': fold_result.get('repeat', 1),
                    'prompt': fold_result['prompts']['optimized'],
                    'validation_accuracy': fold_result.get('validation_accuracy')
                })
        
        # Save prompt comparison
        prompt_path = os.path.join(results_dir, 'prompt_analysis.txt')
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write("PROMPT ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            f.write("INITIAL PROMPT:\n")
            f.write("-"*40 + "\n")
            f.write(prompts_info['initial_prompt'] + "\n\n")
            
            if prompts_info['optimized_prompts']:
                f.write("OPTIMIZED PROMPTS:\n")
                f.write("-"*40 + "\n")
                for opt in prompts_info['optimized_prompts']:
                    f.write(f"\nFold {opt['fold']} (Repeat {opt['repeat']}):\n")
                    f.write(f"Validation Accuracy: {opt['validation_accuracy']:.2%}\n")
                    f.write(opt['prompt'] + "\n")
            else:
                f.write("No prompt optimization performed (initial prompt met all thresholds)\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Unified cross-validation evaluation script'
    )
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--task',
        choices=['elderspeak', 'episode_detection', 'question_detection', 'long_speech_detection'],
        default='elderspeak',
        help='Task to evaluate'
    )
    parser.add_argument(
        '--type',
        default='ToE',
        help='Task type (e.g., ToE or collective for elderspeak, episode_memory for episode_detection)'
    )
    parser.add_argument(
        '--dataset',
        help='Path to dataset (overrides config)'
    )
    parser.add_argument(
        '--output',
        default='crossvalidation_results.json',
        help='Output filename'
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = UnifiedCrossValidator(args.config, args.task, args.type)
    
    # Load dataset
    dataset = evaluator.load_dataset(args.dataset)
    
    # Run cross-validation
    results = evaluator.run_cross_validation(dataset)
    
    # Save results
    results_dir = evaluator.save_results(results, args.output)
    
    # Print summary
    print("\n" + "="*60)
    print("CROSS-VALIDATION COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {results_dir}")
    
    agg_metrics = results['aggregate_metrics']
    
    print(f"\nOverall Performance:")
    if 'accuracy' in agg_metrics:
        print(f"  Accuracy: {agg_metrics['accuracy']['mean']:.2%} ± "
              f"{agg_metrics['accuracy']['std']:.2%}")
    
    # Print relevant metrics based on task
    if args.task == "elderspeak":
        for metric in ['precision_harmful', 'recall_harmful', 'f1_harmful']:
            if metric in agg_metrics:
                print(f"  {metric.replace('_', ' ').title()}: "
                      f"{agg_metrics[metric]['mean']:.3f} ± "
                      f"{agg_metrics[metric]['std']:.3f}")
    else:  # episode_detection
        for metric in ['precision_1', 'recall_1', 'f1_1']:
            if metric in agg_metrics:
                label = metric.replace('_1', ' (Episodic)')
                print(f"  {label.replace('_', ' ').title()}: "
                      f"{agg_metrics[metric]['mean']:.3f} ± "
                      f"{agg_metrics[metric]['std']:.3f}")
    
    print(f"\nTotal Errors: {results['errors']['total']}")
    print(f"  Error Rate: {results['errors']['error_rate']:.2%}")
    
    # Show error breakdown
    if results['errors']['by_type']:
        print("\n  Error Breakdown:")
        for error_type, count in results['errors']['by_type'].items():
            print(f"    {error_type}: {count}")


if __name__ == "__main__":
    main()