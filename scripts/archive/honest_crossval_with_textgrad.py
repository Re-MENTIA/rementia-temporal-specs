#!/usr/bin/env python3
"""
HONEST cross-validation with REAL TextGrad optimization
This script:
1. Runs cross-validation with ORIGINAL prompts
2. Optimizes prompts using TextGrad
3. Runs cross-validation with OPTIMIZED prompts
4. Generates honest comparison reports
"""

import os
import sys
import json
import logging
import argparse
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.model_selection import StratifiedKFold

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from textgrad_optimize_prompts import TextGradPromptOptimizer
from src.utils.data_loader import DataLoader
from src.models.openai_judge import OpenAIJudge
from src.utils.metrics import calculate_metrics
from src.utils.unified_reporting import generate_comparison_report


class HonestCrossValidator:
    """Honest cross-validation with before/after TextGrad comparison"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.loader = DataLoader(self.config)
        
        # Load API key
        from dotenv import load_dotenv
        load_dotenv()
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found")
    
    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger
    
    def evaluate_dataset(self, dataset: List[Dict], prompt: str, 
                        detector_type: str, show_progress: bool = True) -> Tuple[float, List[Dict]]:
        """Evaluate dataset with given prompt"""
        judge = OpenAIJudge(self.config)
        predictions = []
        correct = 0
        
        for i, item in enumerate(dataset):
            if show_progress and i % 10 == 0:
                self.logger.info(f"Progress: {i}/{len(dataset)}")
            
            # Get text and label based on detector type
            text, true_label = self._extract_text_and_label(item, detector_type)
            
            # Get prediction
            try:
                prediction = judge.judge(text, prompt)
                is_correct = prediction == true_label
                
                predictions.append({
                    'input': text,
                    'true': true_label,
                    'pred': prediction,
                    'correct': is_correct
                })
                
                if is_correct:
                    correct += 1
                    
            except Exception as e:
                self.logger.error(f"Error: {e}")
                predictions.append({
                    'input': text,
                    'true': true_label,
                    'pred': '0',
                    'correct': False,
                    'error': str(e)
                })
        
        accuracy = correct / len(dataset) if dataset else 0
        return accuracy, predictions
    
    def _extract_text_and_label(self, item: Dict, detector_type: str) -> Tuple[str, str]:
        """Extract text and label from dataset item"""
        # Get text
        if detector_type in ['ToE', 'collective']:
            text = item.get('sentence', item.get('text', ''))
            label_key = 'label'
        elif detector_type == 'episode_memory':
            text = item.get('sentence', '')
            label_key = 'is_epsodic_memory_related'
        elif detector_type == 'open_end':
            text = item.get('sentence', '')
            label_key = 'question_type'
        elif detector_type == 'long_speech':
            text = item.get('sentences', '')
            label_key = 'is_too_long?'
        else:
            text = ''
            label_key = 'label'
        
        # Get label
        true_label = str(item.get(label_key, ''))
        
        # Apply label mapping
        label_mapping = self._get_label_mapping(detector_type)
        if label_mapping and true_label in label_mapping:
            true_label = label_mapping[true_label]
        
        return text, true_label
    
    def _get_label_mapping(self, detector_type: str) -> Dict:
        """Get label mapping for detector type"""
        if detector_type == 'collective':
            return {
                'collective': '1',
                'non-collective': '0',
                'neutral': '0'
            }
        elif detector_type == 'episode_memory':
            return {
                'true': '1',
                'false': '0',
                'neutral': '0'
            }
        elif detector_type == 'open_end':
            return {
                'open-ended': '1',
                'closed-ended': '0',
                'neutral': '0'
            }
        elif detector_type == 'long_speech':
            return {
                'True': '1',
                'False': '0'
            }
        return {}
    
    def run_cross_validation(self, dataset: List[Dict], prompt: str, 
                           detector_type: str, n_folds: int = 3, 
                           n_repeats: int = 3) -> Dict:
        """Run stratified k-fold cross-validation"""
        # Get labels for stratification
        labels = []
        for item in dataset:
            _, label = self._extract_text_and_label(item, detector_type)
            labels.append(label)
        
        # Convert to numpy array
        X = np.array(dataset)
        y = np.array(labels)
        
        all_fold_results = []
        
        # Repeat cross-validation
        for repeat in range(n_repeats):
            self.logger.info(f"\nRepeat {repeat + 1}/{n_repeats}")
            
            # Create stratified k-fold
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, 
                                random_state=42 + repeat)
            
            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                self.logger.info(f"Fold {fold + 1}/{n_folds}")
                
                test_data = X[test_idx].tolist()
                
                # Evaluate on test set
                test_acc, predictions = self.evaluate_dataset(
                    test_data, prompt, detector_type
                )
                
                # Calculate metrics
                metrics = calculate_metrics(predictions)
                
                fold_result = {
                    'repeat': repeat + 1,
                    'fold': fold + 1,
                    'test_size': len(test_data),
                    'accuracy': test_acc,
                    'metrics': metrics,
                    'errors': [p for p in predictions if not p['correct']]
                }
                
                all_fold_results.append(fold_result)
                
                self.logger.info(f"Accuracy: {test_acc:.2%}")
        
        # Calculate average metrics
        accuracies = [r['accuracy'] for r in all_fold_results]
        
        return {
            'fold_results': all_fold_results,
            'average_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies)
        }
    
    def run_full_evaluation(self, detector_type: str, detector_name: str) -> Dict:
        """Run full evaluation: before optimization, optimization, after optimization"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Evaluating {detector_name}")
        self.logger.info('='*60)
        
        # Load dataset
        if detector_type == 'ToE':
            dataset = self.loader.load_terms_of_endearment_data()
        elif detector_type == 'collective':
            dataset = self.loader.load_collective_instruction_data()
        elif detector_type == 'episode_memory':
            dataset = self.loader.load_episode_memory_data()
        elif detector_type == 'open_end':
            dataset = self.loader.load_open_end_question_data()
        elif detector_type == 'long_speech':
            dataset = self.loader.load_long_speech_data()
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
        
        self.logger.info(f"Loaded {len(dataset)} samples")
        
        # Get initial prompt
        prompt_key = {
            'ToE': 'ToE',
            'collective': 'collective',
            'episode_memory': 'episode_memory',
            'open_end': 'open_end',
            'long_speech': 'long_speech'
        }[detector_type]
        
        initial_prompt = self.config['prompts'][prompt_key]['initial']
        
        # 1. Cross-validation with ORIGINAL prompt
        self.logger.info("\n--- BEFORE OPTIMIZATION ---")
        before_results = self.run_cross_validation(
            dataset, initial_prompt, detector_type
        )
        self.logger.info(f"Average accuracy: {before_results['average_accuracy']:.2%} "
                        f"(±{before_results['std_accuracy']:.2%})")
        
        # 2. Optimize prompt with TextGrad
        self.logger.info("\n--- TEXTGRAD OPTIMIZATION ---")
        optimizer = TextGradPromptOptimizer(self.config_path, self.api_key)
        
        # Split data for optimization
        from sklearn.model_selection import train_test_split
        train_data, val_data = train_test_split(
            dataset, test_size=0.2, random_state=42,
            stratify=[self._extract_text_and_label(d, detector_type)[1] for d in dataset]
        )
        
        optimized_prompt, opt_initial_acc, opt_final_acc, opt_history = \
            optimizer.optimize_prompt(
                initial_prompt, detector_type, train_data, val_data, n_iterations=5
            )
        
        # 3. Cross-validation with OPTIMIZED prompt
        self.logger.info("\n--- AFTER OPTIMIZATION ---")
        after_results = self.run_cross_validation(
            dataset, optimized_prompt, detector_type
        )
        self.logger.info(f"Average accuracy: {after_results['average_accuracy']:.2%} "
                        f"(±{after_results['std_accuracy']:.2%})")
        
        # 4. Compile results
        return {
            'detector_type': detector_type,
            'detector_name': detector_name,
            'dataset_size': len(dataset),
            'prompts': {
                'original': initial_prompt,
                'optimized': optimized_prompt,
                'changed': initial_prompt != optimized_prompt
            },
            'optimization': {
                'initial_validation_acc': opt_initial_acc,
                'final_validation_acc': opt_final_acc,
                'improvement': opt_final_acc - opt_initial_acc,
                'history': opt_history
            },
            'cross_validation': {
                'before': before_results,
                'after': after_results,
                'improvement': after_results['average_accuracy'] - before_results['average_accuracy']
            }
        }
    
    def __init__(self, config_path: str):
        self.config_path = config_path  # Store config path
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.loader = DataLoader(self.config)
        
        # Load API key
        from dotenv import load_dotenv
        load_dotenv()
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found")


def generate_honest_report(results: Dict, output_dir: Path):
    """Generate honest comparison report"""
    report_lines = [
        "# HONEST TextGrad Optimization Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Summary",
        "\n| Detector | Before CV | After CV | Improvement | Optimization Improvement |",
        "|----------|-----------|----------|-------------|-------------------------|"
    ]
    
    for detector_type, result in results.items():
        if 'error' in result:
            continue
        
        before_acc = result['cross_validation']['before']['average_accuracy']
        after_acc = result['cross_validation']['after']['average_accuracy']
        cv_improvement = result['cross_validation']['improvement']
        opt_improvement = result['optimization']['improvement']
        
        report_lines.append(
            f"| {result['detector_name']} | "
            f"{before_acc:.2%} | "
            f"{after_acc:.2%} | "
            f"{cv_improvement:+.2%} | "
            f"{opt_improvement:+.2%} |"
        )
    
    # Add detailed results
    report_lines.extend([
        "\n## Detailed Results",
        ""
    ])
    
    for detector_type, result in results.items():
        if 'error' in result:
            report_lines.append(f"\n### {result.get('detector_name', detector_type)}")
            report_lines.append(f"ERROR: {result['error']}")
            continue
        
        report_lines.extend([
            f"\n### {result['detector_name']}",
            f"\n**Dataset Size**: {result['dataset_size']} samples",
            f"\n**Prompt Changed**: {'Yes' if result['prompts']['changed'] else 'No'}",
            "\n**Cross-Validation Results**:",
            f"- Before: {result['cross_validation']['before']['average_accuracy']:.2%} "
            f"(±{result['cross_validation']['before']['std_accuracy']:.2%})",
            f"- After: {result['cross_validation']['after']['average_accuracy']:.2%} "
            f"(±{result['cross_validation']['after']['std_accuracy']:.2%})",
            f"- Improvement: {result['cross_validation']['improvement']:+.2%}",
            "\n**Optimization Process**:",
            f"- Initial validation accuracy: {result['optimization']['initial_validation_acc']:.2%}",
            f"- Final validation accuracy: {result['optimization']['final_validation_acc']:.2%}",
            f"- Optimization improvement: {result['optimization']['improvement']:+.2%}",
            ""
        ])
        
        # Add prompt comparison if changed
        if result['prompts']['changed']:
            report_lines.extend([
                "**Prompt Changes**:",
                "",
                "*Original prompt excerpt:*",
                "```",
                result['prompts']['original'][:500] + "...",
                "```",
                "",
                "*Optimized prompt excerpt:*",
                "```",
                result['prompts']['optimized'][:500] + "...",
                "```",
                ""
            ])
    
    # Write report
    report_path = output_dir / "honest_comparison_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    return report_path


def main():
    parser = argparse.ArgumentParser(description='Honest cross-validation with TextGrad')
    parser.add_argument('--config', default='config/config.yaml', help='Config file')
    parser.add_argument('--output', default='results/honest_evaluation/', help='Output directory')
    parser.add_argument('--detectors', nargs='+', 
                       default=['ToE', 'collective', 'episode_memory', 'open_end', 'long_speech'],
                       help='Detectors to evaluate')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir = Path(args.output) / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup log file
    log_file = output_dir / 'honest_evaluation.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    # Create evaluator
    evaluator = HonestCrossValidator(args.config)
    
    # Define detectors
    detector_info = {
        'ToE': 'Terms of Endearment',
        'collective': 'Collective Instruction',
        'episode_memory': 'Episode Memory Detection',
        'open_end': 'Open-End Question Detection',
        'long_speech': 'Long Speech Detection'
    }
    
    # Run evaluations
    all_results = {}
    
    for detector_type in args.detectors:
        if detector_type not in detector_info:
            logger.warning(f"Unknown detector: {detector_type}")
            continue
        
        try:
            result = evaluator.run_full_evaluation(
                detector_type, detector_info[detector_type]
            )
            all_results[detector_type] = result
            
            # Save individual result
            result_file = output_dir / f"{detector_type}_honest_results.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Error evaluating {detector_type}: {e}")
            import traceback
            traceback.print_exc()
            all_results[detector_type] = {'error': str(e)}
    
    # Save all results
    all_results_file = output_dir / "all_honest_results.json"
    with open(all_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # Generate report
    report_path = generate_honest_report(all_results, output_dir)
    
    logger.info(f"\n{'='*60}")
    logger.info("HONEST EVALUATION COMPLETE")
    logger.info('='*60)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Report: {report_path}")


if __name__ == "__main__":
    main()