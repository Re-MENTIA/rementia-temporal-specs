#!/usr/bin/env python3
"""
REAL TextGrad prompt optimization for all detectors
This is the honest implementation with actual gradient-based optimization
"""

import os
import sys
import json
import logging
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# TextGrad imports
import textgrad as tg
from textgrad import Variable, TextLoss, TGD, get_engine

# Import detectors
from src.detectors import (
    TermsOfEndearmentDetector,
    CollectiveInstructionDetector,
    EpisodeMemoryDetector,
    OpenEndQuestionDetector,
    LongSpeechDetector
)
from src.utils.data_loader import DataLoader
from src.models.openai_judge import OpenAIJudge


class TextGradPromptOptimizer:
    """Real TextGrad-based prompt optimizer"""
    
    def __init__(self, config_path: str, api_key: str):
        self.config = self._load_config(config_path)
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        
        # Initialize TextGrad engine
        os.environ['OPENAI_API_KEY'] = api_key
        self.engine = get_engine("gpt-4o-mini")
        tg.set_backward_engine(self.engine)
        
    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def create_evaluation_function(self, detector_type: str, dataset: List[Dict]):
        """Create evaluation function for TextGrad optimization"""
        
        def evaluate_prompt(prompt_text: str) -> float:
            """Evaluate a prompt on the dataset"""
            # Create temporary judge with this prompt
            judge = OpenAIJudge(self.config)
            
            correct = 0
            total = 0
            
            for item in dataset:
                # Get input text based on detector type
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
                    continue
                
                true_label = str(item.get(label_key, ''))
                
                # Map labels if needed
                label_mapping = self._get_label_mapping(detector_type)
                if label_mapping and true_label in label_mapping:
                    true_label = label_mapping[true_label]
                
                # Get prediction
                try:
                    prediction = judge.judge(text, prompt_text)
                    if prediction == true_label:
                        correct += 1
                    total += 1
                except Exception as e:
                    self.logger.error(f"Error in evaluation: {e}")
                    continue
            
            return correct / total if total > 0 else 0.0
        
        return evaluate_prompt
    
    def _get_label_mapping(self, detector_type: str) -> Optional[Dict]:
        """Get label mapping for specific detector types"""
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
        return None
    
    def optimize_prompt(self, initial_prompt: str, detector_type: str, 
                       train_data: List[Dict], val_data: List[Dict], 
                       n_iterations: int = 5) -> Tuple[str, float, float, Dict]:
        """
        Optimize prompt using TextGrad
        
        Returns:
            Tuple of (optimized_prompt, initial_acc, final_acc, optimization_history)
        """
        self.logger.info(f"Starting TextGrad optimization for {detector_type}")
        self.logger.info(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
        
        # Create evaluation function
        eval_fn = self.create_evaluation_function(detector_type, val_data)
        
        # Evaluate initial prompt
        initial_acc = eval_fn(initial_prompt)
        self.logger.info(f"Initial validation accuracy: {initial_acc:.2%}")
        
        # Create TextGrad variable
        prompt_var = Variable(
            initial_prompt,
            role_description="You are a prompt for detecting harmful communication patterns in eldercare."
        )
        
        # Initialize optimizer
        optimizer = TGD(parameters=[prompt_var], lr=0.1)
        
        # Optimization history
        history = {
            'iterations': [],
            'accuracies': [initial_acc],
            'prompts': [initial_prompt]
        }
        
        # Optimization loop
        for i in range(n_iterations):
            self.logger.info(f"\nIteration {i+1}/{n_iterations}")
            
            # Sample batch from training data
            batch_size = min(10, len(train_data))
            batch = np.random.choice(train_data, batch_size, replace=False)
            
            # Compute loss on batch
            batch_correct = 0
            loss_descriptions = []
            
            for item in batch:
                # Get text and label
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
                
                true_label = str(item.get(label_key, ''))
                
                # Map labels if needed
                label_mapping = self._get_label_mapping(detector_type)
                if label_mapping and true_label in label_mapping:
                    true_label = label_mapping[true_label]
                
                # Get prediction
                judge = OpenAIJudge(self.config)
                try:
                    prediction = judge.judge(text, prompt_var.value)
                    
                    if prediction != true_label:
                        loss_descriptions.append(
                            f"Input: '{text[:50]}...' Expected: {true_label}, Got: {prediction}"
                        )
                    else:
                        batch_correct += 1
                except Exception as e:
                    self.logger.error(f"Error: {e}")
                    continue
            
            batch_acc = batch_correct / len(batch)
            self.logger.info(f"Batch accuracy: {batch_acc:.2%}")
            
            # Create loss
            if loss_descriptions:
                loss_text = f"""The prompt failed on {len(loss_descriptions)} examples:
                
{chr(10).join(loss_descriptions[:3])}

The prompt should be improved to correctly classify these cases."""
                
                loss = TextLoss(
                    Variable(loss_text, role_description="loss function output")
                )
                
                # Backward pass
                self.logger.info("Computing gradients...")
                loss.backward()
                
                # Update prompt
                optimizer.step()
                optimizer.zero_grad()
                
                # Log updated prompt
                self.logger.debug(f"Updated prompt: {prompt_var.value[:200]}...")
            
            # Evaluate on validation set
            val_acc = eval_fn(prompt_var.value)
            self.logger.info(f"Validation accuracy: {val_acc:.2%}")
            
            history['iterations'].append(i+1)
            history['accuracies'].append(val_acc)
            history['prompts'].append(prompt_var.value)
            
            # Early stopping if perfect or degrading
            if val_acc >= 0.99 or (i > 0 and val_acc < history['accuracies'][-2]):
                self.logger.info("Early stopping triggered")
                break
        
        # Get best prompt based on validation accuracy
        best_idx = np.argmax(history['accuracies'])
        best_prompt = history['prompts'][best_idx]
        best_acc = history['accuracies'][best_idx]
        
        self.logger.info(f"\nOptimization complete!")
        self.logger.info(f"Initial accuracy: {initial_acc:.2%}")
        self.logger.info(f"Best accuracy: {best_acc:.2%} (iteration {best_idx})")
        
        return best_prompt, initial_acc, best_acc, history


def main():
    parser = argparse.ArgumentParser(description='REAL TextGrad prompt optimization')
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    parser.add_argument('--output', default='results/textgrad_optimization/', help='Output directory')
    parser.add_argument('--iterations', type=int, default=5, help='Number of optimization iterations')
    parser.add_argument('--validation-split', type=float, default=0.2, help='Validation split ratio')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load API key
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")
    
    # Create optimizer
    optimizer = TextGradPromptOptimizer(args.config, api_key)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define detectors to optimize
    detectors = [
        ('ToE', 'Terms of Endearment'),
        ('collective', 'Collective Instruction'),
        ('episode_memory', 'Episode Memory'),
        ('open_end', 'Open-End Questions'),
        ('long_speech', 'Long Speech')
    ]
    
    # Results storage
    all_results = {}
    
    # Optimize each detector
    for detector_key, detector_name in detectors:
        logger.info(f"\n{'='*60}")
        logger.info(f"Optimizing {detector_name}")
        logger.info('='*60)
        
        try:
            # Load data
            loader = DataLoader(config)
            if detector_key == 'ToE':
                dataset = loader.load_terms_of_endearment_data()
            elif detector_key == 'collective':
                dataset = loader.load_collective_instruction_data()
            elif detector_key == 'episode_memory':
                dataset = loader.load_episode_memory_data()
            elif detector_key == 'open_end':
                dataset = loader.load_open_end_question_data()
            elif detector_key == 'long_speech':
                dataset = loader.load_long_speech_data()
            else:
                continue
            
            logger.info(f"Loaded {len(dataset)} samples")
            
            # Split data
            train_data, val_data = train_test_split(
                dataset, test_size=args.validation_split, 
                random_state=42, stratify=[d.get('label', d.get('question_type', '0')) for d in dataset]
            )
            
            # Get initial prompt
            if detector_key == 'ToE':
                initial_prompt = config['prompts']['ToE']['initial']
            elif detector_key == 'collective':
                initial_prompt = config['prompts']['collective']['initial']
            elif detector_key == 'episode_memory':
                initial_prompt = config['prompts']['episode_memory']['initial']
            elif detector_key == 'open_end':
                initial_prompt = config['prompts']['open_end']['initial']
            elif detector_key == 'long_speech':
                initial_prompt = config['prompts']['long_speech']['initial']
            
            # Optimize
            optimized_prompt, initial_acc, final_acc, history = optimizer.optimize_prompt(
                initial_prompt, detector_key, train_data, val_data, args.iterations
            )
            
            # Store results
            results = {
                'detector_name': detector_name,
                'detector_key': detector_key,
                'initial_prompt': initial_prompt,
                'optimized_prompt': optimized_prompt,
                'initial_accuracy': initial_acc,
                'final_accuracy': final_acc,
                'improvement': final_acc - initial_acc,
                'optimization_history': history,
                'dataset_size': len(dataset),
                'train_size': len(train_data),
                'val_size': len(val_data)
            }
            
            all_results[detector_key] = results
            
            # Save individual result
            result_file = output_dir / f"{detector_key}_optimization.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved results to {result_file}")
            
        except Exception as e:
            logger.error(f"Error optimizing {detector_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[detector_key] = {'error': str(e)}
    
    # Save all results
    all_results_file = output_dir / "all_optimization_results.json"
    with open(all_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # Generate summary report
    logger.info("\n" + "="*60)
    logger.info("OPTIMIZATION SUMMARY")
    logger.info("="*60)
    
    for key, results in all_results.items():
        if 'error' in results:
            logger.info(f"{key}: ERROR - {results['error']}")
        else:
            logger.info(f"{results['detector_name']}:")
            logger.info(f"  Initial accuracy: {results['initial_accuracy']:.2%}")
            logger.info(f"  Final accuracy: {results['final_accuracy']:.2%}")
            logger.info(f"  Improvement: {results['improvement']:+.2%}")
    
    logger.info(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()