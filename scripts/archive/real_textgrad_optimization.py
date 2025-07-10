#!/usr/bin/env python3
"""
REAL TextGrad implementation for eldercare detection prompt optimization
Using the actual TextGrad library, not custom implementation
"""

import os
import sys
import json
import logging
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import random

def train_test_split(data, test_size=0.2, random_state=42):
    """Simple train test split implementation without numpy"""
    random.seed(random_state)
    n = len(data)
    test_n = int(n * test_size)
    
    indices = list(range(n))
    random.shuffle(indices)
    test_indices = indices[:test_n]
    train_indices = indices[test_n:]
    
    train_data = [data[i] for i in train_indices]
    test_data = [data[i] for i in test_indices]
    
    return train_data, test_data

# Fake tqdm if not available
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable
import concurrent.futures

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# TextGrad imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'textgrad'))
import textgrad as tg
from textgrad.engine import get_engine
from textgrad import Variable, TextualGradientDescent, set_backward_engine, BlackboxLLM, sum as tg_sum

from src.utils.data_loader import DataLoader
from src.models.openai_client import OpenAIClient


class ElderCarePromptOptimizer:
    """Real TextGrad optimizer for eldercare detection prompts"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Load API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        
        # Initialize TextGrad engines
        self.eval_engine = get_engine("gpt-4o-mini")
        self.test_engine = get_engine("gpt-4o-mini")
        set_backward_engine(self.eval_engine, override=True)
        
    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger
    
    def create_evaluation_function(self, detector_type: str):
        """Create evaluation function for specific detector type"""
        
        def eval_fn(inputs: Dict) -> Variable:
            """Evaluation function that returns loss (0 for correct, 1 for incorrect)"""
            prediction = inputs['prediction'].value.strip()
            ground_truth = inputs['ground_truth_answer'].value.strip()
            
            # Clean predictions (sometimes model adds extra text)
            if prediction.startswith('"') and prediction.endswith('"'):
                prediction = prediction[1:-1]
            prediction = prediction.split()[0] if prediction else "0"
            
            # Check if correct
            is_correct = prediction == ground_truth
            
            # Create loss feedback
            if is_correct:
                feedback = "Correct prediction."
                loss_value = 0
            else:
                feedback = f"Incorrect. Predicted '{prediction}' but expected '{ground_truth}'."
                loss_value = 1
            
            return Variable(
                value=str(loss_value),
                role_description="evaluation result (0=correct, 1=incorrect)"
            )
        
        return eval_fn
    
    def prepare_dataset(self, dataset: List[Dict], detector_type: str) -> List[Tuple[str, str]]:
        """Convert dataset to (input, label) tuples"""
        prepared = []
        
        for item in dataset:
            # Extract text and label based on detector type
            if detector_type in ['ToE', 'collective']:
                text = item.get('sentence', item.get('text', ''))
                label_key = 'label' if 'label' in item else 'instruction_type'
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
            
            # Get label
            label = str(item.get(label_key, ''))
            
            # Apply label mapping
            label_mapping = self._get_label_mapping(detector_type)
            if label_mapping and label in label_mapping:
                label = label_mapping[label]
            
            prepared.append((text, label))
        
        return prepared
    
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
    
    def eval_sample(self, item: Tuple[str, str], eval_fn, model) -> int:
        """Evaluate a single sample"""
        x, y = item
        x_var = Variable(x, requires_grad=False, role_description="input text to analyze")
        y_var = Variable(y, requires_grad=False, role_description="correct classification (0 or 1)")
        
        response = model(x_var)
        eval_output = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y_var))
        
        return int(eval_output.value)
    
    def eval_dataset(self, dataset: List[Tuple[str, str]], eval_fn, model, 
                    max_samples: int = None, num_threads: int = 8) -> List[int]:
        """Evaluate entire dataset"""
        if max_samples is None:
            max_samples = len(dataset)
        
        accuracy_list = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for sample in dataset[:max_samples]:
                future = executor.submit(self.eval_sample, sample, eval_fn, model)
                futures.append(future)
            
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(futures), desc="Evaluating"):
                acc_item = future.result()
                accuracy_list.append(acc_item)
        
        return accuracy_list
    
    def optimize_prompt(self, initial_prompt: str, detector_type: str,
                       train_data: List[Dict], val_data: List[Dict],
                       batch_size: int = 5, max_epochs: int = 3) -> Dict:
        """
        Optimize prompt using TextGrad
        
        Returns dictionary with optimization results
        """
        self.logger.info(f"Starting TextGrad optimization for {detector_type}")
        
        # Prepare datasets
        train_set = self.prepare_dataset(train_data, detector_type)
        val_set = self.prepare_dataset(val_data, detector_type)
        
        self.logger.info(f"Train: {len(train_set)}, Val: {len(val_set)}")
        
        # Create evaluation function
        eval_fn = self.create_evaluation_function(detector_type)
        
        # Create prompt variable
        system_prompt = Variable(
            initial_prompt,
            requires_grad=True,
            role_description="system prompt for detecting harmful communication patterns in eldercare"
        )
        
        # Create model
        model = BlackboxLLM(self.test_engine, system_prompt)
        
        # Create optimizer
        optimizer = TextualGradientDescent(
            engine=self.eval_engine,
            parameters=[system_prompt]
        )
        
        # Initialize results
        results = {
            "initial_prompt": initial_prompt,
            "validation_acc": [],
            "prompts": [],
            "train_loss": []
        }
        
        # Initial evaluation
        initial_val_results = self.eval_dataset(val_set, eval_fn, model)
        initial_val_acc = 1 - np.mean(initial_val_results)  # Convert loss to accuracy
        results["validation_acc"].append(initial_val_acc)
        results["prompts"].append(system_prompt.value)
        
        self.logger.info(f"Initial validation accuracy: {initial_val_acc:.2%}")
        
        # Training loop
        for epoch in range(max_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{max_epochs}")
            
            # Shuffle training data
            np.random.shuffle(train_set)
            
            # Process in batches
            epoch_losses = []
            
            for i in range(0, len(train_set), batch_size):
                batch = train_set[i:i + batch_size]
                
                optimizer.zero_grad()
                losses = []
                
                # Process each sample in batch
                for x, y in batch:
                    x_var = Variable(x, requires_grad=False, 
                                       role_description="input text to analyze")
                    y_var = Variable(y, requires_grad=False, 
                                       role_description="correct classification")
                    
                    # Get model response
                    response = model(x_var)
                    
                    # Evaluate response
                    eval_output = eval_fn(inputs=dict(
                        prediction=response,
                        ground_truth_answer=y_var
                    ))
                    
                    # If incorrect, add to losses
                    if eval_output.value == "1":
                        # Create detailed loss message
                        loss_text = f"For input '{x[:50]}...', the model predicted '{response.value.strip()}' but the correct answer is '{y}'"
                        loss_var = Variable(
                            loss_text,
                            role_description="loss information for incorrect prediction"
                        )
                        losses.append(loss_var)
                
                # If there were errors, compute gradients
                if losses:
                    # Aggregate losses
                    total_loss = tg_sum(losses)
                    
                    # Backward pass
                    total_loss.backward()
                    
                    # Update prompt
                    optimizer.step()
                    
                    epoch_losses.extend([1] * len(losses))
                    epoch_losses.extend([0] * (len(batch) - len(losses)))
                else:
                    epoch_losses.extend([0] * len(batch))
                
                # Log batch results
                batch_acc = 1 - (len(losses) / len(batch))
                self.logger.info(f"Batch {i//batch_size + 1}: Accuracy {batch_acc:.2%}")
            
            # Evaluate on validation set
            val_results = self.eval_dataset(val_set, eval_fn, model, max_samples=50)
            val_acc = 1 - (sum(val_results) / len(val_results) if val_results else 0)
            
            results["validation_acc"].append(val_acc)
            results["prompts"].append(system_prompt.value)
            results["train_loss"].append(sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0)
            
            self.logger.info(f"Epoch {epoch + 1} - Train loss: {sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0:.3f}, "
                           f"Val accuracy: {val_acc:.2%}")
            
            # Early stopping if validation accuracy decreases
            if len(results["validation_acc"]) > 2 and \
               results["validation_acc"][-1] < results["validation_acc"][-2]:
                self.logger.info("Early stopping: validation accuracy decreased")
                break
        
        # Select best prompt based on validation accuracy
        best_idx = results["validation_acc"].index(max(results["validation_acc"]))
        results["best_prompt"] = results["prompts"][best_idx]
        results["best_validation_acc"] = results["validation_acc"][best_idx]
        results["improvement"] = results["best_validation_acc"] - initial_val_acc
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Real TextGrad optimization')
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--output', default='results/textgrad_optimization/')
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--max-epochs', type=int, default=3)
    parser.add_argument('--validation-split', type=float, default=0.2)
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create optimizer
    optimizer = ElderCarePromptOptimizer(args.config)
    
    # Load data loader
    loader = DataLoader(optimizer.config)
    
    # Define detectors
    detectors = [
        ('ToE', 'Terms of Endearment', 'ToE'),
        ('collective', 'Collective Instruction', 'collective'),
        ('episode_memory', 'Episode Memory', 'episode_memory'),
        ('open_end', 'Open-End Questions', 'open_end'),
        ('long_speech', 'Long Speech', 'long_speech')
    ]
    
    all_results = {}
    
    for detector_key, detector_name, prompt_key in detectors:
        logger.info(f"\n{'='*60}")
        logger.info(f"Optimizing {detector_name}")
        logger.info('='*60)
        
        try:
            # Load dataset
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
            
            logger.info(f"Loaded {len(dataset)} samples")
            
            # Split data
            train_data, val_data = train_test_split(
                dataset, test_size=args.validation_split, random_state=42
            )
            
            # Get initial prompt
            initial_prompt = optimizer.config['prompts'][prompt_key]['initial']
            
            # Optimize
            results = optimizer.optimize_prompt(
                initial_prompt, detector_key, train_data, val_data,
                batch_size=args.batch_size, max_epochs=args.max_epochs
            )
            
            results['detector_name'] = detector_name
            results['dataset_size'] = len(dataset)
            
            all_results[detector_key] = results
            
            # Save individual result
            with open(output_dir / f"{detector_key}_optimization.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"\nOptimization complete for {detector_name}:")
            logger.info(f"Initial accuracy: {results['validation_acc'][0]:.2%}")
            logger.info(f"Best accuracy: {results['best_validation_acc']:.2%}")
            logger.info(f"Improvement: {results['improvement']:+.2%}")
            
        except Exception as e:
            logger.error(f"Error optimizing {detector_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[detector_key] = {'error': str(e)}
    
    # Save all results
    with open(output_dir / 'all_optimization_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()