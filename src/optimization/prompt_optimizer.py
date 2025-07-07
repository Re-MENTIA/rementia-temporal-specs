"""
Prompt Optimizer - Optimizes prompts for detection tasks
"""

import json
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import yaml

from ..detectors.base import BaseDetector


class PromptOptimizer:
    """Optimizes prompts for detection tasks using validation performance"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize optimizer"""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
            
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        return logger
        
    def optimize(self, detector: BaseDetector, 
                 train_data: List[Dict], 
                 val_data: List[Dict]) -> Tuple[str, float]:
        """
        Optimize prompt based on validation performance
        
        Args:
            detector: Detector instance to optimize
            train_data: Training dataset for learning patterns
            val_data: Validation dataset for evaluation
            
        Returns:
            Tuple of (optimized_prompt, validation_accuracy)
        """
        # Evaluate with initial prompt
        initial_results = detector.evaluate(val_data)
        initial_accuracy = initial_results['accuracy']
        
        self.logger.info(f"Initial validation accuracy: {initial_accuracy:.2%}")
        
        # Check if optimization is needed
        opt_config = self.config['prompt_optimization']
        metrics = initial_results['metrics']
        
        if self._meets_thresholds(initial_accuracy, metrics, opt_config):
            self.logger.info("Initial prompt meets all thresholds, no optimization needed")
            return detector.prompt, initial_accuracy
            
        # Analyze errors
        errors = [p for p in initial_results['predictions'] if not p['correct']]
        
        if not errors:
            return detector.prompt, initial_accuracy
            
        # Simple optimization: Add clarifications based on error patterns
        optimized_prompt = self._create_optimized_prompt(
            detector.prompt, 
            errors,
            detector.detector_config.name
        )
        
        # For very long prompts (like open-end), don't add more text
        if len(detector.prompt) > 2000:
            self.logger.info("Prompt too long for optimization, keeping original")
            return detector.prompt, initial_accuracy
            
        return optimized_prompt, initial_accuracy
        
    def _meets_thresholds(self, accuracy: float, metrics: Dict, 
                         opt_config: Dict) -> bool:
        """Check if metrics meet optimization thresholds"""
        return (accuracy >= opt_config['accuracy_threshold'] and
                metrics.get('recall_1', metrics.get('recall_harmful', 1.0)) >= opt_config['recall_threshold'] and
                metrics.get('precision_1', metrics.get('precision_harmful', 1.0)) >= opt_config['precision_threshold'])
                
    def _create_optimized_prompt(self, initial_prompt: str, 
                                errors: List[Dict], 
                                detector_name: str) -> str:
        """Create optimized prompt based on error analysis"""
        # This is a simplified version - real implementation would use TextGrad
        error_types = self._analyze_error_patterns(errors)
        
        if not error_types:
            return initial_prompt
            
        # Add targeted clarifications
        clarification = "\n\nPay special attention to:\n"
        for error_type, count in error_types.items():
            clarification += f"- {error_type} (seen in {count} errors)\n"
            
        return initial_prompt + clarification
        
    def _analyze_error_patterns(self, errors: List[Dict]) -> Dict[str, int]:
        """Analyze patterns in errors"""
        patterns = {}
        
        for error in errors:
            # Simple pattern detection
            text = error['text'].lower()
            
            if '?' in text:
                patterns['questions'] = patterns.get('questions', 0) + 1
            if any(word in text for word in ['yesterday', '昨日', 'last', '先週']):
                patterns['temporal references'] = patterns.get('temporal references', 0) + 1
            if any(word in text for word in ['remember', '覚えて', 'recall']):
                patterns['memory references'] = patterns.get('memory references', 0) + 1
                
        return patterns
        
    def save_optimization_results(self, results: Dict[str, Dict], 
                                 output_path: str = "src/prompts/optimized_prompts.json"):
        """
        Save optimization results for all detectors
        
        Args:
            results: Dict mapping detector names to optimization results
            output_path: Path to save the results
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"Saved optimization results to {output_path}")