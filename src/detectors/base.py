"""
Base Detector Class - Abstract interface for all detection modules
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
from pathlib import Path
import yaml

from ..models.openai_client import OpenAIClient
from ..utils.metrics import calculate_metrics


@dataclass
class DetectionResult:
    """Structured result from detection"""
    text: str
    prediction: str
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DetectorConfig:
    """Configuration for a detector"""
    name: str
    task_key: str
    type_key: str
    description: str
    labels: List[str]
    label_mapping: Optional[Dict[str, str]] = None
    

class BaseDetector(ABC):
    """Abstract base class for all detectors"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize detector with configuration"""
        self.config = self._load_config(config_path)
        self.client = OpenAIClient()
        self.logger = self._setup_logging()
        self.detector_config = self._get_detector_config()
        self._prompt = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
            
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the detector"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        return logger
        
    @abstractmethod
    def _get_detector_config(self) -> DetectorConfig:
        """Get detector-specific configuration"""
        pass
        
    @property
    def prompt(self) -> str:
        """Get the current prompt (initial or optimized)"""
        if self._prompt is None:
            self._prompt = self._get_initial_prompt()
        return self._prompt
        
    @prompt.setter
    def prompt(self, value: str):
        """Set optimized prompt"""
        self._prompt = value
        
    @abstractmethod
    def _get_initial_prompt(self) -> str:
        """Get initial prompt from configuration"""
        pass
        
    def detect(self, text: str) -> DetectionResult:
        """Perform detection on a single text"""
        try:
            # Get model from config
            model = self.config['models']['default']
            
            # Call API
            response = self.client.predict(
                text=text,
                prompt=self.prompt,
                model=model
            )
            
            # Normalize response
            prediction = self._normalize_response(response)
            
            return DetectionResult(
                text=text,
                prediction=prediction,
                metadata={'raw_response': response}
            )
            
        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            raise
            
    def _normalize_response(self, response: str) -> str:
        """Normalize API response to expected label"""
        response = response.strip()
        
        if response in self.detector_config.labels:
            return response
            
        # Handle common variations
        if response.lower() in ['true', '1', 'yes']:
            return '1'
        elif response.lower() in ['false', '0', 'no']:
            return '0'
            
        self.logger.warning(f"Unexpected response: {response}, defaulting to {self.detector_config.labels[0]}")
        return self.detector_config.labels[0]
        
    def detect_batch(self, texts: List[str], show_progress: bool = True) -> List[DetectionResult]:
        """Perform detection on multiple texts"""
        results = []
        
        for i, text in enumerate(texts):
            if show_progress and i % 10 == 0:
                self.logger.info(f"Progress: {i}/{len(texts)}")
                
            result = self.detect(text)
            results.append(result)
            
        return results
        
    def evaluate(self, dataset: List[Dict[str, str]], 
                 label_field: str = 'label') -> Dict[str, Any]:
        """Evaluate detector on a labeled dataset"""
        predictions = []
        
        for item in dataset:
            result = self.detect(item['text'])
            predictions.append({
                'text': item['text'],
                'true_label': item[label_field],
                'predicted_label': result.prediction,
                'correct': item[label_field] == result.prediction
            })
            
        # Calculate metrics
        metrics = calculate_metrics(predictions)
        
        return {
            'predictions': predictions,
            'metrics': metrics,
            'accuracy': sum(p['correct'] for p in predictions) / len(predictions)
        }
        
    def save_prompt(self, filepath: str):
        """Save current prompt to file"""
        prompt_data = {
            'detector': self.__class__.__name__,
            'name': self.detector_config.name,
            'prompt': self.prompt,
            'is_optimized': self._prompt is not None
        }
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(prompt_data, f, ensure_ascii=False, indent=2)
            
    def load_prompt(self, filepath: str):
        """Load prompt from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if data['detector'] != self.__class__.__name__:
            raise ValueError(f"Prompt file is for {data['detector']}, not {self.__class__.__name__}")
            
        self.prompt = data['prompt']
        self.logger.info(f"Loaded {'optimized' if data['is_optimized'] else 'initial'} prompt")