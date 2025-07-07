"""
Detection Pipeline - Orchestrates multiple detectors for comprehensive analysis
"""

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pathlib import Path

from ..detectors import (
    AccommodationSpeechDetector,
    EpisodeMemoryDetector, 
    OpenEndQuestionDetector,
    LongSpeechDetector
)


@dataclass
class PipelineResult:
    """Result from the detection pipeline"""
    text: str
    detections: Dict[str, Dict[str, Any]]
    risk_level: str  # 'low', 'medium', 'high'
    recommendations: List[str]
    

class DetectionPipeline:
    """Orchestrates multiple detectors for comprehensive eldercare communication analysis"""
    
    def __init__(self, optimized_prompts_path: Optional[str] = None):
        """Initialize pipeline with all detectors"""
        self.logger = self._setup_logging()
        
        # Initialize detectors
        self.detectors = {
            'accommodation': AccommodationSpeechDetector(),
            'episode_memory': EpisodeMemoryDetector(),
            'open_end_question': OpenEndQuestionDetector(),
            'long_speech': LongSpeechDetector()
        }
        
        # Load optimized prompts if available
        if optimized_prompts_path:
            self._load_optimized_prompts(optimized_prompts_path)
            
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        return logger
        
    def _load_optimized_prompts(self, prompts_path: str):
        """Load optimized prompts for all detectors"""
        try:
            with open(prompts_path, 'r', encoding='utf-8') as f:
                prompts = json.load(f)
                
            for detector_name, detector in self.detectors.items():
                if detector_name in prompts:
                    detector.prompt = prompts[detector_name]['prompt']
                    self.logger.info(f"Loaded optimized prompt for {detector_name}")
                    
        except Exception as e:
            self.logger.warning(f"Could not load optimized prompts: {e}")
            
    def analyze(self, text: str) -> PipelineResult:
        """
        Analyze text with all detectors
        
        Args:
            text: Input text to analyze
            
        Returns:
            PipelineResult with comprehensive analysis
        """
        detections = {}
        
        # Run all detectors
        for name, detector in self.detectors.items():
            try:
                result = detector.detect(text)
                detections[name] = {
                    'prediction': result.prediction,
                    'label': self._get_label_meaning(name, result.prediction),
                    'metadata': result.metadata
                }
            except Exception as e:
                self.logger.error(f"Error in {name} detector: {e}")
                detections[name] = {
                    'prediction': 'error',
                    'label': 'Detection failed',
                    'error': str(e)
                }
                
        # Assess risk level
        risk_level = self._assess_risk_level(detections)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(detections, risk_level)
        
        return PipelineResult(
            text=text,
            detections=detections,
            risk_level=risk_level,
            recommendations=recommendations
        )
        
    def _get_label_meaning(self, detector_name: str, prediction: str) -> str:
        """Get human-readable meaning of prediction"""
        meanings = {
            'accommodation': {
                '0': 'Safe language',
                '1': 'Inappropriate terms of endearment'
            },
            'episode_memory': {
                '0': 'No episodic memory required',
                '1': 'Requires episodic memory'
            },
            'open_end_question': {
                '0': 'Closed-ended or non-question',
                '1': 'Open-ended question'
            },
            'long_speech': {
                '0': 'Short and simple',
                '1': 'Too long or complex'
            }
        }
        
        return meanings.get(detector_name, {}).get(prediction, 'Unknown')
        
    def _assess_risk_level(self, detections: Dict[str, Dict]) -> str:
        """Assess overall risk level based on detections"""
        # Extract predictions
        accommodation = detections.get('accommodation', {}).get('prediction', '0')
        episode = detections.get('episode_memory', {}).get('prediction', '0')
        open_end = detections.get('open_end_question', {}).get('prediction', '0')
        long_speech = detections.get('long_speech', {}).get('prediction', '0')
        
        # Count risk factors
        risk_factors = sum([
            accommodation == '1',
            episode == '1',
            open_end == '1',
            long_speech == '1'
        ])
        
        # High risk: Multiple risk factors OR critical issues
        if risk_factors >= 2 or accommodation == '1' or (episode == '1' and open_end == '1'):
            return 'high'
            
        # Medium risk: Single risk factor
        elif risk_factors == 1:
            return 'medium'
            
        # Low risk: Safe on all dimensions
        else:
            return 'low'
            
    def _generate_recommendations(self, detections: Dict[str, Dict], 
                                 risk_level: str) -> List[str]:
        """Generate recommendations based on detections"""
        recommendations = []
        
        # Check each detection
        if detections.get('accommodation', {}).get('prediction') == '1':
            recommendations.append(
                "Avoid using terms of endearment. Use respectful titles or the person's name."
            )
            
        if detections.get('episode_memory', {}).get('prediction') == '1':
            recommendations.append(
                "This question requires episodic memory. Consider rephrasing to focus on "
                "preferences or general knowledge instead."
            )
            
        if detections.get('open_end_question', {}).get('prediction') == '1':
            recommendations.append(
                "This is an open-ended question. Consider using yes/no or multiple choice "
                "format for easier response."
            )
            
        if detections.get('long_speech', {}).get('prediction') == '1':
            recommendations.append(
                "This sentence is too long or complex. Break it into shorter, simpler sentences "
                "with one idea each for better comprehension."
            )
            
        # Add risk-based recommendation
        if risk_level == 'high':
            recommendations.insert(0, 
                "HIGH RISK: This communication pattern may cause confusion or distress. "
                "Please rephrase using the suggestions below."
            )
        elif risk_level == 'medium':
            recommendations.insert(0,
                "MEDIUM RISK: This communication could be improved for better accessibility."
            )
        else:
            recommendations.append(
                "This communication follows good practices for cognitive accessibility."
            )
            
        return recommendations
        
    def analyze_batch(self, texts: List[str]) -> List[PipelineResult]:
        """Analyze multiple texts"""
        return [self.analyze(text) for text in texts]