"""
Long Speech Detector - Identifies overly long or complex sentences
"""

from .base import BaseDetector, DetectorConfig


class LongSpeechDetector(BaseDetector):
    """Detector for long/complex sentences in eldercare communication"""
    
    def _get_detector_config(self) -> DetectorConfig:
        """Get long speech detector configuration"""
        task_config = self.config['tasks']['long_speech_detection']['types']['long_speech']
        
        return DetectorConfig(
            name="Long Speech Detector",
            task_key="long_speech_detection",
            type_key="long_speech",
            description="Detects sentences that are too long or complex for cognitive accessibility",
            labels=task_config['labels'],
            label_mapping=task_config.get('label_mapping')
        )
        
    def _get_initial_prompt(self) -> str:
        """Get initial prompt from configuration"""
        return self.config['prompts']['long_speech']['initial']
        
    def _normalize_response(self, response: str) -> str:
        """Normalize API response with long speech specific logic"""
        response = response.strip().lower()
        
        # Handle long speech specific responses
        if response in ['long', 'complex', 'true', '1', 'yes']:
            return '1'
        elif response in ['short', 'simple', 'false', '0', 'no']:
            return '0'
            
        # Fall back to parent normalization
        return super()._normalize_response(response)