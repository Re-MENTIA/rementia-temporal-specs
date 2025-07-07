"""
Open-End Question Detector - Identifies open-ended vs closed-ended questions
"""

from .base import BaseDetector, DetectorConfig


class OpenEndQuestionDetector(BaseDetector):
    """Detector for open-ended questions"""
    
    def _get_detector_config(self) -> DetectorConfig:
        """Get open-end question detector configuration"""
        task_config = self.config['tasks']['question_detection']['types']['open_end']
        
        return DetectorConfig(
            name="Open-End Question Detector",
            task_key="question_detection", 
            type_key="open_end",
            description="Detects open-ended questions requiring elaborate responses",
            labels=task_config['labels'],
            label_mapping=task_config.get('label_mapping')
        )
        
    def _get_initial_prompt(self) -> str:
        """Get initial prompt from configuration"""
        return self.config['prompts']['open_end']['initial']
        
    def _normalize_response(self, response: str) -> str:
        """Normalize API response with question-specific logic"""
        response = response.strip().lower()
        
        # Handle question type specific responses
        if response in ['open-ended', 'open', '1', 'yes']:
            return '1'
        elif response in ['closed-ended', 'closed', 'neutral', '0', 'no']:
            return '0'
            
        # Fall back to parent normalization
        return super()._normalize_response(response)