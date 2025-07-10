"""
Collective Instruction Detector - Detects inappropriate use of collective pronouns in eldercare
"""

from .base import BaseDetector, DetectorConfig


class CollectiveInstructionDetector(BaseDetector):
    """Detector for collective instruction patterns (part of elderspeak)"""
    
    def _get_detector_config(self) -> DetectorConfig:
        """Get collective instruction detector configuration"""
        task_config = self.config['tasks']['elderspeak']['types']['collective']
        return DetectorConfig(
            name="Collective Instruction Detector",
            task_key="elderspeak",
            type_key="collective",
            description="Detects inappropriate use of 'we/us' when giving instructions",
            labels=task_config['labels'],
            label_mapping=task_config.get('label_mapping')
        )
        
    def _get_initial_prompt(self) -> str:
        """Get initial prompt from configuration"""
        return self.config['prompts']['collective']['initial']