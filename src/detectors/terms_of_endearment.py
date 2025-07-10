"""
Terms of Endearment Detector - Detects inappropriate terms of endearment in eldercare
"""

from .base import BaseDetector, DetectorConfig


class TermsOfEndearmentDetector(BaseDetector):
    """Detector for inappropriate terms of endearment (part of elderspeak)"""
    
    def _get_detector_config(self) -> DetectorConfig:
        """Get terms of endearment detector configuration"""
        return DetectorConfig(
            name="Terms of Endearment Detector",
            task_key="elderspeak",
            type_key="ToE",
            description="Detects inappropriate terms of endearment in eldercare contexts",
            labels=["0", "1"],  # 0=safe, 1=harmful
        )
        
    def _get_initial_prompt(self) -> str:
        """Get initial prompt from configuration"""
        return self.config['prompts']['ToE']['initial']