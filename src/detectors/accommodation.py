"""
Accommodation Speech Detector - Detects inappropriate language patterns in eldercare
"""

from .base import BaseDetector, DetectorConfig


class AccommodationSpeechDetector(BaseDetector):
    """Detector for accommodation speech patterns (elderspeak)"""
    
    def _get_detector_config(self) -> DetectorConfig:
        """Get accommodation speech detector configuration"""
        return DetectorConfig(
            name="Accommodation Speech Detector (Terms of Endearment)",
            task_key="accommodation_speech",
            type_key="ToE",
            description="Detects inappropriate terms of endearment in eldercare contexts",
            labels=["0", "1"],  # 0=safe, 1=harmful
        )
        
    def _get_initial_prompt(self) -> str:
        """Get initial prompt from configuration"""
        return self.config['prompts']['ToE']['initial']