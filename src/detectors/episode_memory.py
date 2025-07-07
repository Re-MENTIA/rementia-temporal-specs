"""
Episode Memory Detector - Identifies questions requiring episodic memory
"""

from .base import BaseDetector, DetectorConfig


class EpisodeMemoryDetector(BaseDetector):
    """Detector for episodic memory requirements in questions"""
    
    def _get_detector_config(self) -> DetectorConfig:
        """Get episode memory detector configuration"""
        task_config = self.config['tasks']['episode_detection']['types']['episode_memory']
        
        return DetectorConfig(
            name="Episode Memory Detector",
            task_key="episode_detection",
            type_key="episode_memory",
            description="Detects questions requiring episodic (autobiographical) memory",
            labels=task_config['labels'],
            label_mapping=task_config.get('label_mapping')
        )
        
    def _get_initial_prompt(self) -> str:
        """Get initial prompt from configuration"""
        return self.config['prompts']['episode_memory']['initial']
        
    def _normalize_response(self, response: str) -> str:
        """Normalize API response with episode memory specific logic"""
        response = response.strip().lower()
        
        # Handle episode memory specific responses
        if response in ['episodic', 'true', '1', 'yes']:
            return '1'
        elif response in ['semantic', 'neutral', 'false', '0', 'no']:
            return '0'
            
        # Fall back to parent normalization
        return super()._normalize_response(response)