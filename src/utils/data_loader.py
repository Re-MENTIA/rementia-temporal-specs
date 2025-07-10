"""
Data Loader - Unified data loading for all detection tasks
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional


class DataLoader:
    """Handles loading and normalization of datasets"""
    
    def __init__(self, config: Dict):
        """Initialize with configuration"""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def load_terms_of_endearment_data(self, type_key: str = "ToE") -> List[Dict]:
        """Load terms of endearment dataset"""
        task_config = self.config['tasks']['elderspeak']
        type_config = task_config['types'][type_key]
        
        # Build path
        base_path = Path(task_config['base_path'])
        data_path = base_path / type_config['path']
        
        # Load data
        data = self._load_json(data_path)
        
        # Normalize
        return self._normalize_dataset(data, type_config)
        
    def load_collective_instruction_data(self) -> List[Dict]:
        """Load collective instruction dataset"""
        task_config = self.config['tasks']['elderspeak']
        type_config = task_config['types']['collective']
        
        # Build path
        base_path = Path(task_config['base_path'])
        data_path = base_path / type_config['path']
        
        # Load data
        data = self._load_json(data_path)
        
        # Normalize
        return self._normalize_dataset(data, type_config)
        
    def load_episode_memory_data(self) -> List[Dict]:
        """Load episode memory dataset"""
        task_config = self.config['tasks']['episode_detection']
        type_config = task_config['types']['episode_memory']
        
        # Build path
        base_path = Path(task_config['base_path'])
        data_path = base_path / type_config['path']
        
        # Load data
        data = self._load_json(data_path)
        
        # Normalize with label mapping
        return self._normalize_dataset(data, type_config)
        
    def load_open_end_question_data(self) -> List[Dict]:
        """Load open-end question dataset"""
        task_config = self.config['tasks']['question_detection']
        type_config = task_config['types']['open_end']
        
        # Build path
        base_path = Path(task_config['base_path'])
        data_path = base_path / type_config['path']
        
        # Load data
        data = self._load_json(data_path)
        
        # Normalize with label mapping
        return self._normalize_dataset(data, type_config)
        
    def load_long_speech_data(self) -> List[Dict]:
        """Load long speech dataset"""
        task_config = self.config['tasks']['long_speech_detection']
        type_config = task_config['types']['long_speech']
        
        # Build path
        base_path = Path(task_config['base_path'])
        data_path = base_path / type_config['path']
        
        # Load data
        data = self._load_json(data_path)
        
        # Normalize with label mapping
        return self._normalize_dataset(data, type_config)
        
    def _load_json(self, path: Path) -> List[Dict]:
        """Load JSON file"""
        self.logger.info(f"Loading dataset from {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.logger.info(f"Loaded {len(data)} samples")
        return data
        
    def _normalize_dataset(self, data: List[Dict], 
                          type_config: Dict) -> List[Dict]:
        """Normalize dataset to common format"""
        normalized = []
        
        # Get field names and label mapping
        label_field = type_config.get('label_field', 'label')
        input_field = type_config.get('input_field', 'text')
        label_mapping = type_config.get('label_mapping', {})
        
        for item in data:
            # Get text and label
            text = item.get(input_field, item.get('text', item.get('sentence', '')))
            label = str(item.get(label_field, item.get('label', '')))
            
            # Apply label mapping if exists
            if label_mapping:
                label = label_mapping.get(label, label)
                
            normalized.append({
                'text': text,
                'label': label,
                'original': item  # Keep original for reference
            })
            
        # Log label distribution
        label_counts = {}
        for item in normalized:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1
            
        self.logger.info(f"Label distribution: {label_counts}")
        
        return normalized