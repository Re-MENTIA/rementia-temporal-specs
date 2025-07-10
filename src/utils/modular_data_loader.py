"""
Modular Data Loader - Abstraction layer for different dataset structures
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from abc import ABC, abstractmethod
import yaml


class DatasetSchema:
    """Represents the schema of a dataset"""
    
    def __init__(self, text_field: str = "text", 
                 label_field: str = "label",
                 metadata_fields: Optional[List[str]] = None):
        """
        Initialize dataset schema
        
        Args:
            text_field: Name of the field containing input text
            label_field: Name of the field containing labels
            metadata_fields: Optional list of additional metadata fields
        """
        self.text_field = text_field
        self.label_field = label_field
        self.metadata_fields = metadata_fields or []
        
    def validate(self, data_item: Dict) -> bool:
        """Validate that a data item conforms to the schema"""
        required_fields = [self.text_field, self.label_field]
        return all(field in data_item for field in required_fields)


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders"""
    
    @abstractmethod
    def load(self, path: Path) -> List[Dict]:
        """Load dataset from path"""
        pass
        
    @abstractmethod
    def get_schema(self) -> DatasetSchema:
        """Get the schema for this dataset"""
        pass


class JSONDatasetLoader(DatasetLoader):
    """Loader for JSON datasets"""
    
    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def load(self, path: Path) -> List[Dict]:
        """Load JSON dataset"""
        self.logger.info(f"Loading JSON dataset from {path}")
        
        with open(path, 'r', encoding=self.encoding) as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            raise ValueError(f"Expected list in JSON file, got {type(data)}")
            
        self.logger.info(f"Loaded {len(data)} samples")
        return data
        
    def get_schema(self) -> DatasetSchema:
        """JSON datasets use default schema"""
        return DatasetSchema()


class ModularDataLoader:
    """
    Modular data loader with abstraction layer for different dataset structures
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize with configuration"""
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize dataset loaders
        self._loaders: Dict[str, DatasetLoader] = {
            '.json': JSONDatasetLoader(
                encoding=self.config['dataset'].get('encoding', 'utf-8')
            )
        }
        
        # Cache for loaded datasets
        self._cache: Dict[str, List[Dict]] = {}
        self._cache_enabled = self.config['dataset'].get('cache_enabled', True)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
            
    def register_loader(self, extension: str, loader: DatasetLoader):
        """Register a custom dataset loader for a file extension"""
        self._loaders[extension] = loader
        self.logger.info(f"Registered loader for {extension} files")
        
    def load_dataset(self, detector_key: str, 
                    use_cache: bool = True) -> List[Dict]:
        """
        Load dataset for a specific detector
        
        Args:
            detector_key: Key identifying the detector
            use_cache: Whether to use cached data if available
            
        Returns:
            List of normalized data items
        """
        # Check cache first
        if use_cache and self._cache_enabled and detector_key in self._cache:
            self.logger.info(f"Using cached dataset for {detector_key}")
            return self._cache[detector_key]
            
        # Get detector configuration
        detector_config = self._get_detector_config(detector_key)
        task_category = detector_config['task_category']
        type_key = detector_config['type_key']
        
        # Get task configuration
        task_config = self.config['tasks'][task_category]
        type_config = task_config['types'][type_key]
        
        # Build dataset path
        base_path = Path(task_config['base_path'])
        data_path = base_path / type_config['path']
        
        # Load raw data
        raw_data = self._load_raw_data(data_path)
        
        # Normalize data
        normalized_data = self._normalize_dataset(
            raw_data, type_config, detector_key
        )
        
        # Apply preprocessing if configured
        if self.config['data_loader']['preprocessing']['strip_whitespace']:
            normalized_data = self._preprocess_data(normalized_data)
            
        # Cache the normalized data
        if self._cache_enabled:
            self._cache[detector_key] = normalized_data
            
        return normalized_data
        
    def _get_detector_config(self, detector_key: str) -> Dict:
        """Get detector configuration"""
        detector_configs = self.config.get('detectors', {})
        if detector_key not in detector_configs:
            raise KeyError(f"Detector '{detector_key}' not found in configuration")
        return detector_configs[detector_key]
        
    def _load_raw_data(self, path: Path) -> List[Dict]:
        """Load raw data using appropriate loader"""
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
            
        # Get loader based on file extension
        extension = path.suffix.lower()
        if extension not in self._loaders:
            raise ValueError(f"No loader registered for {extension} files")
            
        loader = self._loaders[extension]
        return loader.load(path)
        
    def _normalize_dataset(self, data: List[Dict], 
                          type_config: Dict,
                          detector_key: str) -> List[Dict]:
        """Normalize dataset to common format"""
        normalized = []
        
        # Get field mappings
        label_field = type_config.get('label_field', 'label')
        input_field = type_config.get('input_field', 'text')
        label_mapping = type_config.get('label_mapping', {})
        
        # Get valid labels
        valid_labels = set(type_config.get('labels', []))
        
        for idx, item in enumerate(data):
            # Extract text and label
            text = self._extract_field(item, input_field, detector_key)
            label = str(self._extract_field(item, label_field, detector_key))
            
            # Apply label mapping
            if label_mapping:
                label = label_mapping.get(label, label)
                
            # Validate label
            if valid_labels and label not in valid_labels:
                self.logger.warning(
                    f"Invalid label '{label}' in {detector_key} dataset at index {idx}"
                )
                continue
                
            # Create normalized item
            normalized_item = {
                'text': text,
                'label': label,
                'index': idx,
                'detector': detector_key,
                'original': item  # Keep original for reference
            }
            
            normalized.append(normalized_item)
            
        # Log statistics
        self._log_dataset_statistics(normalized, detector_key)
        
        return normalized
        
    def _extract_field(self, item: Dict, field_name: str, 
                      detector_key: str) -> Any:
        """Extract field with fallback logic"""
        # Direct field access
        if field_name in item:
            return item[field_name]
            
        # Try common alternatives
        alternatives = {
            'text': ['sentence', 'input', 'content'],
            'label': ['class', 'category', 'target']
        }
        
        for alt in alternatives.get(field_name.split('_')[0], []):
            if alt in item:
                return item[alt]
                
        # Log error and raise
        self.logger.error(
            f"Field '{field_name}' not found in {detector_key} dataset item"
        )
        raise KeyError(f"Required field '{field_name}' not found")
        
    def _preprocess_data(self, data: List[Dict]) -> List[Dict]:
        """Apply preprocessing to normalized data"""
        preprocessing_config = self.config['data_loader']['preprocessing']
        
        processed = []
        for item in data:
            # Copy item
            processed_item = item.copy()
            
            # Strip whitespace
            if preprocessing_config['strip_whitespace']:
                processed_item['text'] = processed_item['text'].strip()
                
            # Check max length
            max_length = preprocessing_config.get('max_length', float('inf'))
            if len(processed_item['text']) > max_length:
                self.logger.warning(
                    f"Text exceeds max length ({max_length}) in {item['detector']}"
                )
                processed_item['text'] = processed_item['text'][:max_length]
                
            # Skip empty texts
            if preprocessing_config['remove_empty'] and not processed_item['text']:
                continue
                
            processed.append(processed_item)
            
        return processed
        
    def _log_dataset_statistics(self, data: List[Dict], detector_key: str):
        """Log dataset statistics"""
        total = len(data)
        label_counts = {}
        
        for item in data:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1
            
        # Calculate statistics
        stats = {
            'total_samples': total,
            'label_distribution': label_counts,
            'avg_text_length': sum(len(item['text']) for item in data) / total if total > 0 else 0
        }
        
        self.logger.info(f"Dataset statistics for {detector_key}: {stats}")
        
    def get_dataset_info(self, detector_key: str) -> Dict:
        """Get information about a dataset"""
        try:
            # Load minimal data to get info
            data = self.load_dataset(detector_key, use_cache=True)
            
            label_counts = {}
            for item in data:
                label = item['label']
                label_counts[label] = label_counts.get(label, 0) + 1
                
            return {
                'detector': detector_key,
                'total_samples': len(data),
                'label_distribution': label_counts,
                'sample_text': data[0]['text'] if data else None,
                'cached': detector_key in self._cache
            }
        except Exception as e:
            self.logger.error(f"Failed to get dataset info for {detector_key}: {e}")
            return {'error': str(e)}
            
    def clear_cache(self, detector_key: Optional[str] = None):
        """Clear dataset cache"""
        if detector_key:
            self._cache.pop(detector_key, None)
            self.logger.info(f"Cleared cache for {detector_key}")
        else:
            self._cache.clear()
            self.logger.info("Cleared all dataset cache")
            
    def get_train_val_split(self, data: List[Dict], 
                           val_ratio: float = 0.2,
                           stratified: bool = True,
                           random_seed: int = 42) -> tuple:
        """
        Split dataset into training and validation sets
        
        Args:
            data: Dataset to split
            val_ratio: Ratio of validation data
            stratified: Whether to use stratified sampling
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, val_data)
        """
        import random
        random.seed(random_seed)
        
        if stratified:
            # Group by label
            label_groups = {}
            for item in data:
                label = item['label']
                if label not in label_groups:
                    label_groups[label] = []
                label_groups[label].append(item)
                
            # Split each group
            train_data = []
            val_data = []
            
            for label, items in label_groups.items():
                random.shuffle(items)
                split_idx = int(len(items) * (1 - val_ratio))
                train_data.extend(items[:split_idx])
                val_data.extend(items[split_idx:])
                
            # Shuffle final sets
            random.shuffle(train_data)
            random.shuffle(val_data)
            
        else:
            # Simple random split
            shuffled = data.copy()
            random.shuffle(shuffled)
            split_idx = int(len(shuffled) * (1 - val_ratio))
            train_data = shuffled[:split_idx]
            val_data = shuffled[split_idx:]
            
        self.logger.info(
            f"Split dataset: {len(train_data)} train, {len(val_data)} validation"
        )
        
        return train_data, val_data