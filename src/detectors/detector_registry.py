"""
Detector Registry - Dynamic detector registration and management system
"""

import importlib
import logging
from typing import Dict, Type, Optional, List
import yaml
from pathlib import Path

from .base import BaseDetector


class DetectorRegistry:
    """
    Central registry for all detectors in the system.
    Provides dynamic loading and management of detector modules.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the detector registry"""
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._detectors: Dict[str, Type[BaseDetector]] = {}
        self._detector_configs: Dict[str, Dict] = {}
        
        # Auto-register all configured detectors
        self._register_all_detectors()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
            
    def _register_all_detectors(self):
        """Register all detectors defined in configuration"""
        detector_configs = self.config.get('detectors', {})
        
        for detector_key, detector_config in detector_configs.items():
            if detector_config.get('enabled', True):
                try:
                    self.register_detector(detector_key, detector_config)
                    self.logger.info(f"Registered detector: {detector_key}")
                except Exception as e:
                    self.logger.error(f"Failed to register detector {detector_key}: {e}")
                    
    def register_detector(self, key: str, config: Dict):
        """
        Register a single detector
        
        Args:
            key: Unique identifier for the detector
            config: Detector configuration including module and class name
        """
        module_name = config.get('module')
        class_name = config.get('class_name')
        
        if not module_name or not class_name:
            raise ValueError(f"Missing module or class_name in detector config: {key}")
            
        try:
            # Dynamically import the module
            module = importlib.import_module(module_name)
            
            # Get the detector class
            detector_class = getattr(module, class_name)
            
            # Verify it's a subclass of BaseDetector
            if not issubclass(detector_class, BaseDetector):
                raise TypeError(f"{class_name} is not a subclass of BaseDetector")
                
            # Store the detector class and config
            self._detectors[key] = detector_class
            self._detector_configs[key] = config
            
        except ImportError as e:
            raise ImportError(f"Failed to import module {module_name}: {e}")
        except AttributeError as e:
            raise AttributeError(f"Class {class_name} not found in module {module_name}: {e}")
            
    def get_detector(self, key: str, **kwargs) -> BaseDetector:
        """
        Get an instance of a registered detector
        
        Args:
            key: Detector identifier
            **kwargs: Additional arguments to pass to detector constructor
            
        Returns:
            Instantiated detector object
        """
        if key not in self._detectors:
            raise KeyError(f"Detector '{key}' not found in registry")
            
        detector_class = self._detectors[key]
        detector_config = self._detector_configs[key]
        
        # Merge configuration with kwargs
        init_params = {
            'config': self.config,
            'detector_config': detector_config,
            **kwargs
        }
        
        return detector_class(**init_params)
        
    def get_all_detector_keys(self) -> List[str]:
        """Get list of all registered detector keys"""
        return list(self._detectors.keys())
        
    def get_enabled_detector_keys(self) -> List[str]:
        """Get list of enabled detector keys"""
        return [
            key for key, config in self._detector_configs.items()
            if config.get('enabled', True)
        ]
        
    def get_detector_config(self, key: str) -> Dict:
        """Get configuration for a specific detector"""
        if key not in self._detector_configs:
            raise KeyError(f"Detector '{key}' not found in registry")
        return self._detector_configs[key].copy()
        
    def get_detectors_by_category(self, category: str) -> List[str]:
        """Get all detectors belonging to a specific task category"""
        return [
            key for key, config in self._detector_configs.items()
            if config.get('task_category') == category
        ]
        
    def create_detector_suite(self, detector_keys: Optional[List[str]] = None, 
                            **kwargs) -> Dict[str, BaseDetector]:
        """
        Create a suite of detector instances
        
        Args:
            detector_keys: List of detector keys to instantiate. 
                         If None, all enabled detectors are created.
            **kwargs: Additional arguments for detector initialization
            
        Returns:
            Dictionary mapping detector keys to instances
        """
        if detector_keys is None:
            detector_keys = self.get_enabled_detector_keys()
            
        suite = {}
        for key in detector_keys:
            try:
                suite[key] = self.get_detector(key, **kwargs)
                self.logger.info(f"Created detector instance: {key}")
            except Exception as e:
                self.logger.error(f"Failed to create detector {key}: {e}")
                
        return suite
        
    def get_detector_info(self, key: str) -> Dict:
        """Get detailed information about a detector"""
        if key not in self._detector_configs:
            raise KeyError(f"Detector '{key}' not found in registry")
            
        config = self._detector_configs[key]
        return {
            'key': key,
            'name': config.get('class_name'),
            'module': config.get('module'),
            'description': config.get('description', 'No description available'),
            'category': config.get('task_category'),
            'type_key': config.get('type_key'),
            'enabled': config.get('enabled', True)
        }
        
    def list_all_detectors(self) -> List[Dict]:
        """Get information about all registered detectors"""
        return [self.get_detector_info(key) for key in self._detectors.keys()]


# Singleton instance
_registry_instance = None


def get_detector_registry(config_path: str = "config/config.yaml") -> DetectorRegistry:
    """
    Get the singleton detector registry instance
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        DetectorRegistry instance
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = DetectorRegistry(config_path)
    return _registry_instance