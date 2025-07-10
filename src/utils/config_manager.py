"""
Configuration Manager - Centralized configuration access and validation
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
from datetime import datetime


class ConfigManager:
    """
    Centralized configuration manager for the eldercare detection system.
    Provides validated access to all configuration parameters.
    """
    
    _instance = None
    _config = None
    
    def __new__(cls, config_path: str = "config/config.yaml"):
        """Singleton pattern to ensure single configuration instance"""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialize(config_path)
        return cls._instance
        
    def _initialize(self, config_path: str):
        """Initialize configuration manager"""
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._load_config()
        self._validate_config()
        
    def _load_config(self):
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
            self.logger.info(f"Loaded configuration from {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse configuration file: {e}")
            
    def _validate_config(self):
        """Validate configuration structure and required fields"""
        required_sections = [
            'system', 'dataset', 'detectors', 'tasks', 'models', 
            'api', 'cross_validation', 'prompts', 'output', 'logging'
        ]
        
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required configuration section: {section}")
                
        # Validate detector configurations
        for detector_key, detector_config in self._config['detectors'].items():
            required_fields = ['class_name', 'module', 'type_key', 'task_category']
            for field in required_fields:
                if field not in detector_config:
                    raise ValueError(
                        f"Missing required field '{field}' in detector '{detector_key}'"
                    )
                    
        self.logger.info("Configuration validation passed")
        
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to configuration value (e.g., 'models.default')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
        
    def get_section(self, section: str) -> Dict:
        """Get entire configuration section"""
        if section not in self._config:
            raise KeyError(f"Configuration section '{section}' not found")
        return self._config[section].copy()
        
    def get_detector_config(self, detector_key: str) -> Dict:
        """Get configuration for specific detector"""
        detectors = self.get_section('detectors')
        if detector_key not in detectors:
            raise KeyError(f"Detector '{detector_key}' not found in configuration")
        return detectors[detector_key].copy()
        
    def get_task_config(self, task_category: str, type_key: str) -> Dict:
        """Get configuration for specific task type"""
        tasks = self.get_section('tasks')
        if task_category not in tasks:
            raise KeyError(f"Task category '{task_category}' not found")
            
        task = tasks[task_category]
        if type_key not in task.get('types', {}):
            raise KeyError(f"Type '{type_key}' not found in task '{task_category}'")
            
        return task['types'][type_key].copy()
        
    def get_model_config(self, model_name: Optional[str] = None) -> Dict:
        """Get model configuration"""
        models = self.get_section('models')
        
        if model_name is None:
            model_name = models.get('default')
            
        available_models = models.get('available', {})
        if model_name not in available_models:
            raise KeyError(f"Model '{model_name}' not found in available models")
            
        return available_models[model_name].copy()
        
    def get_prompt(self, detector_key: str) -> str:
        """Get prompt for specific detector"""
        prompts = self.get_section('prompts')
        if detector_key not in prompts:
            # Try to use base template
            base_template = prompts.get('base_template')
            if base_template:
                self.logger.warning(
                    f"No specific prompt for '{detector_key}', using base template"
                )
                return base_template
            raise KeyError(f"Prompt for detector '{detector_key}' not found")
            
        prompt_config = prompts[detector_key]
        if isinstance(prompt_config, dict):
            return prompt_config.get('initial', '')
        return prompt_config
        
    def get_cross_validation_config(self, mode: str = 'standard') -> Dict:
        """Get cross-validation configuration"""
        cv_config = self.get_section('cross_validation')
        
        if mode == 'fast':
            return cv_config.get('fast', cv_config.get('standard', {}))
        elif mode == 'standard':
            return cv_config.get('standard', {})
        else:
            raise ValueError(f"Unknown cross-validation mode: {mode}")
            
    def get_output_config(self) -> Dict:
        """Get output configuration"""
        return self.get_section('output')
        
    def get_logging_config(self) -> Dict:
        """Get logging configuration"""
        return self.get_section('logging')
        
    def get_enabled_detectors(self) -> List[str]:
        """Get list of enabled detector keys"""
        detectors = self.get_section('detectors')
        return [
            key for key, config in detectors.items()
            if config.get('enabled', True)
        ]
        
    def get_api_config(self, provider: str = 'openai') -> Dict:
        """Get API configuration for specific provider"""
        api_config = self.get_section('api')
        if provider not in api_config:
            raise KeyError(f"API provider '{provider}' not found in configuration")
        return api_config[provider].copy()
        
    def get_textgrad_config(self) -> Dict:
        """Get TextGrad optimization configuration"""
        return self.get_section('textgrad_optimization')
        
    def get_streamlit_config(self) -> Dict:
        """Get Streamlit UI configuration"""
        return self.get_section('streamlit')
        
    def get_metrics_config(self) -> Dict:
        """Get evaluation metrics configuration"""
        return self.get_section('metrics')
        
    def save_config(self, backup: bool = True):
        """Save current configuration back to file"""
        if backup:
            # Create backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.config_path.with_suffix(f".{timestamp}.bak")
            self.config_path.rename(backup_path)
            self.logger.info(f"Created configuration backup: {backup_path}")
            
        # Save configuration
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
        self.logger.info(f"Saved configuration to {self.config_path}")
        
    def update_config(self, key_path: str, value: Any):
        """
        Update configuration value
        
        Args:
            key_path: Dot-separated path to configuration value
            value: New value to set
        """
        keys = key_path.split('.')
        config = self._config
        
        # Navigate to parent
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
            
        # Set value
        config[keys[-1]] = value
        self.logger.info(f"Updated configuration: {key_path} = {value}")
        
    def export_config(self, output_path: str, format: str = 'yaml'):
        """Export configuration to file"""
        output_file = Path(output_path)
        
        if format == 'yaml':
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
        elif format == 'json':
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
        self.logger.info(f"Exported configuration to {output_file}")
        
    def get_environment(self) -> str:
        """Get current environment setting"""
        return self.get('system.environment', 'production')
        
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.get_environment() == 'development'
        
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.get_environment() == 'production'
        
    def validate_paths(self) -> List[str]:
        """Validate that all configured paths exist"""
        missing_paths = []
        
        # Check dataset paths
        for task_name, task_config in self.get_section('tasks').items():
            base_path = Path(task_config.get('base_path', ''))
            for type_name, type_config in task_config.get('types', {}).items():
                full_path = base_path / type_config.get('path', '')
                if not full_path.exists():
                    missing_paths.append(str(full_path))
                    
        return missing_paths
        
    def get_all_prompts(self) -> Dict[str, str]:
        """Get all configured prompts"""
        prompts = self.get_section('prompts').copy()
        # Remove base template from the list
        prompts.pop('base_template', None)
        return prompts
        
    def reload_config(self):
        """Reload configuration from file"""
        self._load_config()
        self._validate_config()
        self.logger.info("Configuration reloaded")


# Convenience function to get config manager instance
def get_config_manager(config_path: str = "config/config.yaml") -> ConfigManager:
    """Get the singleton ConfigManager instance"""
    return ConfigManager(config_path)