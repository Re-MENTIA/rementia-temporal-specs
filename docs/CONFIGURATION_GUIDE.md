# Configuration Guide - Eldercare Detection System

## Overview

The Eldercare Detection System uses a comprehensive modular configuration system that serves as the single source of truth for all components. This guide explains how to use and customize the configuration.

## Configuration Structure

The main configuration file is located at `config/config.yaml` and contains the following sections:

### 1. System Configuration
```yaml
system:
  name: "Eldercare Communication Analysis System"
  version: "2.0.0"
  environment: "production"  # development, testing, production
```

### 2. Detector Registry
All detectors are registered in the configuration with their metadata:

```yaml
detectors:
  terms_of_endearment:
    enabled: true
    class_name: "TermsOfEndearmentDetector"
    module: "src.detectors.terms_of_endearment"
    type_key: "ToE"
    task_category: "elderspeak"
    description: "Detects infantilizing terms of endearment"
```

### 3. Dataset Configuration
Each detector's dataset is configured with paths and label mappings:

```yaml
tasks:
  elderspeak:
    base_path: "datasets/Elderspeak"
    types:
      ToE:
        name: "Terms of Endearment"
        path: "ToE/terms_of_endearment_dataset_refined_v2.json"
        labels: ["0", "1"]
        label_mapping:
          "safe": "0"
          "harmful": "1"
```

### 4. Model Configuration
Configure available models and their parameters:

```yaml
models:
  default: "gpt-4o-mini"
  available:
    gpt-4o-mini:
      provider: "openai"
      max_tokens: 2
      temperature: 0
      cost_per_1k_tokens: 0.00015
```

### 5. TextGrad Optimization
Configure automatic prompt optimization:

```yaml
textgrad_optimization:
  enabled: true
  engine:
    model: "gpt-4o"
    temperature: 0.7
  parameters:
    max_iterations: 5
    learning_rate: 0.1
```

## Using the Configuration System

### 1. Configuration Manager

The `ConfigManager` provides centralized access to all configuration:

```python
from src.utils.config_manager import get_config_manager

# Get configuration manager instance
config_manager = get_config_manager()

# Access configuration values
system_name = config_manager.get('system.name')
default_model = config_manager.get('models.default')

# Get entire sections
detector_configs = config_manager.get_section('detectors')

# Get specific configurations
model_config = config_manager.get_model_config('gpt-4o-mini')
cv_config = config_manager.get_cross_validation_config('standard')
```

### 2. Detector Registry

The `DetectorRegistry` dynamically loads and manages detectors:

```python
from src.detectors import get_detector_registry

# Get registry instance
registry = get_detector_registry()

# List all detectors
all_detectors = registry.list_all_detectors()

# Create a detector instance
detector = registry.get_detector('terms_of_endearment')

# Create multiple detectors
detector_suite = registry.create_detector_suite(['ToE', 'collective'])
```

### 3. Modular Data Loader

The `ModularDataLoader` provides abstraction for different dataset formats:

```python
from src.utils.modular_data_loader import ModularDataLoader

# Create data loader
data_loader = ModularDataLoader()

# Load dataset for a detector
dataset = data_loader.load_dataset('terms_of_endearment')

# Get train/validation split
train_data, val_data = data_loader.get_train_val_split(
    dataset, 
    val_ratio=0.2,
    stratified=True
)
```

## Adding a New Detector

To add a new detector to the system:

### 1. Create the detector class
```python
# src/detectors/my_new_detector.py
from .base import BaseDetector

class MyNewDetector(BaseDetector):
    def __init__(self, **kwargs):
        super().__init__(detector_type='my_new', **kwargs)
    
    def get_prompt(self) -> str:
        return self.config['prompts'].get('my_new', {}).get('initial')
```

### 2. Register in configuration
```yaml
# config/config.yaml
detectors:
  my_new:
    enabled: true
    class_name: "MyNewDetector"
    module: "src.detectors.my_new_detector"
    type_key: "my_new"
    task_category: "my_category"
    description: "Description of what it detects"

tasks:
  my_category:
    base_path: "datasets/my_category"
    types:
      my_new:
        name: "My New Detection Task"
        path: "my_new_dataset.json"
        labels: ["0", "1"]

prompts:
  my_new:
    initial: |
      Your detection prompt here...
```

### 3. Import in __init__.py
```python
# src/detectors/__init__.py
from .my_new_detector import MyNewDetector
```

The detector will be automatically available through the registry!

## Environment-Specific Configuration

The system supports different environments:

```python
# Check environment
if config_manager.is_development():
    # Use development settings
    cv_config = config_manager.get_cross_validation_config('fast')
else:
    # Use production settings
    cv_config = config_manager.get_cross_validation_config('standard')
```

## Configuration Validation

The system automatically validates configuration on load:

```python
# Validate all dataset paths exist
missing_paths = config_manager.validate_paths()
if missing_paths:
    print(f"Missing paths: {missing_paths}")
```

## Best Practices

1. **Never hardcode values** - Always reference the configuration
2. **Use descriptive keys** - Make configuration self-documenting
3. **Provide defaults** - Use the `get()` method with defaults
4. **Validate early** - Check configuration at startup
5. **Document changes** - Update this guide when adding features

## Common Configuration Tasks

### Change the default model
```yaml
models:
  default: "gpt-4o"  # Changed from gpt-4o-mini
```

### Disable a detector
```yaml
detectors:
  long_speech:
    enabled: false  # Detector won't be loaded
```

### Add API provider
```yaml
api:
  anthropic:
    base_url: "https://api.anthropic.com/v1/messages"
    rate_limit_delay: 0.2
```

### Configure fast evaluation
```yaml
cross_validation:
  fast:
    n_folds: 2
    n_repeats: 1
```

## Troubleshooting

### Configuration not found
```python
# Always provide defaults
value = config_manager.get('some.key.path', default='fallback')
```

### Detector not loading
Check:
1. Module path is correct in configuration
2. Class name matches exactly
3. Detector is imported in `__init__.py`
4. `enabled: true` in configuration

### Dataset not found
1. Verify path in configuration
2. Check base_path + type path combination
3. Ensure file exists at expected location

## Advanced Features

### Dynamic configuration updates
```python
# Update configuration at runtime
config_manager.update_config('models.default', 'gpt-4')

# Save changes
config_manager.save_config(backup=True)
```

### Export configuration
```python
# Export as JSON for external tools
config_manager.export_config('config_export.json', format='json')
```

### Custom dataset loaders
```python
# Register custom loader for CSV files
from src.utils.modular_data_loader import DatasetLoader

class CSVDatasetLoader(DatasetLoader):
    def load(self, path):
        # Custom loading logic
        pass

data_loader.register_loader('.csv', CSVDatasetLoader())
```