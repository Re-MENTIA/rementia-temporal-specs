"""
Demo Script - Demonstrates the modular configuration system
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config_manager import get_config_manager
from src.utils.modular_data_loader import ModularDataLoader
from src.detectors import get_detector_registry


def main():
    """Demonstrate the modular configuration system"""
    
    print("=" * 60)
    print("Eldercare Detection System - Modular Configuration Demo")
    print("=" * 60)
    
    # 1. Configuration Manager Demo
    print("\n1. Configuration Manager Demo")
    print("-" * 30)
    
    config_manager = get_config_manager()
    
    # Get system information
    print(f"System Name: {config_manager.get('system.name')}")
    print(f"Version: {config_manager.get('system.version')}")
    print(f"Environment: {config_manager.get_environment()}")
    
    # Get enabled detectors
    enabled_detectors = config_manager.get_enabled_detectors()
    print(f"\nEnabled Detectors: {', '.join(enabled_detectors)}")
    
    # Get model configuration
    default_model = config_manager.get('models.default')
    model_config = config_manager.get_model_config(default_model)
    print(f"\nDefault Model: {default_model}")
    print(f"Model Config: {model_config}")
    
    # 2. Detector Registry Demo
    print("\n\n2. Detector Registry Demo")
    print("-" * 30)
    
    registry = get_detector_registry()
    
    # List all detectors
    print("\nRegistered Detectors:")
    for detector_info in registry.list_all_detectors():
        print(f"  - {detector_info['key']}: {detector_info['description']}")
        print(f"    Category: {detector_info['category']}")
        print(f"    Enabled: {detector_info['enabled']}")
        
    # Get detectors by category
    elderspeak_detectors = registry.get_detectors_by_category('elderspeak')
    print(f"\nElderspeak Detectors: {', '.join(elderspeak_detectors)}")
    
    # 3. Modular Data Loader Demo
    print("\n\n3. Modular Data Loader Demo")
    print("-" * 30)
    
    data_loader = ModularDataLoader()
    
    # Load dataset for a detector
    detector_key = 'terms_of_endearment'
    print(f"\nLoading dataset for '{detector_key}'...")
    
    try:
        dataset_info = data_loader.get_dataset_info(detector_key)
        print(f"Total Samples: {dataset_info['total_samples']}")
        print(f"Label Distribution: {dataset_info['label_distribution']}")
        print(f"Cached: {dataset_info['cached']}")
        
        if dataset_info.get('sample_text'):
            print(f"Sample Text: {dataset_info['sample_text'][:100]}...")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        
    # 4. Creating Detector Instances
    print("\n\n4. Creating Detector Instances")
    print("-" * 30)
    
    # Create a single detector
    try:
        toe_detector = registry.get_detector('terms_of_endearment')
        detector_info = toe_detector.get_detector_info()
        print(f"\nCreated Detector: {detector_info['name']}")
        print(f"Type: {detector_info['type']}")
        print(f"Model: {detector_info['model']}")
    except Exception as e:
        print(f"Error creating detector: {e}")
        
    # Create detector suite
    print("\n\nCreating detector suite...")
    detector_suite = registry.create_detector_suite(['terms_of_endearment', 'use_pronoun'])
    print(f"Created {len(detector_suite)} detectors")
    
    # 5. Configuration Access Patterns
    print("\n\n5. Configuration Access Patterns")
    print("-" * 30)
    
    # Get cross-validation config
    cv_config = config_manager.get_cross_validation_config('standard')
    print(f"\nCross-Validation Config:")
    print(f"  Folds: {cv_config.get('n_folds')}")
    print(f"  Repeats: {cv_config.get('n_repeats')}")
    print(f"  Stratified: {cv_config.get('stratified')}")
    
    # Get TextGrad config
    textgrad_config = config_manager.get_textgrad_config()
    print(f"\nTextGrad Optimization:")
    print(f"  Enabled: {textgrad_config.get('enabled')}")
    print(f"  Engine Model: {textgrad_config.get('engine', {}).get('model')}")
    print(f"  Max Iterations: {textgrad_config.get('parameters', {}).get('max_iterations')}")
    
    # Get output config
    output_config = config_manager.get_output_config()
    print(f"\nOutput Configuration:")
    print(f"  Results Directory: {output_config.get('results_dir')}")
    print(f"  Report Formats: {output_config.get('reports', {}).get('formats')}")
    
    # 6. Validate Configuration
    print("\n\n6. Configuration Validation")
    print("-" * 30)
    
    missing_paths = config_manager.validate_paths()
    if missing_paths:
        print(f"\nMissing dataset paths:")
        for path in missing_paths:
            print(f"  - {path}")
    else:
        print("\nAll configured paths exist!")
        
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()