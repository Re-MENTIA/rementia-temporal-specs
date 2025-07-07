"""
Cross Validator - Handles k-fold cross-validation for detectors
"""

import logging
import random
from collections import defaultdict
from typing import Dict, List, Any

from ..detectors.base import BaseDetector


class CrossValidator:
    """Performs stratified k-fold cross-validation"""
    
    def __init__(self, config: Dict):
        """Initialize with configuration"""
        self.config = config
        self.cv_config = config['cross_validation']
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def run_cross_validation(self, detector: BaseDetector, 
                           dataset: List[Dict]) -> Dict[str, Any]:
        """
        Run k-fold cross-validation
        
        Args:
            detector: Detector instance to evaluate
            dataset: Full dataset to use
            
        Returns:
            Dictionary with cross-validation results
        """
        n_folds = self.cv_config['n_folds']
        n_repeats = self.cv_config['n_repeats']
        
        self.logger.info(f"Starting {n_folds}-fold cross-validation with {n_repeats} repeats")
        
        all_results = []
        
        for repeat in range(n_repeats):
            self.logger.info(f"\n=== REPEAT {repeat + 1}/{n_repeats} ===")
            
            # Create folds
            folds = self._create_stratified_folds(dataset, n_folds)
            
            for fold_idx in range(n_folds):
                self.logger.info(f"Running Fold {fold_idx + 1}")
                
                # Split data
                test_fold = folds[fold_idx]
                train_folds = [f for i, f in enumerate(folds) if i != fold_idx]
                train_data = [item for fold in train_folds for item in fold]
                
                # Evaluate on test fold
                fold_results = detector.evaluate(test_fold)
                
                # Add fold info
                fold_results['fold'] = fold_idx + 1
                fold_results['repeat'] = repeat + 1
                
                all_results.append(fold_results)
                
                self.logger.info(
                    f"Fold {fold_idx + 1} - Accuracy: {fold_results['accuracy']:.2%}"
                )
        
        # Aggregate results
        return self._aggregate_results(all_results, dataset)
        
    def _create_stratified_folds(self, dataset: List[Dict], 
                                n_folds: int) -> List[List[Dict]]:
        """Create stratified k-folds"""
        # Group by label
        label_groups = defaultdict(list)
        for item in dataset:
            label_groups[item['label']].append(item)
            
        # Shuffle each group
        for label_items in label_groups.values():
            random.shuffle(label_items)
            
        # Create folds
        folds = [[] for _ in range(n_folds)]
        
        # Distribute items from each label
        for label, items in label_groups.items():
            for i, item in enumerate(items):
                fold_idx = i % n_folds
                folds[fold_idx].append(item)
                
        # Shuffle each fold
        for fold in folds:
            random.shuffle(fold)
            
        return folds
        
    def _aggregate_results(self, fold_results: List[Dict], 
                          dataset: List[Dict]) -> Dict[str, Any]:
        """Aggregate results across all folds"""
        # Collect all metrics
        metrics_list = defaultdict(list)
        
        for result in fold_results:
            metrics_list['accuracy'].append(result['accuracy'])
            
            # Collect other metrics
            for metric_name, value in result['metrics'].items():
                if isinstance(value, (int, float)):
                    metrics_list[metric_name].append(value)
                    
        # Calculate mean and std for each metric
        aggregate_metrics = {}
        
        for metric_name, values in metrics_list.items():
            if values:
                mean_val = sum(values) / len(values)
                std_val = (sum((x - mean_val) ** 2 for x in values) / len(values)) ** 0.5
                aggregate_metrics[metric_name] = {
                    'mean': mean_val,
                    'std': std_val,
                    'values': values
                }
                
        # Dataset statistics
        label_counts = defaultdict(int)
        for item in dataset:
            label_counts[item['label']] += 1
            
        return {
            'fold_results': fold_results,
            'aggregate_metrics': aggregate_metrics,
            'dataset_stats': {
                'total_samples': len(dataset),
                'label_distribution': dict(label_counts)
            },
            'cv_config': {
                'n_folds': self.cv_config['n_folds'],
                'n_repeats': self.cv_config['n_repeats']
            }
        }