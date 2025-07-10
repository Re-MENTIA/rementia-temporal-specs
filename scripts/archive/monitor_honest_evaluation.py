#!/usr/bin/env python3
"""
Monitor progress of honest evaluation for all detectors
"""

import json
import time
from pathlib import Path
from datetime import datetime

def monitor_progress():
    """Monitor evaluation progress"""
    # Find latest honest evaluation directory
    results_dirs = list(Path("results").glob("honest_all_detectors_*"))
    if not results_dirs:
        print("No evaluation running")
        return
    
    latest_dir = max(results_dirs, key=lambda p: p.stat().st_mtime)
    print(f"Monitoring: {latest_dir}")
    
    while True:
        print(f"\n{'='*80}")
        print(f"HONEST EVALUATION PROGRESS - {datetime.now().strftime('%H:%M:%S')}")
        print('='*80)
        
        # Check each detector
        for detector_dir in sorted(latest_dir.iterdir()):
            if detector_dir.is_dir():
                print(f"\n{detector_dir.name}:")
                
                # Check for results
                results_file = detector_dir / "results.json"
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    
                    # Count optimizations
                    fold_results = results.get('fold_results', [])
                    optimized = sum(1 for fold in fold_results if fold['prompts']['changed'])
                    
                    print(f"  Status: COMPLETE ✓")
                    print(f"  REAL optimizations: {optimized}/{len(fold_results)}")
                    
                    # Show accuracy
                    agg = results.get('aggregate_metrics', {})
                    if 'accuracy' in agg:
                        print(f"  Accuracy: {agg['accuracy']['mean']:.2%} ± {agg['accuracy']['std']:.2%}")
                else:
                    # Check log for progress
                    log_file = detector_dir / "evaluation.log"
                    if log_file.exists() and log_file.stat().st_size > 0:
                        # Get last few lines
                        with open(log_file, 'r') as f:
                            lines = f.readlines()
                            
                        # Look for optimization status
                        optimizations = [l for l in lines if "Optimization complete:" in l]
                        improvements = [l for l in lines if "✓ Improvement found!" in l]
                        current_fold = len([l for l in lines if "Running Fold" in l])
                        
                        print(f"  Status: RUNNING")
                        print(f"  Current fold: {current_fold}")
                        print(f"  Optimizations found: {len(improvements)}")
                        
                        # Show last status
                        for line in lines[-5:]:
                            if any(key in line for key in ["Fold", "Optimization", "✓", "✗", "accuracy"]):
                                print(f"    {line.strip()}")
                    else:
                        print(f"  Status: WAITING")
        
        # Check if all complete
        all_complete = all(
            (latest_dir / d / "results.json").exists() 
            for d in latest_dir.iterdir() 
            if d.is_dir()
        )
        
        if all_complete:
            print("\n✓ ALL DETECTORS COMPLETE!")
            break
        
        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    monitor_progress()