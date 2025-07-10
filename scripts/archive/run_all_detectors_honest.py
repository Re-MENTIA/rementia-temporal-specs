#!/usr/bin/env python3
"""
Run ALL detectors with HONEST TextGrad implementation
"""

import os
import sys
import json
import yaml
import subprocess
from pathlib import Path
from datetime import datetime

# Detectors to run
DETECTORS = [
    {"task": "elderspeak", "task_type": "collective", "name": "Collective Instruction"},
    {"task": "elderspeak", "task_type": "ToE", "name": "Terms of Endearment"},
    {"task": "episode_detection", "task_type": "episode_memory", "name": "Episode Memory"},
    {"task": "episode_detection", "task_type": "open_end", "name": "Open-Ended Questions"},
    {"task": "long_speech_detection", "task_type": "long_speech", "name": "Long Speech Detection"}
]

def run_detector(detector_info, output_dir):
    """Run a single detector with honest implementation"""
    print(f"\n{'='*80}")
    print(f"Running: {detector_info['name']}")
    print(f"Task: {detector_info['task']}, Type: {detector_info['task_type']}")
    print('='*80)
    
    # Create output subdirectory
    detector_output = output_dir / f"{detector_info['task']}_{detector_info['task_type']}"
    detector_output.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable,
        "scripts/run_unified_crossval.py",
        "--task", detector_info['task'],
        "--type", detector_info['task_type'],
        "--output", f"honest_{detector_info['task_type']}"
    ]
    
    # Set environment
    env = os.environ.copy()
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    env[key] = value
    
    # Run the detector
    start_time = datetime.now()
    try:
        # Run with output capture
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
            universal_newlines=True
        )
        
        # Save output to log file
        log_file = detector_output / "evaluation.log"
        with open(log_file, 'w') as f:
            for line in process.stdout:
                print(line, end='')  # Print to console
                f.write(line)       # Save to file
        
        process.wait()
        
        if process.returncode == 0:
            print(f"\n✓ {detector_info['name']} completed successfully")
        else:
            print(f"\n✗ {detector_info['name']} failed with code {process.returncode}")
            
    except Exception as e:
        print(f"\n✗ Error running {detector_info['name']}: {e}")
        return False
    
    # Calculate duration
    duration = datetime.now() - start_time
    print(f"Duration: {duration}")
    
    # Look for results
    results_pattern = f"results/*{detector_info['task_type']}*/detailed_results.json"
    results_files = list(Path("results").glob(results_pattern))
    
    if results_files:
        latest_result = max(results_files, key=lambda p: p.stat().st_mtime)
        print(f"Results saved to: {latest_result}")
        
        # Copy to output directory
        import shutil
        shutil.copy2(latest_result, detector_output / "results.json")
        
        # Analyze results
        with open(latest_result, 'r') as f:
            results = json.load(f)
        
        # Count optimizations
        optimized = sum(1 for fold in results.get('fold_results', [])
                       if fold['prompts']['changed'])
        total_folds = len(results.get('fold_results', []))
        
        print(f"\nOptimization Summary:")
        print(f"  REAL optimizations: {optimized}/{total_folds}")
        print(f"  Optimization rate: {optimized/total_folds*100:.1f}%" if total_folds > 0 else "N/A")
        
        # Show performance
        agg = results.get('aggregate_metrics', {})
        if 'accuracy' in agg:
            print(f"  Accuracy: {agg['accuracy']['mean']:.2%} ± {agg['accuracy']['std']:.2%}")
    
    return True

def main():
    """Run all detectors with honest implementation"""
    print("\n" + "="*80)
    print("RUNNING ALL DETECTORS WITH HONEST TEXTGRAD IMPLEMENTATION")
    print("="*80)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/honest_all_detectors_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary tracking
    summary = {
        "start_time": timestamp,
        "detectors": {},
        "total_optimizations": 0,
        "total_folds": 0
    }
    
    # Run each detector
    for detector in DETECTORS:
        success = run_detector(detector, output_dir)
        summary["detectors"][detector['name']] = {
            "success": success,
            "task": detector['task'],
            "task_type": detector['task_type']
        }
    
    # Generate final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY - HONEST IMPLEMENTATION")
    print("="*80)
    
    successful = sum(1 for d in summary["detectors"].values() if d["success"])
    print(f"\nDetectors completed: {successful}/{len(DETECTORS)}")
    
    for name, info in summary["detectors"].items():
        status = "✓" if info["success"] else "✗"
        print(f"  {status} {name}")
    
    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nAll results saved to: {output_dir}")
    print("\nHONEST TEXTGRAD IMPLEMENTATION COMPLETE! 🎯")

if __name__ == "__main__":
    main()