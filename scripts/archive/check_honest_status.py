#!/usr/bin/env python3
"""Quick status check for honest evaluation"""

import json
from pathlib import Path
from datetime import datetime

# Find latest directory
latest = max(Path("results").glob("honest_all_detectors_*"), key=lambda p: p.stat().st_mtime)
print(f"Checking: {latest}")
print(f"Time: {datetime.now().strftime('%H:%M:%S')}\n")

# Check each detector
detectors = ["elderspeak_collective", "elderspeak_ToE", "episode_detection_episode_memory", 
             "episode_detection_open_end", "long_speech_detection_long_speech"]

completed = 0
for detector in detectors:
    detector_dir = latest / detector
    results_file = detector_dir / "results.json"
    
    if results_file.exists():
        completed += 1
        with open(results_file, 'r') as f:
            results = json.load(f)
        fold_results = results.get('fold_results', [])
        optimized = sum(1 for fold in fold_results if fold['prompts']['changed'])
        print(f"✓ {detector}: {optimized}/{len(fold_results)} optimized")
    else:
        log_file = detector_dir / "evaluation.log"
        if log_file.exists() and log_file.stat().st_size > 0:
            print(f"⚡ {detector}: RUNNING")
        else:
            print(f"⏳ {detector}: WAITING")

print(f"\nProgress: {completed}/{len(detectors)} complete")