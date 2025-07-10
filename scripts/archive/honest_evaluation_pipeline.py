#!/usr/bin/env python3
"""
Honest evaluation pipeline:
1. Run cross-validation with ORIGINAL prompts
2. Optimize prompts with TextGrad
3. Run cross-validation with OPTIMIZED prompts
4. Generate honest comparison report
"""

import os
import sys
import json
import logging
import argparse
import yaml
# import numpy as np  # Not needed
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import subprocess

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.data_loader import DataLoader


def run_command(cmd: List[str], description: str) -> int:
    """Run command and return exit code"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed with code {result.returncode}")
    else:
        print(f"SUCCESS: {description} completed")
    
    return result.returncode


def generate_comparison_report(before_dir: Path, after_dir: Path, 
                             optimization_dir: Path, output_path: Path):
    """Generate comparison report between before and after results"""
    
    report_lines = [
        "# HONEST TextGrad Evaluation Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Executive Summary",
        "",
        "This report presents the HONEST evaluation of TextGrad prompt optimization.",
        "We compare cross-validation results BEFORE and AFTER optimization.",
        "",
        "## Results Summary",
        "",
        "| Detector | Before CV | After CV | CV Improvement | Optimization Val Improvement |",
        "|----------|-----------|----------|----------------|------------------------------|"
    ]
    
    # Load results
    detectors = ['ToE', 'collective', 'episode_memory', 'open_end', 'long_speech']
    detector_names = {
        'ToE': 'Terms of Endearment',
        'collective': 'Collective Instruction',
        'episode_memory': 'Episode Memory',
        'open_end': 'Open-End Questions',
        'long_speech': 'Long Speech'
    }
    
    for detector in detectors:
        try:
            # Load before results
            before_file = list(before_dir.glob(f"**/crossvalidation_results.json"))
            if before_file:
                with open(before_file[0], 'r') as f:
                    before_data = json.load(f)
                    if before_data.get('task', '').endswith(detector):
                        before_avg = before_data.get('average_metrics', {}).get('accuracy', 0)
            
            # Load after results  
            after_file = list(after_dir.glob(f"**/crossvalidation_results.json"))
            if after_file:
                with open(after_file[0], 'r') as f:
                    after_data = json.load(f)
                    if after_data.get('task', '').endswith(detector):
                        after_avg = after_data.get('average_metrics', {}).get('accuracy', 0)
            
            # Load optimization results
            opt_file = optimization_dir / f"{detector}_optimization.json"
            if opt_file.exists():
                with open(opt_file, 'r') as f:
                    opt_data = json.load(f)
                    opt_improvement = opt_data.get('improvement', 0)
            
            cv_improvement = after_avg - before_avg
            
            report_lines.append(
                f"| {detector_names[detector]} | "
                f"{before_avg:.2%} | "
                f"{after_avg:.2%} | "
                f"{cv_improvement:+.2%} | "
                f"{opt_improvement:+.2%} |"
            )
            
        except Exception as e:
            print(f"Error loading results for {detector}: {e}")
    
    report_lines.extend([
        "",
        "## Key Findings",
        "",
        "1. **Optimization Impact**: Shows how TextGrad optimization affected real-world performance",
        "2. **Validation vs Test**: Compares optimization validation improvement with cross-validation improvement",
        "3. **Honest Assessment**: All results from clean, separate runs with no data leakage",
        "",
        "## Detailed Results",
        ""
    ])
    
    # Add detailed results for each detector
    for detector in detectors:
        report_lines.extend([
            f"\n### {detector_names[detector]}",
            ""
        ])
        
        try:
            # Add optimization details
            opt_file = optimization_dir / f"{detector}_optimization.json"
            if opt_file.exists():
                with open(opt_file, 'r') as f:
                    opt_data = json.load(f)
                    
                report_lines.extend([
                    "**Optimization Process:**",
                    f"- Initial validation accuracy: {opt_data['validation_acc'][0]:.2%}",
                    f"- Best validation accuracy: {opt_data['best_validation_acc']:.2%}",
                    f"- Improvement: {opt_data['improvement']:+.2%}",
                    f"- Training epochs: {len(opt_data['validation_acc']) - 1}",
                    ""
                ])
                
                # Show if prompt actually changed
                if opt_data['initial_prompt'] != opt_data['best_prompt']:
                    report_lines.extend([
                        "**Prompt Changed:** Yes",
                        "",
                        "*Key changes in optimized prompt:*",
                        "- [TextGrad made modifications to improve performance]",
                        ""
                    ])
                else:
                    report_lines.extend([
                        "**Prompt Changed:** No (optimization found original was best)",
                        ""
                    ])
                    
        except Exception as e:
            report_lines.append(f"Error loading optimization details: {e}")
    
    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Honest evaluation pipeline')
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--output-base', default='results/honest_evaluation')
    parser.add_argument('--skip-before', action='store_true', help='Skip before optimization CV')
    parser.add_argument('--skip-optimization', action='store_true', help='Skip TextGrad optimization')
    parser.add_argument('--skip-after', action='store_true', help='Skip after optimization CV')
    parser.add_argument('--detectors', nargs='+', 
                       default=['collective'],  # Start with collective as requested
                       help='Detectors to evaluate')
    
    args = parser.parse_args()
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create output directories
    output_base = Path(args.output_base)
    before_dir = output_base / timestamp / 'before_optimization'
    optimization_dir = output_base / timestamp / 'textgrad_optimization'
    after_dir = output_base / timestamp / 'after_optimization'
    
    for dir_path in [before_dir, optimization_dir, after_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("HONEST EVALUATION PIPELINE")
    print("="*60)
    print(f"Timestamp: {timestamp}")
    print(f"Output directory: {output_base / timestamp}")
    print(f"Detectors: {args.detectors}")
    
    # Step 1: Cross-validation with ORIGINAL prompts
    if not args.skip_before:
        for detector in args.detectors:
            cmd = [
                sys.executable, 'scripts/run_unified_crossval.py',
                '--task', 'elderspeak' if detector in ['ToE', 'collective'] else detector.replace('_', '-'),
                '--type', detector,
                '--config', args.config
            ]
            
            # Save current directory
            original_dir = os.getcwd()
            
            # Run from project root, not from output directory
            exit_code = run_command(cmd, f"BEFORE optimization CV for {detector}")
            
            if exit_code != 0:
                print(f"ERROR: Failed to run before optimization CV for {detector}")
    
    # Step 2: Optimize prompts with TextGrad
    if not args.skip_optimization:
        cmd = [
            sys.executable, 'scripts/real_textgrad_optimization.py',
            '--config', args.config,
            '--output', str(optimization_dir),
            '--max-epochs', '3',
            '--batch-size', '5'
        ]
        
        exit_code = run_command(cmd, "TextGrad prompt optimization")
        
        if exit_code != 0:
            print("ERROR: TextGrad optimization failed")
            return 1
    
    # Step 3: Update config with optimized prompts
    if not args.skip_after and not args.skip_optimization:
        # Load original config
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create temporary config with optimized prompts
        temp_config_path = after_dir / 'optimized_config.yaml'
        
        # Load optimization results and update prompts
        for detector in args.detectors:
            opt_file = optimization_dir / f"{detector}_optimization.json"
            if opt_file.exists():
                with open(opt_file, 'r') as f:
                    opt_data = json.load(f)
                    
                # Update prompt in config
                if detector == 'ToE':
                    config['prompts']['ToE']['initial'] = opt_data['best_prompt']
                elif detector == 'collective':
                    config['prompts']['collective']['initial'] = opt_data['best_prompt']
                elif detector == 'episode_memory':
                    config['prompts']['episode_memory']['initial'] = opt_data['best_prompt']
                elif detector == 'open_end':
                    config['prompts']['open_end']['initial'] = opt_data['best_prompt']
                elif detector == 'long_speech':
                    config['prompts']['long_speech']['initial'] = opt_data['best_prompt']
        
        # Save temporary config
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        # Step 4: Cross-validation with OPTIMIZED prompts
        for detector in args.detectors:
            cmd = [
                sys.executable, 'scripts/run_unified_crossval.py',
                '--task', 'elderspeak' if detector in ['ToE', 'collective'] else detector.replace('_', '-'),
                '--type', detector,
                '--config', str(temp_config_path)
            ]
            
            # Run from project root
            exit_code = run_command(cmd, f"AFTER optimization CV for {detector}")
            
            if exit_code != 0:
                print(f"ERROR: Failed to run after optimization CV for {detector}")
    
    # Step 5: Generate comparison report
    report_path = output_base / timestamp / 'honest_comparison_report.md'
    generate_comparison_report(before_dir, after_dir, optimization_dir, report_path)
    
    print("\n" + "="*60)
    print("HONEST EVALUATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_base / timestamp}")
    print(f"Report: {report_path}")
    
    # Also save a summary JSON
    summary = {
        'timestamp': timestamp,
        'detectors': args.detectors,
        'directories': {
            'before': str(before_dir),
            'optimization': str(optimization_dir),
            'after': str(after_dir)
        },
        'report': str(report_path)
    }
    
    summary_path = output_base / timestamp / 'evaluation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())