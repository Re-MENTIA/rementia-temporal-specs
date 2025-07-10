#!/usr/bin/env python3
"""
単一の検出器を正直に実行（デバッグ用）
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

def run_single_detector(task, task_type):
    """単一の検出器を実行"""
    print(f"\n{'='*60}")
    print(f"実行: {task}/{task_type}")
    print(f"時刻: {datetime.now()}")
    print('='*60)
    
    # 環境変数設定
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    
    # コマンド構築
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cmd = [
        sys.executable,
        "scripts/run_unified_crossval.py",
        "--task", task,
        "--type", task_type,
        "--output", f"single_honest_{task_type}_{timestamp}"
    ]
    
    print(f"コマンド: {' '.join(cmd)}")
    
    # 実行
    try:
        process = subprocess.run(
            cmd,
            capture_output=False,  # 直接出力を表示
            text=True
        )
        
        if process.returncode == 0:
            print(f"\n✓ 成功")
        else:
            print(f"\n✗ 失敗（コード: {process.returncode}）")
            
        return process.returncode
        
    except Exception as e:
        print(f"\n✗ エラー: {e}")
        return -1


def main():
    parser = argparse.ArgumentParser(description='単一検出器の実行')
    parser.add_argument('--task', required=True, help='タスク名')
    parser.add_argument('--type', required=True, help='タスクタイプ')
    args = parser.parse_args()
    
    return run_single_detector(args.task, args.type)


if __name__ == "__main__":
    sys.exit(main())