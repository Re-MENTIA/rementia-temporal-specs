#!/usr/bin/env python3
"""
美しく、正直なTextGrad評価スクリプト
エラーを隠さず、結果を偽らない
"""

import os
import sys
import json
import yaml
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/honest_evaluation.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


class HonestTextGradEvaluator:
    """正直で美しいTextGrad評価クラス"""
    
    DETECTORS = [
        {"task": "elderspeak", "type": "collective", "name": "Collective Instruction"},
        {"task": "elderspeak", "type": "ToE", "name": "Terms of Endearment"},
        {"task": "episode_detection", "type": "episode_memory", "name": "Episode Memory"},
        {"task": "question_detection", "type": "open_end", "name": "Open-Ended Questions"},  # 修正！
        {"task": "long_speech_detection", "type": "long_speech", "name": "Long Speech Detection"}
    ]
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"results/honest_textgrad_{self.timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # APIキーを環境変数に設定
        self._setup_environment()
        
        # 結果トラッキング
        self.results = {
            "timestamp": self.timestamp,
            "detectors": {},
            "errors": [],
            "summary": {
                "total": len(self.DETECTORS),
                "completed": 0,
                "failed": 0,
                "with_optimization": 0,
                "total_folds": 0,
                "optimized_folds": 0
            }
        }
    
    def _setup_environment(self):
        """環境変数を設定"""
        if os.path.exists('.env'):
            with open('.env', 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
    
    def run_detector(self, detector: Dict) -> Tuple[bool, Optional[Dict]]:
        """単一の検出器を実行"""
        logger.info(f"\n{'='*60}")
        logger.info(f"実行中: {detector['name']}")
        logger.info(f"Task: {detector['task']}, Type: {detector['type']}")
        
        detector_dir = self.output_dir / f"{detector['task']}_{detector['type']}"
        detector_dir.mkdir(parents=True, exist_ok=True)
        
        # コマンド構築
        cmd = [
            sys.executable,
            "scripts/run_unified_crossval.py",
            "--task", detector['task'],
            "--type", detector['type'],
            "--output", f"honest_{detector['type']}_{self.timestamp}"
        ]
        
        try:
            # プロセス実行
            start_time = datetime.now()
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30分タイムアウト
            )
            
            duration = datetime.now() - start_time
            
            # 出力を保存
            with open(detector_dir / "stdout.log", 'w') as f:
                f.write(process.stdout)
            with open(detector_dir / "stderr.log", 'w') as f:
                f.write(process.stderr)
            
            if process.returncode != 0:
                error_msg = f"プロセスがコード {process.returncode} で終了"
                logger.error(f"✗ {detector['name']}: {error_msg}")
                self.results["errors"].append({
                    "detector": detector['name'],
                    "error": error_msg,
                    "stderr": process.stderr[:500]  # 最初の500文字
                })
                return False, None
            
            # 結果ファイルを探す
            results_pattern = f"results/*honest_{detector['type']}_{self.timestamp}*/detailed_results.json"
            results_files = list(Path("results").glob(results_pattern))
            
            if not results_files:
                error_msg = "結果ファイルが見つかりません"
                logger.error(f"✗ {detector['name']}: {error_msg}")
                self.results["errors"].append({
                    "detector": detector['name'],
                    "error": error_msg
                })
                return False, None
            
            # 最新の結果を読み込む
            latest_result = max(results_files, key=lambda p: p.stat().st_mtime)
            with open(latest_result, 'r') as f:
                results = json.load(f)
            
            # 結果をコピー
            import shutil
            shutil.copy2(latest_result, detector_dir / "results.json")
            
            # 統計を計算
            fold_results = results.get('fold_results', [])
            optimized = sum(1 for fold in fold_results if fold['prompts']['changed'])
            
            logger.info(f"✓ {detector['name']} 完了")
            logger.info(f"  実行時間: {duration}")
            logger.info(f"  最適化: {optimized}/{len(fold_results)} フォールド")
            
            # 精度を表示
            agg = results.get('aggregate_metrics', {})
            if 'accuracy' in agg:
                logger.info(f"  精度: {agg['accuracy']['mean']:.2%} ± {agg['accuracy']['std']:.2%}")
            
            return True, {
                "duration": str(duration),
                "folds": len(fold_results),
                "optimized_folds": optimized,
                "accuracy": agg.get('accuracy', {}).get('mean', None),
                "accuracy_std": agg.get('accuracy', {}).get('std', None)
            }
            
        except subprocess.TimeoutExpired:
            error_msg = "タイムアウト (30分)"
            logger.error(f"✗ {detector['name']}: {error_msg}")
            self.results["errors"].append({
                "detector": detector['name'],
                "error": error_msg
            })
            return False, None
        except Exception as e:
            error_msg = str(e)
            logger.error(f"✗ {detector['name']}: エラー - {error_msg}")
            self.results["errors"].append({
                "detector": detector['name'],
                "error": error_msg
            })
            return False, None
    
    def run_all(self):
        """すべての検出器を実行"""
        logger.info("="*60)
        logger.info("正直なTextGrad評価を開始")
        logger.info(f"出力ディレクトリ: {self.output_dir}")
        logger.info("="*60)
        
        for detector in self.DETECTORS:
            success, stats = self.run_detector(detector)
            
            self.results["detectors"][detector['name']] = {
                "success": success,
                "task": detector['task'],
                "type": detector['type'],
                "stats": stats
            }
            
            if success:
                self.results["summary"]["completed"] += 1
                if stats:
                    self.results["summary"]["total_folds"] += stats["folds"]
                    self.results["summary"]["optimized_folds"] += stats["optimized_folds"]
                    if stats["optimized_folds"] > 0:
                        self.results["summary"]["with_optimization"] += 1
            else:
                self.results["summary"]["failed"] += 1
        
        self._generate_final_report()
    
    def _generate_final_report(self):
        """最終レポートを生成"""
        logger.info("\n" + "="*60)
        logger.info("最終レポート - 正直なTextGrad実装")
        logger.info("="*60)
        
        summary = self.results["summary"]
        
        # 完了状況
        logger.info(f"\n完了状況:")
        logger.info(f"  成功: {summary['completed']}/{summary['total']}")
        logger.info(f"  失敗: {summary['failed']}/{summary['total']}")
        
        # 最適化統計
        if summary['total_folds'] > 0:
            opt_rate = summary['optimized_folds'] / summary['total_folds'] * 100
            logger.info(f"\n最適化統計:")
            logger.info(f"  総フォールド数: {summary['total_folds']}")
            logger.info(f"  最適化されたフォールド: {summary['optimized_folds']}")
            logger.info(f"  最適化率: {opt_rate:.1f}%")
            logger.info(f"  最適化を含む検出器: {summary['with_optimization']}/{summary['completed']}")
        
        # 各検出器の結果
        logger.info(f"\n検出器別結果:")
        for name, info in self.results["detectors"].items():
            status = "✓" if info["success"] else "✗"
            logger.info(f"\n{status} {name}:")
            if info["success"] and info["stats"]:
                stats = info["stats"]
                logger.info(f"  最適化: {stats['optimized_folds']}/{stats['folds']} フォールド")
                if stats["accuracy"] is not None:
                    logger.info(f"  精度: {stats['accuracy']:.2%} ± {stats['accuracy_std']:.2%}")
            else:
                logger.info(f"  失敗")
        
        # エラーレポート
        if self.results["errors"]:
            logger.error(f"\nエラー詳細:")
            for error in self.results["errors"]:
                logger.error(f"\n{error['detector']}:")
                logger.error(f"  {error['error']}")
                if 'stderr' in error and error['stderr']:
                    logger.error(f"  標準エラー: {error['stderr'][:200]}...")
        
        # 結果を保存
        results_file = self.output_dir / "evaluation_summary.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n結果保存先: {results_file}")
        
        # 成功/失敗の判定
        if summary['failed'] == 0:
            logger.info("\n🎯 すべての検出器が正常に完了しました！")
        else:
            logger.warning(f"\n⚠️  {summary['failed']} 個の検出器が失敗しました")


def main():
    """メイン関数"""
    evaluator = HonestTextGradEvaluator()
    evaluator.run_all()


if __name__ == "__main__":
    main()