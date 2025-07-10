#!/usr/bin/env python3
"""
最適化された正直な評価（gpt-4.1-mini使用）
レート制限を考慮した効率的な実装
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 親ディレクトリをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from src.models.openai_client import OpenAIClient
from scripts.run_unified_crossval import UnifiedCrossValidator


class OptimizedHonestEvaluator:
    """最適化された正直な評価器"""
    
    DETECTORS = [
        {"task": "elderspeak", "type": "collective", "name": "Collective Instruction"},
        {"task": "elderspeak", "type": "ToE", "name": "Terms of Endearment"},
        {"task": "episode_detection", "type": "episode_memory", "name": "Episode Memory"},
        {"task": "question_detection", "type": "open_end", "name": "Open-Ended Questions"},
        {"task": "long_speech_detection", "type": "long_speech", "name": "Long Speech Detection"}
    ]
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"results/optimized_honest_{self.timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ロギング設定
        self._setup_logging()
        
        # 統計
        self.stats = {
            "total_requests": 0,
            "total_errors": 0,
            "start_time": time.time()
        }
        self.stats_lock = threading.Lock()
        
    def _setup_logging(self):
        """ロギング設定"""
        log_file = self.output_dir / "evaluation.log"
        
        # ハンドラー設定
        handlers = [
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode='w')
        ]
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
        self.logger = logging.getLogger(__name__)
    
    def run_detector_with_optimization(self, detector: Dict) -> Dict:
        """単一検出器を最適化設定で実行"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"実行: {detector['name']}")
        self.logger.info(f"Task: {detector['task']}, Type: {detector['type']}")
        
        start_time = time.time()
        
        try:
            # 評価器を作成（レート制限を削減）
            evaluator = UnifiedCrossValidator(
                task=detector['task'],
                task_type=detector['type']
            )
            
            # レート制限を動的に調整（gpt-4.1-miniは高速）
            evaluator.config['api']['rate_limit_delay'] = 0.05  # 50ms
            
            # データセットをロード
            dataset = evaluator.load_dataset()
            self.logger.info(f"データ数: {len(dataset)}")
            
            # クロスバリデーション実行
            results = evaluator.run_cross_validation(dataset)
            
            # 統計を更新
            with self.stats_lock:
                self.stats["total_requests"] += results.get('total_api_calls', 0)
                self.stats["total_errors"] += len(results.get('errors', {}).get('details', []))
            
            # 結果を保存
            detector_dir = self.output_dir / f"{detector['task']}_{detector['type']}"
            detector_dir.mkdir(exist_ok=True)
            
            with open(detector_dir / "results.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # サマリー
            duration = time.time() - start_time
            fold_results = results.get('fold_results', [])
            optimized = sum(1 for fold in fold_results if fold['prompts']['changed'])
            
            self.logger.info(f"\n✓ {detector['name']} 完了")
            self.logger.info(f"  実行時間: {duration:.1f}秒")
            self.logger.info(f"  最適化: {optimized}/{len(fold_results)} フォールド")
            
            # 精度
            agg = results.get('aggregate_metrics', {})
            if 'accuracy' in agg:
                self.logger.info(f"  精度: {agg['accuracy']['mean']:.2%} ± {agg['accuracy']['std']:.2%}")
            
            return {
                "success": True,
                "duration": duration,
                "optimizations": optimized,
                "folds": len(fold_results),
                "accuracy": agg.get('accuracy', {}).get('mean'),
                "accuracy_std": agg.get('accuracy', {}).get('std')
            }
            
        except Exception as e:
            self.logger.error(f"\n✗ {detector['name']} 失敗: {e}")
            return {
                "success": False,
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    def run_all_parallel(self, max_workers: int = 2):
        """すべての検出器を並列実行"""
        self.logger.info("="*60)
        self.logger.info("最適化された正直なTextGrad評価")
        self.logger.info(f"モデル: gpt-4.1-mini")
        self.logger.info(f"並列実行: 最大{max_workers}ワーカー")
        self.logger.info("="*60)
        
        results = {}
        
        # ThreadPoolExecutorで並列実行
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # タスクを投入
            future_to_detector = {
                executor.submit(self.run_detector_with_optimization, detector): detector
                for detector in self.DETECTORS
            }
            
            # 完了したタスクから結果を取得
            for future in as_completed(future_to_detector):
                detector = future_to_detector[future]
                try:
                    result = future.result()
                    results[detector['name']] = result
                except Exception as e:
                    self.logger.error(f"実行エラー {detector['name']}: {e}")
                    results[detector['name']] = {
                        "success": False,
                        "error": str(e)
                    }
        
        # 最終レポート
        self._generate_final_report(results)
    
    def _generate_final_report(self, results: Dict):
        """最終レポート生成"""
        total_time = time.time() - self.stats["start_time"]
        
        self.logger.info("\n" + "="*60)
        self.logger.info("最終レポート")
        self.logger.info("="*60)
        
        # 成功/失敗カウント
        successful = sum(1 for r in results.values() if r.get('success', False))
        failed = len(results) - successful
        
        self.logger.info(f"\n完了状況:")
        self.logger.info(f"  成功: {successful}/{len(results)}")
        self.logger.info(f"  失敗: {failed}/{len(results)}")
        
        # 最適化統計
        total_optimizations = sum(r.get('optimizations', 0) for r in results.values())
        total_folds = sum(r.get('folds', 0) for r in results.values())
        
        if total_folds > 0:
            self.logger.info(f"\n最適化統計:")
            self.logger.info(f"  総フォールド数: {total_folds}")
            self.logger.info(f"  最適化されたフォールド: {total_optimizations}")
            self.logger.info(f"  最適化率: {total_optimizations/total_folds*100:.1f}%")
        
        # 各検出器の結果
        self.logger.info(f"\n検出器別結果:")
        for name, result in results.items():
            if result.get('success'):
                self.logger.info(f"\n✓ {name}:")
                self.logger.info(f"  実行時間: {result['duration']:.1f}秒")
                self.logger.info(f"  最適化: {result['optimizations']}/{result['folds']}")
                if result.get('accuracy') is not None:
                    self.logger.info(f"  精度: {result['accuracy']:.2%} ± {result.get('accuracy_std', 0):.2%}")
            else:
                self.logger.info(f"\n✗ {name}:")
                self.logger.info(f"  エラー: {result.get('error', 'Unknown')}")
        
        # API統計
        self.logger.info(f"\nAPI統計:")
        self.logger.info(f"  総リクエスト数: {self.stats['total_requests']}")
        self.logger.info(f"  エラー数: {self.stats['total_errors']}")
        self.logger.info(f"  総実行時間: {total_time:.1f}秒")
        if self.stats['total_requests'] > 0:
            self.logger.info(f"  リクエスト/秒: {self.stats['total_requests']/total_time:.1f}")
        
        # サマリーを保存
        summary = {
            "timestamp": self.timestamp,
            "total_time": total_time,
            "stats": self.stats,
            "results": results
        }
        
        with open(self.output_dir / "summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"\n結果保存先: {self.output_dir}")
        
        if failed == 0:
            self.logger.info("\n🎯 すべての検出器が正常に完了しました！")
        else:
            self.logger.warning(f"\n⚠️  {failed} 個の検出器が失敗しました")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='最適化された正直な評価')
    parser.add_argument('--workers', type=int, default=2, 
                      help='並列ワーカー数（デフォルト: 2）')
    args = parser.parse_args()
    
    evaluator = OptimizedHonestEvaluator()
    evaluator.run_all_parallel(max_workers=args.workers)


if __name__ == "__main__":
    main()