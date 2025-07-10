#!/usr/bin/env python3
"""
高速・正直なTextGrad評価
gpt-4.1-miniの高レート制限を活用した並列実行
"""

import asyncio
import aiohttp
import json
import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import concurrent.futures
from collections import defaultdict

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 親ディレクトリをパスに追加
sys.path.append(str(Path(__file__).parent.parent))


class FastHonestEvaluator:
    """高速・正直な評価器"""
    
    DETECTORS = [
        {"task": "elderspeak", "type": "collective", "name": "Collective Instruction"},
        {"task": "elderspeak", "type": "ToE", "name": "Terms of Endearment"},
        {"task": "episode_detection", "type": "episode_memory", "name": "Episode Memory"},
        {"task": "question_detection", "type": "open_end", "name": "Open-Ended Questions"},
        {"task": "long_speech_detection", "type": "long_speech", "name": "Long Speech Detection"}
    ]
    
    def __init__(self):
        self.api_key = self._get_api_key()
        self.model = "gpt-4.1-mini"  # 高速モデル！
        self.max_concurrent = 50  # 並列数を増やす（10,000 RPMなので余裕）
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        self.request_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
    def _get_api_key(self) -> str:
        """APIキーを取得"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key and os.path.exists('.env'):
            with open('.env', 'r') as f:
                for line in f:
                    if line.startswith('OPENAI_API_KEY='):
                        api_key = line.split('=', 1)[1].strip()
                        break
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        return api_key
    
    async def call_api(self, session: aiohttp.ClientSession, 
                      messages: List[Dict], max_tokens: int = 2) -> Optional[str]:
        """非同期API呼び出し（リトライ付き）"""
        async with self.semaphore:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": max_tokens
            }
            
            for retry in range(3):
                try:
                    async with session.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        self.request_count += 1
                        
                        if response.status == 200:
                            result = await response.json()
                            return result['choices'][0]['message']['content'].strip()
                        else:
                            error_text = await response.text()
                            logger.warning(f"API error {response.status}: {error_text[:100]}")
                            if response.status == 429:  # Rate limit
                                await asyncio.sleep(2 ** retry)
                            
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout (retry {retry + 1}/3)")
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.warning(f"Error (retry {retry + 1}/3): {e}")
                    await asyncio.sleep(1)
            
            self.error_count += 1
            return None
    
    async def evaluate_batch_async(self, session: aiohttp.ClientSession,
                                  samples: List[Dict], prompt: str) -> List[Dict]:
        """バッチ評価（非同期）"""
        tasks = []
        for sample in samples:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": sample['input']}
            ]
            task = self.call_api(session, messages)
            tasks.append(task)
        
        # 並列実行
        responses = await asyncio.gather(*tasks)
        
        # 結果をまとめる
        results = []
        for sample, response in zip(samples, responses):
            if response:
                results.append({
                    'input': sample['input'],
                    'true': sample['label'],
                    'pred': response,
                    'correct': response == sample['label']
                })
            else:
                results.append({
                    'input': sample['input'],
                    'true': sample['label'],
                    'pred': "0",
                    'correct': False,
                    'error': True
                })
        
        return results
    
    async def optimize_prompt_async(self, session: aiohttp.ClientSession,
                                   initial_prompt: str, errors: List[Dict]) -> Optional[str]:
        """プロンプト最適化（非同期）"""
        if not errors:
            return None
        
        # エラー分析プロンプト
        gradient_prompt = f"""You are an expert prompt engineer.

Current prompt:
{initial_prompt}

Misclassifications ({len(errors)} errors):
"""
        for i, err in enumerate(errors[:5]):
            gradient_prompt += f"\n{i+1}. Input: \"{err['input']}\"\n   Expected: {err['true']}, Got: {err['pred']}\n"
        
        gradient_prompt += """
Analyze these errors and provide an improved prompt that addresses the issues.
Provide ONLY the complete improved prompt, nothing else."""
        
        messages = [
            {"role": "system", "content": "You are an expert at improving classification prompts."},
            {"role": "user", "content": gradient_prompt}
        ]
        
        response = await self.call_api(session, messages, max_tokens=4000)
        return response if response and len(response) > 100 else None
    
    async def run_cross_validation_async(self, task: str, task_type: str, 
                                       dataset: List[Dict]) -> Dict:
        """非同期クロスバリデーション"""
        results = {
            "task": task,
            "type": task_type,
            "folds": [],
            "optimizations": 0,
            "total_time": 0
        }
        
        start_time = time.time()
        
        # 簡易3分割クロスバリデーション
        fold_size = len(dataset) // 3
        
        async with aiohttp.ClientSession() as session:
            for fold in range(3):
                logger.info(f"  Fold {fold + 1}/3")
                
                # データ分割
                test_start = fold * fold_size
                test_end = (fold + 1) * fold_size if fold < 2 else len(dataset)
                test_data = dataset[test_start:test_end]
                train_data = dataset[:test_start] + dataset[test_end:]
                
                # バリデーション用に訓練データを分割
                val_size = len(train_data) // 5
                val_data = train_data[:val_size]
                
                # 初期プロンプト評価
                initial_prompt = self._get_initial_prompt(task, task_type)
                val_results = await self.evaluate_batch_async(session, val_data, initial_prompt)
                initial_acc = sum(1 for r in val_results if r['correct']) / len(val_results)
                
                # 最適化
                optimized_prompt = initial_prompt
                prompt_changed = False
                
                if initial_acc < 0.95:  # 最適化が必要
                    errors = [r for r in val_results if not r['correct']]
                    for iteration in range(3):
                        logger.info(f"    Optimization iteration {iteration + 1}/3")
                        new_prompt = await self.optimize_prompt_async(session, optimized_prompt, errors)
                        
                        if new_prompt:
                            # 新プロンプトを評価
                            new_results = await self.evaluate_batch_async(session, val_data, new_prompt)
                            new_acc = sum(1 for r in new_results if r['correct']) / len(new_results)
                            
                            if new_acc > initial_acc:
                                optimized_prompt = new_prompt
                                prompt_changed = True
                                logger.info(f"    ✓ Improved: {initial_acc:.2%} → {new_acc:.2%}")
                                break
                            else:
                                logger.info(f"    ✗ No improvement: {new_acc:.2%}")
                
                # テストセット評価
                test_results = await self.evaluate_batch_async(session, test_data, optimized_prompt)
                test_acc = sum(1 for r in test_results if r['correct']) / len(test_results)
                
                results["folds"].append({
                    "fold": fold + 1,
                    "test_accuracy": test_acc,
                    "prompt_changed": prompt_changed
                })
                
                if prompt_changed:
                    results["optimizations"] += 1
        
        results["total_time"] = time.time() - start_time
        results["avg_accuracy"] = sum(f["test_accuracy"] for f in results["folds"]) / len(results["folds"])
        
        return results
    
    def _get_initial_prompt(self, task: str, task_type: str) -> str:
        """初期プロンプトを取得"""
        # 簡易版 - 実際はconfig.yamlから読む
        prompts = {
            ("elderspeak", "collective"): "Analyze if the text contains collective instructions using 'we/us'.",
            ("elderspeak", "ToE"): "Analyze if the text contains terms of endearment.",
            ("episode_detection", "episode_memory"): "Analyze if the text relates to episodic memory.",
            ("question_detection", "open_end"): "Analyze if the text is an open-ended question.",
            ("long_speech_detection", "long_speech"): "Analyze if the text is too long for the context."
        }
        
        prompt = prompts.get((task, task_type), "Classify the text.")
        return prompt + "\n\nRespond with only '0' or '1'."
    
    def load_dataset(self, task: str, task_type: str) -> List[Dict]:
        """データセットをロード"""
        # データセットパスのマッピング
        dataset_paths = {
            ("elderspeak", "collective"): "datasets/Elderspeak/collective/collective_instruction_dataset.json",
            ("elderspeak", "ToE"): "datasets/Elderspeak/ToE/terms_of_endearment_dataset_refined_v2.json",
            ("episode_detection", "episode_memory"): "datasets/episode-open-end-Q/episode-memory/episode_memory_dataset.json",
            ("question_detection", "open_end"): "datasets/episode-open-end-Q/open-end-Q/open_end_Q_dataset.json",
            ("long_speech_detection", "long_speech"): "datasets/long-speech/long_speech_dataset.json"
        }
        
        path = dataset_paths.get((task, task_type))
        if not path:
            raise ValueError(f"Unknown task/type: {task}/{task_type}")
        
        with open(path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # データ形式を統一
        dataset = []
        for item in raw_data:
            # ラベルマッピング
            if 'instruction_type' in item:
                label = '1' if item['instruction_type'] == 'collective' else '0'
            elif 'label' in item:
                label = str(item['label'])
            elif 'is_too_long?' in item:
                label = '1' if item['is_too_long?'] == 'True' else '0'
            elif 'question_type' in item:
                label = '1' if item['question_type'] == 'open-ended' else '0'
            elif 'is_epsodic_memory_related' in item:
                label = '1' if item['is_epsodic_memory_related'] == 'true' else '0'
            else:
                label = '0'
            
            dataset.append({
                'input': item.get('sentence', item.get('text', '')),
                'label': label
            })
        
        return dataset
    
    async def run_all_async(self):
        """すべての検出器を並列実行"""
        logger.info("="*60)
        logger.info("高速・正直なTextGrad評価開始")
        logger.info(f"モデル: {self.model}")
        logger.info(f"最大並列数: {self.max_concurrent}")
        logger.info("="*60)
        
        all_results = {}
        
        # 各検出器を非同期で実行
        tasks = []
        for detector in self.DETECTORS:
            logger.info(f"\n開始: {detector['name']}")
            try:
                dataset = self.load_dataset(detector['task'], detector['type'])
                logger.info(f"  データ数: {len(dataset)}")
                
                # 非同期タスクを作成
                task = self.run_cross_validation_async(
                    detector['task'], 
                    detector['type'], 
                    dataset
                )
                tasks.append((detector['name'], task))
                
            except Exception as e:
                logger.error(f"  エラー: {e}")
                all_results[detector['name']] = {"error": str(e)}
        
        # すべてのタスクを並列実行
        for name, task in tasks:
            try:
                result = await task
                all_results[name] = result
                logger.info(f"\n完了: {name}")
                logger.info(f"  平均精度: {result['avg_accuracy']:.2%}")
                logger.info(f"  最適化: {result['optimizations']}/3 フォールド")
                logger.info(f"  実行時間: {result['total_time']:.1f}秒")
            except Exception as e:
                logger.error(f"\n失敗: {name} - {e}")
                all_results[name] = {"error": str(e)}
        
        # 最終統計
        total_time = time.time() - self.start_time
        logger.info("\n" + "="*60)
        logger.info("最終統計")
        logger.info("="*60)
        logger.info(f"総リクエスト数: {self.request_count}")
        logger.info(f"エラー数: {self.error_count}")
        logger.info(f"総実行時間: {total_time:.1f}秒")
        logger.info(f"リクエスト/秒: {self.request_count/total_time:.1f}")
        
        # 結果を保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results/fast_honest_{timestamp}.json"
        os.makedirs("results", exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n結果保存: {output_file}")
        logger.info("\n🎯 高速・正直な評価完了！")


async def main():
    """メイン関数"""
    evaluator = FastHonestEvaluator()
    await evaluator.run_all_async()


if __name__ == "__main__":
    asyncio.run(main())