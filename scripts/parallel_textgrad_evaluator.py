#!/usr/bin/env python3
"""
並列実行対応のTextGrad評価器
レート制限を考慮した効率的な実装
"""

import asyncio
import aiohttp
import json
import os
import time
from typing import List, Dict, Tuple
from datetime import datetime
import logging

class ParallelTextGradEvaluator:
    """並列実行対応の評価器"""
    
    def __init__(self, api_key: str, max_concurrent: int = 10):
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.request_times = []
        self.rpm_limit = 10000  # gpt-4o-miniのRPM制限
        
    async def rate_limit_check(self):
        """レート制限チェック（1分間に10000リクエストまで）"""
        now = time.time()
        # 1分以内のリクエストをカウント
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        if len(self.request_times) >= self.rpm_limit - 100:  # 安全マージン
            # 待機時間を計算
            wait_time = 60 - (now - self.request_times[0])
            logging.info(f"レート制限に近づいています。{wait_time:.1f}秒待機")
            await asyncio.sleep(wait_time)
            
        self.request_times.append(now)
    
    async def call_api(self, session: aiohttp.ClientSession, messages: List[Dict]) -> str:
        """非同期API呼び出し"""
        async with self.semaphore:
            await self.rate_limit_check()
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-4o-mini",  # または gpt-4.1-mini
                "messages": messages,
                "temperature": 0,
                "max_tokens": 2
            }
            
            try:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['choices'][0]['message']['content'].strip()
                    else:
                        error_text = await response.text()
                        logging.error(f"API error {response.status}: {error_text}")
                        return None
                        
            except asyncio.TimeoutError:
                logging.error("API timeout")
                return None
            except Exception as e:
                logging.error(f"API error: {e}")
                return None
    
    async def evaluate_batch(self, session: aiohttp.ClientSession, 
                           samples: List[Dict], prompt: str) -> List[Dict]:
        """バッチ評価（並列実行）"""
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
                # エラーの場合は予測を"0"とする
                results.append({
                    'input': sample['input'],
                    'true': sample['label'],
                    'pred': "0",
                    'correct': False,
                    'error': True
                })
        
        return results
    
    async def run_evaluation(self, dataset: List[Dict], prompt: str, 
                           batch_size: int = 20) -> Tuple[float, List[Dict]]:
        """データセット全体の評価"""
        all_results = []
        
        async with aiohttp.ClientSession() as session:
            # バッチ処理
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i+batch_size]
                logging.info(f"Processing batch {i//batch_size + 1}/{(len(dataset)-1)//batch_size + 1}")
                
                batch_results = await self.evaluate_batch(session, batch, prompt)
                all_results.extend(batch_results)
                
                # バッチ間の短い待機（必要に応じて）
                await asyncio.sleep(0.1)
        
        # 精度計算
        correct = sum(1 for r in all_results if r.get('correct', False))
        accuracy = correct / len(all_results) if all_results else 0
        
        return accuracy, all_results


# 使用例
async def main():
    api_key = os.getenv('OPENAI_API_KEY')
    evaluator = ParallelTextGradEvaluator(api_key, max_concurrent=20)
    
    # テストデータ
    dataset = [
        {"input": "一緒に頑張りましょう", "label": "1"},
        {"input": "お薬を飲んでください", "label": "0"},
        # ... 他のデータ
    ]
    
    prompt = "Analyze the text and classify..."
    
    start_time = time.time()
    accuracy, results = await evaluator.run_evaluation(dataset, prompt)
    end_time = time.time()
    
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Time: {end_time - start_time:.1f}秒")
    print(f"Requests/sec: {len(dataset)/(end_time - start_time):.1f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())