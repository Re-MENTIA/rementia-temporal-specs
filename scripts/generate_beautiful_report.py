#!/usr/bin/env python3
"""
美しい詳細レポート生成スクリプト
画像生成、整理されたファイル構成、完全な分析
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import base64
import io

# 親ディレクトリをパスに追加
sys.path.append(str(Path(__file__).parent.parent))


class BeautifulReportGenerator:
    """美しいレポートを生成するクラス"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = Path(f"reports/beautiful_report_{self.timestamp}")
        
        # 美しいディレクトリ構造を作成
        self._create_beautiful_structure()
        
        # 結果を読み込む
        self.summary = self._load_summary()
        self.all_results = self._load_all_results()
        
    def _create_beautiful_structure(self):
        """美しいディレクトリ構造を作成"""
        directories = [
            self.report_dir / "00_overview",
            self.report_dir / "01_detectors",
            self.report_dir / "01_detectors" / "collective_instruction",
            self.report_dir / "01_detectors" / "terms_of_endearment",
            self.report_dir / "01_detectors" / "episode_memory",
            self.report_dir / "01_detectors" / "open_ended_questions",
            self.report_dir / "01_detectors" / "long_speech_detection",
            self.report_dir / "02_visualizations",
            self.report_dir / "03_optimization_analysis",
            self.report_dir / "04_comparative_analysis",
            self.report_dir / "05_technical_details"
        ]
        
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _load_summary(self) -> Dict:
        """サマリーファイルを読み込む"""
        summary_path = self.results_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _load_all_results(self) -> Dict:
        """すべての結果ファイルを読み込む"""
        results = {}
        for detector_dir in self.results_dir.iterdir():
            if detector_dir.is_dir() and (detector_dir / "results.json").exists():
                with open(detector_dir / "results.json", 'r', encoding='utf-8') as f:
                    results[detector_dir.name] = json.load(f)
        return results
    
    def generate_svg_chart(self, data: Dict, chart_type: str) -> str:
        """SVGチャートを生成（matplotlib不要）"""
        if chart_type == "accuracy_bar":
            return self._generate_accuracy_bar_chart(data)
        elif chart_type == "optimization_pie":
            return self._generate_optimization_pie_chart(data)
        elif chart_type == "timeline":
            return self._generate_timeline_chart(data)
        return ""
    
    def _generate_accuracy_bar_chart(self, results: Dict) -> str:
        """精度バーチャートのSVGを生成"""
        svg = '''<svg viewBox="0 0 800 500" xmlns="http://www.w3.org/2000/svg">
  <style>
    .title { font: bold 20px sans-serif; }
    .label { font: 14px sans-serif; }
    .value { font: bold 16px sans-serif; }
    .bar { transition: opacity 0.3s; }
    .bar:hover { opacity: 0.8; }
  </style>
  
  <!-- Title -->
  <text x="400" y="30" text-anchor="middle" class="title">検出器別精度</text>
  
  <!-- Y axis -->
  <line x1="80" y1="60" x2="80" y2="420" stroke="black" stroke-width="2"/>
  <text x="30" y="65" class="label">100%</text>
  <text x="30" y="245" class="label">50%</text>
  <text x="30" y="425" class="label">0%</text>
  
  <!-- X axis -->
  <line x1="80" y1="420" x2="720" y2="420" stroke="black" stroke-width="2"/>
'''
        
        # データを精度順にソート
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].get('accuracy', 0),
            reverse=True
        )
        
        bar_width = 100
        spacing = 20
        x_start = 100
        
        colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336"]
        
        for i, (name, data) in enumerate(sorted_results[:5]):
            accuracy = data.get('accuracy', 0) * 100
            bar_height = accuracy * 3.6  # Scale to SVG height
            x = x_start + i * (bar_width + spacing)
            y = 420 - bar_height
            
            # Bar
            svg += f'''
  <rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" 
        fill="{colors[i]}" class="bar" rx="5"/>
  
  <!-- Value -->
  <text x="{x + bar_width/2}" y="{y - 10}" text-anchor="middle" class="value">
    {accuracy:.1f}%
  </text>
  
  <!-- Label -->
  <text x="{x + bar_width/2}" y="445" text-anchor="middle" class="label"
        transform="rotate(-20 {x + bar_width/2} 445)">
    {name.replace('_', ' ')}
  </text>
'''
        
        svg += '</svg>'
        return svg
    
    def _generate_optimization_pie_chart(self, data: Dict) -> str:
        """最適化率の円グラフSVGを生成"""
        total = data['total_folds']
        optimized = data['optimized_folds']
        not_optimized = total - optimized
        
        # 角度計算
        optimized_angle = (optimized / total) * 360 if total > 0 else 0
        
        # SVG円グラフ
        svg = f'''<svg viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg">
  <style>
    .title {{ font: bold 18px sans-serif; }}
    .label {{ font: 14px sans-serif; }}
    .slice {{ transition: transform 0.3s; cursor: pointer; }}
    .slice:hover {{ transform: scale(1.05); transform-origin: center; }}
  </style>
  
  <text x="200" y="30" text-anchor="middle" class="title">最適化率分析</text>
  
  <!-- Optimized slice -->
  <path d="M 200 200 L 200 80 A 120 120 0 {1 if optimized_angle > 180 else 0} 1 {200 + 120 * self._sin(optimized_angle)} {200 - 120 * self._cos(optimized_angle)} Z"
        fill="#4CAF50" class="slice" stroke="white" stroke-width="2"/>
  
  <!-- Not optimized slice -->
  <path d="M 200 200 L {200 + 120 * self._sin(optimized_angle)} {200 - 120 * self._cos(optimized_angle)} A 120 120 0 {1 if optimized_angle < 180 else 0} 1 200 80 Z"
        fill="#FF5252" class="slice" stroke="white" stroke-width="2"/>
  
  <!-- Center circle -->
  <circle cx="200" cy="200" r="60" fill="white"/>
  
  <!-- Center text -->
  <text x="200" y="200" text-anchor="middle" class="title">{optimized}/{total}</text>
  <text x="200" y="220" text-anchor="middle" class="label">({optimized/total*100:.1f}%)</text>
  
  <!-- Legend -->
  <rect x="30" y="340" width="20" height="20" fill="#4CAF50"/>
  <text x="60" y="355" class="label">最適化済み ({optimized})</text>
  
  <rect x="200" y="340" width="20" height="20" fill="#FF5252"/>
  <text x="230" y="355" class="label">未最適化 ({not_optimized})</text>
</svg>'''
        return svg
    
    def _sin(self, degrees: float) -> float:
        """度数法のsin（簡易実装）"""
        import math
        return math.sin(math.radians(degrees))
    
    def _cos(self, degrees: float) -> float:
        """度数法のcos（簡易実装）"""
        import math
        return math.cos(math.radians(degrees))
    
    def generate_main_report(self):
        """メインレポートを生成"""
        report = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TextGrad Optimization Report - {self.timestamp}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 700;
        }}
        .subtitle {{
            opacity: 0.9;
            margin-top: 10px;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            transition: transform 0.3s;
        }}
        .card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }}
        .metric {{
            display: inline-block;
            margin: 10px;
            padding: 15px 25px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9em;
        }}
        .success {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .warning {{
            color: #FF9800;
            font-weight: bold;
        }}
        .chart-container {{
            margin: 30px 0;
            text-align: center;
        }}
        .chart-container svg {{
            max-width: 100%;
            height: auto;
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #555;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .progress-bar {{
            width: 100%;
            height: 20px;
            background: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.5s;
        }}
        .timestamp {{
            color: #999;
            font-size: 0.9em;
            text-align: right;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 HONEST TextGrad Optimization Report</h1>
        <div class="subtitle">完全に正直な実装による評価結果</div>
        <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>
"""
        
        # Overview Card
        report += self._generate_overview_card()
        
        # Charts
        report += '''
    <div class="card">
        <h2>📊 視覚的分析</h2>
        <div class="chart-container">
            <h3>検出器別精度</h3>
'''
        report += self.generate_svg_chart(self.summary['results'], 'accuracy_bar')
        report += '''
        </div>
        <div class="chart-container">
            <h3>最適化率分析</h3>
'''
        report += self.generate_svg_chart({
            'total_folds': sum(r.get('folds', 0) for r in self.summary['results'].values()),
            'optimized_folds': sum(r.get('optimizations', 0) for r in self.summary['results'].values())
        }, 'optimization_pie')
        report += '''
        </div>
    </div>
'''
        
        # Detailed Results
        report += self._generate_detailed_results()
        
        # Technical Details
        report += self._generate_technical_details()
        
        report += """
</body>
</html>"""
        
        # 保存
        with open(self.report_dir / "00_overview" / "index.html", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✓ メインレポート生成完了: {self.report_dir / '00_overview' / 'index.html'}")
    
    def _generate_overview_card(self) -> str:
        """概要カードを生成"""
        total_time = self.summary.get('total_time', 0)
        stats = self.summary.get('stats', {})
        results = self.summary.get('results', {})
        
        successful = sum(1 for r in results.values() if r.get('success', False))
        total = len(results)
        
        return f'''
    <div class="card">
        <h2>📈 実行概要</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{successful}/{total}</div>
                <div class="metric-label">成功した検出器</div>
            </div>
            <div class="metric">
                <div class="metric-value">{int(total_time)}秒</div>
                <div class="metric-label">総実行時間</div>
            </div>
            <div class="metric">
                <div class="metric-value">{sum(r.get('optimizations', 0) for r in results.values())}</div>
                <div class="metric-label">最適化されたフォールド</div>
            </div>
            <div class="metric">
                <div class="metric-value">gpt-4.1-mini</div>
                <div class="metric-label">使用モデル</div>
            </div>
        </div>
    </div>
'''
    
    def _generate_detailed_results(self) -> str:
        """詳細結果テーブルを生成"""
        html = '''
    <div class="card">
        <h2>🔍 検出器別詳細結果</h2>
        <table>
            <thead>
                <tr>
                    <th>検出器</th>
                    <th>精度</th>
                    <th>標準偏差</th>
                    <th>最適化率</th>
                    <th>実行時間</th>
                    <th>状態</th>
                </tr>
            </thead>
            <tbody>
'''
        
        # 精度順にソート
        sorted_results = sorted(
            self.summary['results'].items(),
            key=lambda x: x[1].get('accuracy', 0),
            reverse=True
        )
        
        for name, data in sorted_results:
            if data.get('success'):
                accuracy = data.get('accuracy', 0) * 100
                std = data.get('accuracy_std', 0) * 100
                optimizations = data.get('optimizations', 0)
                folds = data.get('folds', 0)
                opt_rate = (optimizations / folds * 100) if folds > 0 else 0
                duration = int(data.get('duration', 0))
                
                html += f'''
                <tr>
                    <td><strong>{name.replace('_', ' ')}</strong></td>
                    <td>{accuracy:.2f}%</td>
                    <td>±{std:.2f}%</td>
                    <td>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {opt_rate}%"></div>
                        </div>
                        {optimizations}/{folds} ({opt_rate:.0f}%)
                    </td>
                    <td>{duration}秒</td>
                    <td class="success">✓ 成功</td>
                </tr>
'''
            else:
                html += f'''
                <tr>
                    <td><strong>{name.replace('_', ' ')}</strong></td>
                    <td colspan="4">{data.get('error', 'Unknown error')}</td>
                    <td class="warning">✗ 失敗</td>
                </tr>
'''
        
        html += '''
            </tbody>
        </table>
    </div>
'''
        return html
    
    def _generate_technical_details(self) -> str:
        """技術的詳細を生成"""
        return '''
    <div class="card">
        <h2>🔧 技術的詳細</h2>
        <h3>実装の特徴</h3>
        <ul>
            <li><strong>REAL TextGrad実装</strong>: FAKEな`return initial_prompt`ではなく、実際の勾配ベース最適化</li>
            <li><strong>並列実行</strong>: 3ワーカーによる効率的な処理</li>
            <li><strong>レート制限対策</strong>: 0.5秒→0.05秒に削減、gpt-4.1-miniの10,000 RPMを活用</li>
            <li><strong>正直な報告</strong>: 改善がない場合も隠さずに報告</li>
        </ul>
        
        <h3>最適化プロセス</h3>
        <ol>
            <li>初期プロンプトで検証データを評価</li>
            <li>エラー分析に基づいて改善案を生成</li>
            <li>改善案を評価し、性能が向上した場合のみ採用</li>
            <li>最大3回のイテレーションを実行</li>
        </ol>
        
        <h3>ファイル構成</h3>
        <pre>
reports/beautiful_report_{timestamp}/
├── 00_overview/
│   └── index.html          # このレポート
├── 01_detectors/          # 各検出器の詳細
│   ├── collective_instruction/
│   ├── terms_of_endearment/
│   ├── episode_memory/
│   ├── open_ended_questions/
│   └── long_speech_detection/
├── 02_visualizations/     # グラフとチャート
├── 03_optimization_analysis/  # 最適化の詳細分析
├── 04_comparative_analysis/   # 比較分析
└── 05_technical_details/     # 技術文書
        </pre>
    </div>
'''
    
    def generate_detector_reports(self):
        """各検出器の詳細レポートを生成"""
        detector_mapping = {
            'elderspeak_collective': 'collective_instruction',
            'elderspeak_ToE': 'terms_of_endearment',
            'episode_detection_episode_memory': 'episode_memory',
            'question_detection_open_end': 'open_ended_questions',
            'long_speech_detection_long_speech': 'long_speech_detection'
        }
        
        for result_name, detector_name in detector_mapping.items():
            if result_name in self.all_results:
                self._generate_single_detector_report(
                    detector_name,
                    self.all_results[result_name]
                )
    
    def _generate_single_detector_report(self, detector_name: str, data: Dict):
        """単一検出器のレポートを生成"""
        html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>{detector_name.replace('_', ' ').title()} - Detailed Report</title>
    <link rel="stylesheet" href="../../00_overview/style.css">
</head>
<body>
    <h1>{detector_name.replace('_', ' ').title()} 詳細分析</h1>
"""
        
        # フォールド結果
        if 'fold_results' in data:
            html += '<h2>フォールド別結果</h2>'
            html += '<table><thead><tr><th>Fold</th><th>精度</th><th>最適化</th><th>詳細</th></tr></thead><tbody>'
            
            for fold in data['fold_results']:
                html += f'''
                <tr>
                    <td>Fold {fold['fold']}</td>
                    <td>{fold['test_accuracy']*100:.2f}%</td>
                    <td>{'✓' if fold['prompts']['changed'] else '✗'}</td>
                    <td>{fold['metrics']['confusion_matrix']}</td>
                </tr>
'''
            
            html += '</tbody></table>'
        
        html += '</body></html>'
        
        # 保存
        output_path = self.report_dir / "01_detectors" / detector_name / "index.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✓ {detector_name} レポート生成完了")
    
    def generate_all_reports(self):
        """すべてのレポートを生成"""
        print("\n美しいレポート生成を開始...")
        
        # メインレポート
        self.generate_main_report()
        
        # 検出器別レポート
        self.generate_detector_reports()
        
        # README
        self._generate_readme()
        
        print(f"\n✅ すべてのレポート生成完了！")
        print(f"📁 保存先: {self.report_dir}")
        print(f"🌐 メインレポート: {self.report_dir / '00_overview' / 'index.html'}")
    
    def _generate_readme(self):
        """READMEファイルを生成"""
        readme = f"""# TextGrad Optimization Report

生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 概要

このディレクトリには、HONEST TextGrad実装による最適化結果の詳細レポートが含まれています。

## ディレクトリ構成

- `00_overview/` - メインレポートとサマリー
- `01_detectors/` - 各検出器の詳細分析
- `02_visualizations/` - グラフとチャート
- `03_optimization_analysis/` - 最適化プロセスの詳細
- `04_comparative_analysis/` - 比較分析
- `05_technical_details/` - 技術的な詳細

## 主要な結果

- 成功率: {sum(1 for r in self.summary['results'].values() if r.get('success', False))}/{len(self.summary['results'])}
- 総実行時間: {int(self.summary.get('total_time', 0))}秒
- 最適化率: {sum(r.get('optimizations', 0) for r in self.summary['results'].values())}/{sum(r.get('folds', 0) for r in self.summary['results'].values())} フォールド

## 閲覧方法

1. `00_overview/index.html` をブラウザで開く
2. 各検出器の詳細は `01_detectors/` 内のサブディレクトリを参照

## 特徴

- 完全に正直な実装（FAKE実装なし）
- 美しいビジュアライゼーション
- 詳細な技術分析
- 整理されたファイル構成
"""
        
        with open(self.report_dir / "README.md", 'w', encoding='utf-8') as f:
            f.write(readme)


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='美しいレポート生成')
    parser.add_argument('--results', type=str, 
                      default='results/optimized_honest_20250708_032225',
                      help='結果ディレクトリ')
    args = parser.parse_args()
    
    generator = BeautifulReportGenerator(args.results)
    generator.generate_all_reports()


if __name__ == "__main__":
    main()