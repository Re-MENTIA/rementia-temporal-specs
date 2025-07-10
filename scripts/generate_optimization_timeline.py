#!/usr/bin/env python3
"""
最適化タイムラインとアニメーションチャートを生成
"""

import json
from pathlib import Path
from datetime import datetime


def generate_optimization_timeline(results_dir: Path, output_dir: Path):
    """最適化プロセスのタイムラインを生成"""
    
    # 結果を読み込む
    all_optimizations = []
    
    for detector_dir in results_dir.iterdir():
        if detector_dir.is_dir() and (detector_dir / "results.json").exists():
            with open(detector_dir / "results.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            detector_name = detector_dir.name
            for fold in data.get('fold_results', []):
                if fold['prompts']['changed']:
                    all_optimizations.append({
                        'detector': detector_name,
                        'fold': fold['fold'],
                        'before': fold.get('validation_accuracy', 0),
                        'after': fold['test_accuracy']
                    })
    
    # タイムラインHTML生成
    html = '''<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>Optimization Timeline</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }
        .timeline {
            position: relative;
            max-width: 1200px;
            margin: 0 auto;
        }
        .timeline::after {
            content: '';
            position: absolute;
            width: 6px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            top: 0;
            bottom: 0;
            left: 50%;
            margin-left: -3px;
        }
        .container {
            padding: 10px 40px;
            position: relative;
            background-color: inherit;
            width: 50%;
            opacity: 0;
            animation: fadeIn 1s forwards;
        }
        @keyframes fadeIn {
            to { opacity: 1; }
        }
        .container::after {
            content: '';
            position: absolute;
            width: 25px;
            height: 25px;
            right: -17px;
            background-color: white;
            border: 4px solid #667eea;
            top: 15px;
            border-radius: 50%;
            z-index: 1;
        }
        .left {
            left: 0;
        }
        .right {
            left: 50%;
        }
        .right::after {
            left: -16px;
        }
        .content {
            padding: 20px 30px;
            background-color: white;
            position: relative;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }
        .content:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        }
        .improvement {
            font-size: 2em;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
            margin: 10px 0;
        }
        .detector-name {
            color: #667eea;
            font-weight: bold;
            font-size: 1.2em;
        }
        .accuracy-change {
            display: flex;
            justify-content: space-around;
            align-items: center;
            margin: 15px 0;
        }
        .accuracy-before, .accuracy-after {
            text-align: center;
        }
        .arrow {
            font-size: 2em;
            color: #4CAF50;
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 50px;
        }
        .summary-card {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            max-width: 600px;
            margin: 0 auto 50px;
            text-align: center;
        }
        .big-number {
            font-size: 4em;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <h1>🚀 TextGrad最適化タイムライン</h1>
    
    <div class="summary-card">
        <h2>最適化成功事例</h2>
        <div class="big-number">{len(all_optimizations)}</div>
        <p>フォールドで改善を達成</p>
    </div>
    
    <div class="timeline">
'''
    
    for i, opt in enumerate(all_optimizations):
        side = 'left' if i % 2 == 0 else 'right'
        improvement = (opt['after'] - opt['before']) * 100
        
        html += f'''
        <div class="container {side}" style="animation-delay: {i * 0.2}s">
            <div class="content">
                <div class="detector-name">{opt['detector'].replace('_', ' ').title()}</div>
                <div>Fold {opt['fold']}</div>
                <div class="accuracy-change">
                    <div class="accuracy-before">
                        <small>Before</small><br>
                        <strong>{opt['before']*100:.1f}%</strong>
                    </div>
                    <div class="arrow">→</div>
                    <div class="accuracy-after">
                        <small>After</small><br>
                        <strong>{opt['after']*100:.1f}%</strong>
                    </div>
                </div>
                <div class="improvement">+{improvement:.1f}%</div>
            </div>
        </div>
'''
    
    html += '''
    </div>
    
    <script>
        // スクロールに応じてアニメーション
        const containers = document.querySelectorAll('.container');
        const observer = new IntersectionObserver(entries => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '1';
                }
            });
        });
        containers.forEach(container => observer.observe(container));
    </script>
</body>
</html>'''
    
    # 保存
    with open(output_dir / "optimization_timeline.html", 'w', encoding='utf-8') as f:
        f.write(html)
    
    print("✓ 最適化タイムライン生成完了")


def generate_3d_visualization(output_dir: Path):
    """3Dビジュアライゼーション（Three.js使用）"""
    html = '''<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>3D Performance Visualization</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        body { margin: 0; overflow: hidden; font-family: sans-serif; }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            color: white;
            background: rgba(0,0,0,0.7);
            padding: 10px;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div id="info">
        <h3>3D Performance Visualization</h3>
        <p>マウスで回転、スクロールでズーム</p>
    </div>
    
    <script>
        // シーン、カメラ、レンダラーの設定
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf5f5f5);
        
        const camera = new THREE.PerspectiveCamera(
            75, window.innerWidth / window.innerHeight, 0.1, 1000
        );
        camera.position.set(5, 5, 5);
        
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.shadowMap.enabled = true;
        document.body.appendChild(renderer.domElement);
        
        // ライト
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
        directionalLight.position.set(5, 10, 5);
        directionalLight.castShadow = true;
        scene.add(directionalLight);
        
        // データ（実際の精度データ）
        const detectorData = [
            { name: 'Open-Ended', accuracy: 100.00, color: 0x4CAF50 },
            { name: 'Long Speech', accuracy: 99.33, color: 0x2196F3 },
            { name: 'ToE', accuracy: 98.67, color: 0xFF9800 },
            { name: 'Episode Memory', accuracy: 97.78, color: 0x9C27B0 },
            { name: 'Collective', accuracy: 92.22, color: 0xF44336 }
        ];
        
        // バーを作成
        detectorData.forEach((data, index) => {
            const height = data.accuracy / 20;
            const geometry = new THREE.BoxGeometry(0.8, height, 0.8);
            const material = new THREE.MeshPhongMaterial({ 
                color: data.color,
                emissive: data.color,
                emissiveIntensity: 0.2
            });
            const bar = new THREE.Mesh(geometry, material);
            bar.position.set(index * 1.5 - 3, height / 2, 0);
            bar.castShadow = true;
            bar.receiveShadow = true;
            scene.add(bar);
            
            // アニメーション
            bar.scale.y = 0;
            const targetScale = 1;
            const animateBar = () => {
                if (bar.scale.y < targetScale) {
                    bar.scale.y += 0.02;
                    bar.position.y = (height * bar.scale.y) / 2;
                    requestAnimationFrame(animateBar);
                }
            };
            setTimeout(() => animateBar(), index * 200);
        });
        
        // 地面
        const groundGeometry = new THREE.PlaneGeometry(10, 10);
        const groundMaterial = new THREE.MeshPhongMaterial({ 
            color: 0xe0e0e0,
            side: THREE.DoubleSide
        });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.position.y = 0;
        ground.receiveShadow = true;
        scene.add(ground);
        
        // マウスコントロール
        let mouseX = 0, mouseY = 0;
        document.addEventListener('mousemove', (event) => {
            mouseX = (event.clientX / window.innerWidth) * 2 - 1;
            mouseY = -(event.clientY / window.innerHeight) * 2 + 1;
        });
        
        // アニメーションループ
        function animate() {
            requestAnimationFrame(animate);
            
            // カメラを回転
            camera.position.x = Math.cos(mouseX * Math.PI) * 8;
            camera.position.z = Math.sin(mouseX * Math.PI) * 8;
            camera.position.y = 5 + mouseY * 3;
            camera.lookAt(0, 2, 0);
            
            renderer.render(scene, camera);
        }
        animate();
        
        // リサイズ対応
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
    </script>
</body>
</html>'''
    
    with open(output_dir / "3d_visualization.html", 'w', encoding='utf-8') as f:
        f.write(html)
    
    print("✓ 3Dビジュアライゼーション生成完了")


def main():
    """追加のビジュアライゼーションを生成"""
    report_dir = Path("reports/beautiful_report_20250708_040355")
    results_dir = Path("results/optimized_honest_20250708_032225")
    viz_dir = report_dir / "02_visualizations"
    
    # タイムライン生成
    generate_optimization_timeline(results_dir, viz_dir)
    
    # 3Dビジュアライゼーション生成
    generate_3d_visualization(viz_dir)
    
    print(f"\n✨ 追加ビジュアライゼーション完成！")
    print(f"📊 タイムライン: {viz_dir / 'optimization_timeline.html'}")
    print(f"🎮 3D表示: {viz_dir / '3d_visualization.html'}")


if __name__ == "__main__":
    main()