#!/usr/bin/env python3
"""
Demo script showing the complete detection system
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.workflow import DetectionPipeline


def print_separator():
    print("=" * 60)


def demo_analysis(pipeline, text, description):
    """Run analysis and display results"""
    print_separator()
    print(f"ANALYZING: {description}")
    print(f"Text: {text}")
    print_separator()
    
    # Run analysis
    result = pipeline.analyze(text)
    
    # Display results
    print(f"\n🎯 Risk Level: {result.risk_level.upper()}")
    
    print("\n📊 Detection Results:")
    for detector, detection in result.detections.items():
        if 'error' not in detection:
            status = "⚠️ DETECTED" if detection['prediction'] == '1' else "✅ SAFE"
            print(f"  {detector}: {status} - {detection['label']}")
    
    print("\n💡 Recommendations:")
    for rec in result.recommendations:
        print(f"  • {rec}")
    
    print()


def main():
    """Run demo"""
    print("🏥 Eldercare Communication Analyzer - Demo")
    print_separator()
    
    # Initialize pipeline with optimized prompts
    prompts_path = "src/prompts/optimized_prompts.json"
    if Path(prompts_path).exists():
        print("✓ Loading optimized prompts")
        pipeline = DetectionPipeline(prompts_path)
    else:
        print("ℹ Using default prompts")
        pipeline = DetectionPipeline()
    
    # Test cases
    test_cases = [
        ("こんにちは。今日はいい天気ですね。", "Safe greeting"),
        ("おばあちゃん、今日も可愛いね！", "Inappropriate endearment"),
        ("昨日の夕食で何を食べましたか？", "Episode memory question"),
        ("コーヒーと紅茶、どちらがお好きですか？", "Safe preference question"),
        ("先週の旅行はどうでしたか？詳しく聞かせてください。", "High-risk: Open-ended + Episode"),
    ]
    
    # Run demos
    for text, description in test_cases:
        demo_analysis(pipeline, text, description)
        input("Press Enter to continue...")
    
    print_separator()
    print("Demo complete! Run the Streamlit app for an interactive interface:")
    print("streamlit run src/app/streamlit_app.py")


if __name__ == "__main__":
    main()