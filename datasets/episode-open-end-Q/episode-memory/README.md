# Episode Memory Detection Dataset

## Overview

This dataset is designed to train and evaluate models that can distinguish between episodic memory references and non-episodic memory content in Japanese text. This is particularly relevant in eldercare contexts where understanding memory types can inform communication strategies.

## Dataset Description

- **File**: `episode_memory_dataset.json`
- **Total Samples**: 90 (originally 120 with 3 classes, converted to 90 for binary classification)
- **Languages**: Japanese
- **Task**: Binary classification

## Label Distribution

- **Episodic Memory (label: "1")**: 30 samples (33.3%)
  - Originally labeled as "true" in the dataset
- **Non-Episodic Memory (label: "0")**: 60 samples (66.7%)
  - Combines original "false" (30) and "neutral" (30) samples

## Memory Type Definitions

### Episodic Memory (Label: "1")
Personal memories of specific events that an individual has experienced, characterized by:
- **Spatiotemporal context**: Clear "when" and "where"
- **Personal experience**: First-person autobiographical events
- **Contextual details**: Emotions, sensory information, situations
- **Examples**:
  - "去年のクリスマスに家族で旅行した時のことを覚えている？" (Do you remember when we traveled with family last Christmas?)
  - "小学校の入学式の日に、校門で写真を撮った記憶が今でも残っているよ。" (I still have the memory of taking a photo at the school gate on elementary school entrance ceremony day)

### Non-Episodic Memory (Label: "0")
All other types of memory and non-memory content, including:
- **Semantic memory**: General knowledge, facts (e.g., "東京は日本の首都である" - Tokyo is the capital of Japan)
- **Procedural memory**: Skills and how-to knowledge (e.g., "自転車の乗り方は一度覚えれば簡単には忘れない" - Once you learn how to ride a bicycle, you don't easily forget)
- **Future/Hypothetical**: Speculations and plans (e.g., "今日は雨が降るかもしれないね" - It might rain today)
- **General observations**: Non-personal statements (e.g., "最近、新しいカフェがオープンしたらしいよ" - A new cafe apparently opened recently)

## Data Format

```json
[
  {
    "sentence": "去年のクリスマスに家族で旅行した時のことを覚えている？",
    "is_epsodic_memory_related": "true"
  }
]
```

Note: The field name contains a typo ("epsodic" instead of "episodic") which is maintained for compatibility.

## Evaluation Results

Using GPT-4 with our improved prompt:
- **Accuracy**: 97.78% ± 2.22% (2-fold cross-validation)
- **Precision (Episodic)**: 94.1% ± 5.9%
- **Recall (Episodic)**: 100.0% ± 0.0%
- **F1 Score (Episodic)**: 96.9% ± 3.1%

## Usage

```python
# Load dataset
import json
with open('episode_memory_dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# Convert labels for binary classification
for item in dataset:
    if item['is_epsodic_memory_related'] == 'true':
        item['label'] = '1'  # Episodic memory
    else:
        item['label'] = '0'  # Non-episodic (false or neutral)
```

## Applications

1. **Cognitive Assessment**: Automated analysis of patient responses
2. **Communication Training**: Teaching appropriate question types for memory-impaired individuals
3. **Research**: Understanding memory patterns in eldercare conversations

## Notes

- The dataset focuses on naturalistic Japanese expressions
- Binary classification simplifies the original 3-class problem for practical applications
- High performance suggests the task is well-defined with clear boundaries