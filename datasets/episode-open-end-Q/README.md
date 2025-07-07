# Episode Memory & Open-End Question Detection Datasets

This directory contains datasets for two binary classification tasks designed to detect inappropriate question types in eldercare conversations.

## Dataset Structure

```
episode-open-end-Q/
├── episode-memory/
│   └── episode_memory_dataset.json    # 90 samples for episodic memory detection
└── open-end-Q/
    └── open_end_Q_dataset.json        # 90 samples for open-end question detection
```

## Episode Memory Detection

### Purpose
Detects whether a question requires episodic memory (personal past experiences) to answer. This is critical in eldercare contexts where individuals with dementia may struggle with episodic memory questions.

### Classification
- **Label 1 (Episodic Memory)**: Questions requiring recall of specific personal experiences
  - Example: "昨日の夕食で何を食べましたか？" (What did you eat for dinner yesterday?)
  
- **Label 0 (Non-Episodic Memory)**: Questions about general knowledge or current state
  - Example: "犬と猫、どちらが好きですか？" (Do you prefer dogs or cats?)

### Performance
- **Accuracy**: 97.78% ± 4.44%
- **Precision**: 0.941 ± 0.118
- **Recall**: 1.000 ± 0.000
- **F1 Score**: 0.966 ± 0.067

## Open-End Question Detection

### Purpose
Identifies open-ended questions that may be challenging for individuals with cognitive impairments, as they require elaborate responses and can cause frustration or confusion.

### Classification
- **Label 1 (Open-Ended)**: Questions allowing free-form responses
  - Example: "最近のニュースで気になったことは何ですか？" (What recent news caught your attention?)
  
- **Label 0 (Closed-Ended/Non-Question)**: YES/NO questions, multiple choice, or statements
  - Example: "ニュースはもう見ましたか？" (Have you watched the news?)

### Performance
- **Accuracy**: 100.00% ± 0.00%
- **Precision**: 1.000 ± 0.000
- **Recall**: 1.000 ± 0.000
- **F1 Score**: 1.000 ± 0.000

## Combined Use Case

These detectors work together to identify problematic question patterns in eldercare:
- **High Risk**: Open-ended questions requiring episodic memory (e.g., "昨日どこに行きましたか？")
- **Low Risk**: Closed-ended questions about preferences (e.g., "コーヒーは好きですか？")

## Data Format

Both datasets follow the same JSON structure:
```json
[
  {
    "sentence": "Question or statement text",
    "question_type": "open-ended|closed-ended|neutral"  // for open-end dataset
    "memory_type": "episodic|semantic|neutral"          // for episode dataset
  }
]
```

## Label Mapping

### Episode Memory Detection
- `episodic` → `1`
- `semantic`, `neutral` → `0`

### Open-End Question Detection  
- `open-ended` → `1`
- `closed-ended`, `neutral` → `0`

## Integration Notes

Both detectors are integrated into the unified evaluation framework and can be accessed via:
- Cross-validation: `scripts/run_unified_crossval.py --task [task_name] --type [type]`
- Streamlit app: Real-time detection interface (coming soon)

## References

Based on research in cognitive accessibility and eldercare communication best practices.