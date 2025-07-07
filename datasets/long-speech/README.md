# Long Speech Detection Dataset

## Overview
This dataset is designed to detect overly long or complex sentences that may be difficult for elderly individuals with cognitive impairments to understand.

## Dataset Structure
The dataset contains 100 samples in JSON format with the following structure:

```json
{
  "sentences": "文章のテキスト",
  "is_too_long?": true/false
}
```

### Fields
- **sentences**: The text to be analyzed
- **is_too_long?**: Boolean label indicating if the sentence is too long/complex
  - `true`: The sentence is too long or complex for cognitive accessibility
  - `false`: The sentence is appropriately short and simple

## Label Distribution
- **Too Long (true)**: 50 samples (50%)
- **Appropriate (false)**: 50 samples (50%)

## Usage
This dataset is used by the Long Speech Detector to identify sentences that should be simplified for better cognitive accessibility in eldercare communication.

## Performance
- **Accuracy**: 99.00% ± 1.42%
- **Precision**: 0.981
- **Recall**: 1.000
- **F1 Score**: 0.990

## Examples
### Too Long (Label: true)
- Complex sentences with multiple clauses
- Sentences containing technical jargon
- Long explanations that could be simplified

### Appropriate Length (Label: false)
- Short, clear sentences
- Simple questions
- Direct statements

## Integration
This dataset integrates with:
- TextGrad prompt optimization
- Cross-validation evaluation
- Streamlit web interface
- Detection pipeline workflow