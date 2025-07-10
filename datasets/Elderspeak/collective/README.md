# Collective Instruction Dataset

## Overview
This dataset is designed to detect inappropriate use of collective pronouns ("we", "us", "our", "let's", "私たち", "一緒に") when giving instructions in eldercare settings. This is a form of elderspeak that removes individual autonomy by masking the directive nature of instructions.

## Background
Collective instruction is one of the 11 elderspeak attributes identified in the Iowa Coding Scheme for Elderspeak (ICodE). Research shows that this pattern appears in 66-69% of care interactions, making it one of the most frequent elderspeak features. It's particularly prevalent when communicating with individuals with cognitive impairments.

## Dataset Structure
The dataset contains 90 samples in JSON format with the following structure:

```json
{
  "instruction_type": "collective/non-collective/neutral",
  "sentence": "Instruction or statement text"
}
```

### Fields
- **sentence**: The text to be analyzed
- **instruction_type**: Three categories that are mapped to binary labels:
  - `collective`: Instructions using inappropriate "we/us" forms → Label: 1 (harmful)
  - `non-collective`: Direct instructions using appropriate pronouns → Label: 0 (safe)
  - `neutral`: Non-instructional text → Label: 0 (safe)

## Label Distribution
- **Collective (harmful)**: 30 samples (33.3%) → Label: 1
- **Non-collective + Neutral (safe)**: 60 samples (66.7%) → Label: 0

## Examples

### Collective Instructions (Label: 1)
- "さあ、一緒にお薬を飲みましょうね。" (Let's take our medicine together)
- "私たちで今夜の献立を決めましょうか。" (Shall we decide tonight's menu?)
- "Let's take our bath now" (when only the patient will bathe)

### Non-collective Instructions (Label: 0)
- "お薬を飲んでください。" (Please take your medicine)
- "窓を閉めてください。" (Please close the window)
- Direct, respectful commands that acknowledge individual autonomy

### Neutral Statements (Label: 0)
- "今日は良い天気です。" (It's nice weather today)
- Observations or descriptions without any instructional content

## Performance
- **Accuracy**: 100.00%
- **Precision**: 1.000
- **Recall**: 1.000
- **F1 Score**: 1.000

## Clinical Significance
While collective pronouns may seem polite, they can be problematic in care settings because they:
- Remove individual autonomy and decision-making power
- Create a false sense of shared action when only the care recipient performs the task
- Mask the directive nature of instructions
- Treat adults as lacking independent agency

## Integration
This dataset integrates with:
- TextGrad prompt optimization
- Cross-validation evaluation
- Streamlit web interface
- Detection pipeline workflow as part of elderspeak detection