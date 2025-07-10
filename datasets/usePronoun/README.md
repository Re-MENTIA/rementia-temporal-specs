# Use Pronoun Dataset

This dataset is designed to detect vague pronoun usage in eldercare communication that could create confusion or safety risks.

## Dataset Structure

The dataset contains 90 samples with the following fields:
- `sentence`: The text to be analyzed
- `label`: Binary classification (1 = harmful vague pronoun usage, 0 = safe/clear communication)
- `language`: Either "Japanese" or "English"
- `context`: Description of the scenario and why it's classified as harmful or safe

## Label Definitions

### Label 1 (Harmful - Vague Pronoun Usage)
Examples that contain:
- Demonstrative pronouns (これ/それ/あれ, this/that/those) without clear antecedents
- Personal pronouns (彼/彼女, he/she/they) without prior identification
- Spatial references (ここ/そこ/あそこ, here/there) without specific locations
- Multiple vague pronouns that compound confusion
- Instructions that rely on unclear references

Common patterns:
- "これを飲んでください" (Take this) - without specifying what medication
- "Put that over there" - unclear item and location
- "彼女が来たら、彼に渡してください" - unclear who "she" and "he" refer to

### Label 0 (Safe - Clear Communication)
Examples that contain:
- Specific nouns instead of pronouns ("田中さん", "血圧の薬", "青いセーター")
- Clear antecedents when pronouns are used
- Proper context that makes references unambiguous
- Non-instructional neutral statements
- Conversational text without confusing references

## Dataset Distribution
- Total samples: 90
- Harmful (label 1): 30 samples (33.3%)
- Safe (label 0): 60 samples (66.7%)
- Languages: Balanced mix of Japanese and English examples

## Use Cases

This dataset is particularly relevant for:
1. Training AI systems to detect potentially confusing communication in eldercare settings
2. Improving clarity in caregiver-patient interactions
3. Preventing medication errors or safety issues due to unclear instructions
4. Supporting dementia care where clear communication is crucial

## Context Categories

The dataset covers various eldercare scenarios:
- Healthcare/Medication administration
- Daily care activities
- Social interactions
- Movement and mobility
- Dressing assistance
- Pain assessment
- Task sequencing
- Location guidance
- Neutral conversations

## Example Usage

```python
import json

# Load the dataset
with open('use_pronoun_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Filter harmful examples
harmful_examples = [item for item in data if item['label'] == 1]

# Filter by language
japanese_examples = [item for item in data if item['language'] == 'Japanese']
```

## Notes

- The dataset focuses on realistic, nuanced examples rather than obvious cases
- Examples are drawn from typical eldercare communication scenarios
- Both formal and informal speech patterns are included
- The dataset can be used for binary classification or as training data for language models