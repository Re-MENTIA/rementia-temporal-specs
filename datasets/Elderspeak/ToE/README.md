# Terms of Endearment Detection Dataset

This dataset is designed for training and evaluating AI systems to detect inappropriate terms of endearment and diminutives in eldercare contexts.

## Dataset Structure

The dataset follows TextGrad's required format:
- **Format**: JSON array of objects with `input` (text) and `label` (classification)
- **Labels**: 
  - `"1"`: Harmful - Contains inappropriate terms of endearment
  - `"0"`: Safe - Appropriate, respectful communication

## Dataset Statistics

- **Total samples**: 150
- **Harmful (1)**: 50 samples
- **Safe (0)**: 100 samples (50 appropriate eldercare + 50 neutral)
- **Text length**: 2-4 sentences per sample

## Key Features to Detect

### Harmful Patterns (Label "1"):
- Pet names: honey, sweetie, darling, sugar, baby, angel
- Japanese diminutives: -chan, -kun, ちゃん
- Infantilizing language: "good boy/girl", "cutie pie"
- Baby talk: elongated vowels, childish expressions
- Patronizing tone treating adults as children

### Safe Patterns (Label "0"):
- Respectful titles: Mr./Mrs./Ms. + surname
- Japanese honorifics: -san, -sama, 様
- Professional, dignified language
- Age-appropriate communication

## Usage

### Training with TextGrad:
```bash
cd datasets/Elderspeak
python terms_of_endearment_train.py
```

### Expected Output:
1. Optimized prompt after TextGrad training
2. Confusion matrix showing classification performance
3. Examples of correct and incorrect predictions
4. Saved files:
   - `optimized_prompt_terms_of_endearment.txt`
   - `evaluation_results_terms_of_endearment.json`

## Data Generation Guidelines

This dataset was created following the guidelines in `/docs/Elderspeak/EoT.md`:
- Diverse topics (healthcare, daily activities, news, etc.)
- Natural AI-like responses
- No repetitive patterns
- Balanced representation of different inappropriate terms

## Evaluation Metrics

The system is evaluated using:
- Accuracy
- Confusion Matrix
- Precision/Recall/F1-score
- Example analysis of predictions

## Notes

- The dataset includes both English and Japanese examples
- Neutral samples (general news/information) are labeled as "0"
- The goal is to help AI systems maintain respectful communication with elderly individuals