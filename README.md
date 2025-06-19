# Accommodation Speech Detection System

A research system for detecting various types of accommodation speech in eldercare settings using LLM-as-judge approach with prompt optimization.

## Features

- **Multiple Accommodation Types**: Configurable support for different types of elderspeak
- **LLM-as-Judge**: Uses OpenAI GPT models for evaluation
- **Cross-Validation**: Stratified k-fold cross-validation with multiple repeats
- **Prompt Optimization**: Automatic prompt refinement based on validation performance
- **Comprehensive Reporting**: JSON and Markdown reports with visualizations
- **TextGrad Integration**: Support for gradient-based prompt optimization

## Project Structure

```
├── config/
│   └── config.yaml         # Main configuration file
├── datasets/
│   └── Elderspeak/
│       └── ToE/           # Terms of Endearment datasets
├── docs/                  # Documentation
├── results/              # Evaluation results
├── scripts/
│   └── run_evaluation.py # Main evaluation script
├── src/
│   ├── models/          # Model wrappers
│   └── utils/           # Utility functions
└── textgrad/            # TextGrad library
```

## Setup

1. **Environment Setup**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **API Key Configuration**
   Create a `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Basic Evaluation
```bash
python scripts/run_evaluation.py
```

### Specify Accommodation Type
```bash
python scripts/run_evaluation.py --type ToE
```

### Custom Dataset
```bash
python scripts/run_evaluation.py --dataset path/to/dataset.json
```

### Full Options
```bash
python scripts/run_evaluation.py \
    --config config/config.yaml \
    --type ToE \
    --output results/toe_results.json
```

## Configuration

Edit `config/config.yaml` to configure:

- **Dataset Paths**: Add new accommodation types under `dataset.accommodation_types`
- **Model Settings**: Change models, temperature, max tokens
- **Cross-Validation**: Adjust folds, repeats, validation split
- **Prompts**: Define initial prompts for each accommodation type

### Adding New Accommodation Types

1. Add dataset configuration:
   ```yaml
   dataset:
     accommodation_types:
       NewType:
         name: "New Accommodation Type"
         path: "NewType/dataset.json"
   ```

2. Add prompt configuration:
   ```yaml
   prompts:
     NewType:
       initial: |
         Your prompt here...
   ```

3. Create dataset in `datasets/Elderspeak/NewType/`

## Dataset Format

Datasets should be JSON files with the following structure:
```json
[
  {
    "input": "Text to evaluate",
    "label": "1"  // 1 for harmful, 0 for safe
  }
]
```

## Output

Results are saved in the `results/` directory:
- `crossvalidation_results.json`: Detailed results
- `evaluation_report.md`: Markdown report
- `evaluation_results.png`: Visualizations

## Models

Currently supports:
- GPT-4
- GPT-3.5-turbo
- GPT-4-turbo

Configure in `config/config.yaml` under `models.default`.

## License

MIT License - see LICENSE file for details.