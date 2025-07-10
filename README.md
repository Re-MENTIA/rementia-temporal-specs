# Eldercare Communication Analyzer

A comprehensive system for detecting problematic communication patterns in eldercare settings using LLM-as-judge approach with prompt optimization. Includes both research tools and an interactive Streamlit application.

## 🎯 Key Features

### Detection Capabilities
1. **Elderspeak Detection** (2 subtypes):
   - **Terms of Endearment**: Inappropriate pet names or baby talk (honey, sweetie, ちゃん suffix)
   - **Collective Instruction**: Inappropriate use of "we/us" when giving instructions
2. **Episode Memory Detection**: Detects questions requiring personal memory recall  
3. **Open-End Question Detection**: Identifies questions needing elaborate responses
4. **Long Speech Detection**: Identifies overly long or complex sentences that may be difficult to process

### Technical Features
- **LLM-as-Judge**: Uses OpenAI GPT models for evaluation
- **Cross-Validation**: Stratified k-fold cross-validation with multiple repeats
- **Prompt Optimization**: Automatic prompt refinement based on validation performance
- **Interactive UI**: Streamlit application for real-time analysis
- **Comprehensive Reporting**: JSON and Markdown reports with visualizations
- **TextGrad Integration**: Support for gradient-based prompt optimization

## 📊 Performance

- **Elderspeak Detection**:
  - **Terms of Endearment**: ~85% accuracy
  - **Collective Instruction**: 100% accuracy (F1: 1.000)
- **Episode Memory**: ~98% accuracy  
- **Open-End Question**: ~100% accuracy
- **Long Speech**: 99% accuracy (F1: 0.990)

## 🗂️ Project Structure

```
├── config/
│   └── config.yaml             # Main configuration file
├── datasets/
│   ├── Elderspeak/            # Elderspeak detection datasets
│   │   ├── ToE/              # Terms of Endearment
│   │   └── collective/       # Collective Instruction
│   ├── episode-open-end-Q/   # Episode memory & open-end datasets
│   └── long-speech/          # Long speech detection dataset
├── examples/                  # Demo scripts
│   ├── demo.py               # Interactive demo
│   └── demo_noninteractive.py # Non-interactive demo
├── scripts/                   # Main execution scripts
│   ├── evaluate_all_detectors.py
│   ├── optimize_all_prompts.py
│   └── run_unified_crossval.py
├── src/
│   ├── app/                  # Streamlit application
│   ├── detectors/           # Detection modules
│   ├── models/              # Model wrappers
│   ├── optimization/        # Prompt optimization
│   ├── prompts/             # Optimized prompts storage
│   ├── utils/               # Utility functions
│   └── workflow/            # Detection pipeline
└── results/                  # Evaluation results
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. API Key Configuration
Create a `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

### 3. Run the Streamlit App
```bash
streamlit run src/app/streamlit_app.py
```

## 💻 Usage Options

### Interactive Streamlit App (Recommended)

```bash
# Option 1: Use default prompts
streamlit run src/app/streamlit_app.py

# Option 2: Optimize prompts first (better performance)
python scripts/optimize_all_prompts.py
streamlit run src/app/streamlit_app.py
```

### Command Line Evaluation

```bash
# Evaluate all detectors
python scripts/evaluate_all_detectors.py

# Run specific cross-validation
python scripts/run_unified_crossval.py --task elderspeak --type ToE
python scripts/run_unified_crossval.py --task elderspeak --type collective
python scripts/run_unified_crossval.py --task episode_detection --type episode_memory
python scripts/run_unified_crossval.py --task long_speech_detection --type long_speech

# Run demo
python examples/demo_noninteractive.py
```

### Prompt Optimization

```bash
# Optimize prompts for all detectors
python scripts/optimize_all_prompts.py --output src/prompts/optimized_prompts.json
```

## 🎨 Streamlit App Features

1. **Real-time Analysis**: Enter text and get instant risk assessment
2. **Risk Levels**: 
   - 🟢 **Low Risk**: Safe communication patterns
   - 🟡 **Medium Risk**: Could be improved for accessibility
   - 🔴 **High Risk**: May cause confusion or distress
3. **Detailed Recommendations**: Specific suggestions for improving communication
4. **Example Library**: Pre-loaded examples for testing

## ⚙️ Configuration

Edit `config/config.yaml` to configure:

- **Models**: Change between GPT models (gpt-4.1-mini, o4-mini-2025-04-16)
- **Dataset Paths**: Add new detection types
- **Cross-Validation**: Adjust folds, repeats, validation split
- **Prompts**: Define initial prompts for each detection type

### Adding New Detection Types

1. Add dataset configuration:
   ```yaml
   tasks:
     new_detection:
       base_path: "datasets/new_type"
       types:
         new_type:
           name: "New Detection Type"
           path: "new_type/dataset.json"
           labels: ["0", "1"]
   ```

2. Add prompt configuration:
   ```yaml
   prompts:
     new_type:
       initial: |
         Your prompt here...
   ```

3. Create detector class in `src/detectors/`

## 📄 Dataset Format

Datasets should be JSON files with the following structure:
```json
[
  {
    "sentence": "Text to evaluate",
    "question_type": "open-ended",  // For question detection
    "memory_type": "episodic",      // For memory detection
    "label": "1"                    // For accommodation speech
  }
]
```

## 📈 Output

Results are saved in timestamped directories under `results/`:
- `crossvalidation_results.json`: Detailed cross-validation results
- `evaluation_report.md`: Markdown summary report
- `evaluation_results.png`: 9-panel visualization
- `prompt_analysis.txt`: Prompt optimization details

## 🛠️ Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/
flake8 src/
```

## 📚 References

Based on research in cognitive accessibility and eldercare communication best practices.

## 📄 License

MIT License - see LICENSE file for details.