# Elderspeak Detection Datasets

## Overview
Elderspeak is a speech register characterized by modifications in speech patterns when younger individuals communicate with older adults. It includes simplified vocabulary, exaggerated intonation, slower speech rate, and various linguistic features that can be perceived as patronizing or demeaning.

## Iowa Coding Scheme for Elderspeak (ICodE)
The datasets in this directory are based on the Iowa Coding Scheme for Elderspeak, which identifies 11 linguistic attributes commonly found in elderspeak. Currently, we have implemented detection for two key attributes:

### 1. Terms of Endearment (ToE)
- Inappropriate use of pet names, baby talk, or infantilizing expressions
- Examples: honey, sweetie, darling, ちゃん suffix, よしよし
- Located in: `ToE/` directory
- Performance: ~85% accuracy

### 2. Collective Instruction
- Inappropriate use of inclusive pronouns ("we/us/our/let's") when giving instructions where the speaker won't actually participate
- Includes emotional manipulation patterns like "私のために" (for my sake) or "私も心配だから" (because I'm worried too)
- Examples: 
  - "Let's take our medicine" (when only the patient takes medicine)
  - "私のためにも、お薬飲んでいただけますか？" (Could you take your medicine for my sake too?)
- Located in: `collective/` directory
- Dataset enhanced with 20% challenging samples including subtle manipulation patterns and genuine collaborative activities
- Performance: 100% accuracy (dataset recently made more challenging)

## Why Elderspeak Matters
Research shows that elderspeak:
- Can negatively impact older adults' self-esteem and sense of competence
- May lead to increased resistance to care
- Is associated with cognitive and functional decline
- Undermines the dignity and autonomy of care recipients

## Future Expansions
The ICodE scheme includes additional attributes that could be added:
- Shortened sentences
- Expanded prosody (exaggerated intonation)
- High pitch
- Slow speech rate
- Use of repetition
- Use of diminutives
- Tag questions
- Controlling language
- Clarification strategies

## Dataset Format
All datasets follow a consistent JSON format with:
- An input field (text to analyze)
- A label field (classification result)
- Label mapping for converting categorical labels to binary classification

## Integration
These datasets are integrated with:
- The unified detection pipeline
- TextGrad prompt optimization
- Cross-validation evaluation system
- Streamlit web interface