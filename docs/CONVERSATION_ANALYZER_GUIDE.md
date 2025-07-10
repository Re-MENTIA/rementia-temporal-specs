# Conversation History Analyzer Guide

## Overview
The Conversation History Analyzer is a tool that analyzes AI-human conversation logs to detect potentially problematic communication patterns in eldercare contexts.

## Features
- **JSON Validation**: Validates conversation JSON structure before analysis
- **Parallel Processing**: Analyzes multiple AI messages simultaneously for efficiency
- **Comprehensive Detection**: Uses all 6 eldercare communication detectors
- **Risk Assessment**: Categorizes messages by risk level (high/medium/low)
- **Recommendations**: Provides specific advice for improving communication
- **Export Results**: Saves detailed analysis results as JSON

## How to Use

### 1. Start the Streamlit App
```bash
# Using Python script
python run_streamlit.py

# Or using bash script
./run_streamlit.sh
```

### 2. Navigate to Conversation Analyzer
- In the sidebar, select "Conversation History Analyzer" from the dropdown

### 3. Upload Conversation JSON
- Click "Browse files" to upload your conversation JSON file
- The file should contain an array of message objects

### Expected JSON Format
```json
[
  {
    "role": "user",
    "content": "Message content here"
  },
  {
    "role": "ai",  // or "assistant"
    "content": "AI response here",
    "timestamp": "2024-01-15T10:00:00"  // optional
  }
]
```

### 4. Analyze Conversation
- Click "🔍 Analyze Conversation" button
- Wait for analysis to complete (parallel processing makes it fast!)

### 5. View Results
The analysis results include:
- **Summary Statistics**: Total messages, risk distribution
- **Risk Distribution Chart**: Visual breakdown of risk levels
- **Most Common Issues**: Top detected problems
- **Message-by-Message Analysis**: Detailed breakdown of each AI message

### 6. Download Results
- Click "Download Analysis Results (JSON)" to save the complete analysis

## Analysis Output Format
```json
{
  "total_messages": 12,
  "ai_messages_analyzed": 6,
  "user_messages": 6,
  "high_risk_messages": 2,
  "medium_risk_messages": 3,
  "low_risk_messages": 1,
  "message_analyses": [
    {
      "message_id": 1,
      "role": "ai",
      "content": "さあ、一緒にお薬を飲みましょうね。",
      "timestamp": "2024-01-15T10:00:15",
      "detections": {
        "collective_instruction": {
          "prediction": "1",
          "label": "Collective instruction detected"
        }
      },
      "risk_level": "high",
      "recommendations": [
        "Use direct instructions instead of collective language"
      ]
    }
  ],
  "summary": {
    "detector_statistics": {...},
    "most_common_issues": [...],
    "recommendations_summary": {...}
  },
  "generated_at": "2024-01-15T14:30:00"
}
```

## Sample File
A sample conversation file is provided at:
`samples/conversation_sample.json`

## Tips
- Filter results by risk level to focus on problematic messages
- Use the recommendations to improve AI communication patterns
- The analyzer processes AI messages in parallel for faster results
- All 6 detectors run on each message:
  - Terms of Endearment
  - Collective Instruction
  - Episode Memory
  - Open-End Questions
  - Long Speech
  - Use Pronoun (vague references)