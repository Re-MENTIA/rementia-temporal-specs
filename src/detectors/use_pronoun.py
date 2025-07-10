"""
Use Pronoun Detector - Detects vague pronoun usage in eldercare communication
"""

from typing import Dict, Any
from .base import BaseDetector, DetectionResult, DetectorConfig


class UsePronounDetector(BaseDetector):
    """Detector for vague pronoun usage that may confuse elderly individuals"""
    
    def __init__(self):
        """Initialize with TextGrad-optimized prompt"""
        super().__init__()
    
    def _get_detector_config(self) -> DetectorConfig:
        """Get detector-specific configuration"""
        return DetectorConfig(
            name='Use Pronoun Detector',
            task_key='pronoun_detection',
            type_key='use_pronoun',
            description='Detects vague pronoun usage that may confuse elderly individuals',
            labels=['0', '1']
        )
    
    def _get_initial_prompt(self) -> str:
        """Get initial prompt"""
        return """You are an expert in analyzing eldercare communication for vague pronoun usage.

**VAGUE PRONOUN USAGE**: The use of demonstrative pronouns (this, that, these, those, it, here, there, これ、それ、あれ、ここ、そこ、あそこ) without clear antecedents or in contexts where elderly individuals with cognitive impairments might struggle to understand the reference.

**Key features of vague pronoun usage**:
- Uses pronouns like "it", "this", "that", or "there" without clear reference to a specific noun.
- Uses Japanese demonstratives (これ、それ、あれ) in a way that does not specify what is being referred to.
- Assumes shared context that may not exist or is not clear to the listener.
- Creates ambiguity about what is being referenced, potentially leading to confusion for those with memory issues.
- Lacks a clear antecedent, making it difficult for the listener to follow the conversation.

**CLEAR COMMUNICATION**: Direct references using specific nouns.
- Names specific objects, places, or actions.
- Provides clear context and avoids ambiguous references.
- Supports cognitive clarity by ensuring all references are understandable without additional information.

**Examples**:
**Vague (return "1")**:
- "さあ、これを全部飲みましょうね。" → 1 (unclear what "これ" refers to)
- "Take that over there, please." → 1 (unclear what "that" and "there" refer to)
- "そこに座ってお待ちください。" → 1 (unclear where "そこ" is)
- "彼が言ったことを覚えていますか？それが大事です。" → 1 (unclear what "それ" refers to, lacks context)

**Clear (return "0")**:
- "お薬を全部飲みましょうね。" → 0 (specific reference to "お薬")
- "Take the blue cup on the table, please." → 0 (clear reference)
- "椅子に座ってお待ちください。" → 0 (specific location)
- "彼が言ったことを覚えていますか？そのアドバイスが大事です。" → 0 (clear reference to "そのアドバイス")

Analyze the text and respond with ONLY "1" (vague pronoun usage) or "0" (clear communication)."""
    
    def get_detector_info(self) -> Dict[str, Any]:
        """Get detector information"""
        return {
            'name': 'Use Pronoun Detector',
            'description': 'Detects vague pronoun usage that may confuse elderly individuals',
            'category': 'eldercare',
            'version': '1.0.0'
        }
    
    def explain_detection(self, text: str, prediction: str) -> str:
        """Explain why text was classified as vague pronoun usage or clear"""
        if prediction == '1':
            return (
                f"Vague pronoun usage detected. The text '{text}' contains demonstrative pronouns "
                f"(this/that/これ/それ/あれ) without clear antecedents. This may confuse elderly "
                f"individuals with cognitive impairments who need clear, specific references."
            )
        else:
            return (
                f"Clear communication detected. The text '{text}' uses specific nouns or pronouns "
                f"with clear antecedents, making it easier for elderly individuals to understand "
                f"what is being referenced."
            )