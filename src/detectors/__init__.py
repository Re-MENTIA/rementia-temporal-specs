"""
Unified Detection System for Eldercare Communication Analysis
"""

from .base import BaseDetector
from .terms_of_endearment import TermsOfEndearmentDetector
from .collective_instruction import CollectiveInstructionDetector
from .episode_memory import EpisodeMemoryDetector
from .open_end_question import OpenEndQuestionDetector
from .long_speech import LongSpeechDetector
from .use_pronoun import UsePronounDetector
from .detector_registry import DetectorRegistry, get_detector_registry

__all__ = [
    'BaseDetector',
    'TermsOfEndearmentDetector',
    'CollectiveInstructionDetector', 
    'EpisodeMemoryDetector',
    'OpenEndQuestionDetector',
    'LongSpeechDetector',
    'UsePronounDetector',
    'DetectorRegistry',
    'get_detector_registry'
]