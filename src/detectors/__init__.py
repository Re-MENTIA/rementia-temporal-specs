"""
Unified Detection System for Eldercare Communication Analysis
"""

from .base import BaseDetector
from .accommodation import AccommodationSpeechDetector
from .episode_memory import EpisodeMemoryDetector
from .open_end_question import OpenEndQuestionDetector
from .long_speech import LongSpeechDetector

__all__ = [
    'BaseDetector',
    'AccommodationSpeechDetector', 
    'EpisodeMemoryDetector',
    'OpenEndQuestionDetector',
    'LongSpeechDetector'
]