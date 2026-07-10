"""
External Judge Package

This package provides simplified safety judges for batch evaluation.
All judges support the evaluate_batch() method that returns simple safe/unsafe lists.
"""

from .llama_guard import LlamaGuard2Judge, evaluate_safety_batch_llama
from .md_judge import MDJudge, evaluate_safety_batch
from .combined_judge import CombinedJudge, evaluate_safety_batch_combined

__all__ = [
    'LlamaGuard2Judge',
    'MDJudge',
    'CombinedJudge',
    'evaluate_safety_batch_llama',
    'evaluate_safety_batch',
    'evaluate_safety_batch_combined'
]

__version__ = "1.0.0" 