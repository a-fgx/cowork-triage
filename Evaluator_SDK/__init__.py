"""
Evaluator SDK - LLM-as-Judge Evaluation Framework using LangSmith

This module provides tools for evaluating LLM performance using
another LLM as a judge (LLM-as-judge pattern) with LangSmith SDK.

Components:
- SpamJudge: Gemini-powered spam classifier
- SpamDatasetLoader: Loader for Kaggle spam dataset
- run_spam_evaluation: Main evaluation function using LangSmith evaluate()

Usage:
    from Evaluator_SDK import run_spam_evaluation

    # Run evaluation
    results = run_spam_evaluation(
        csv_path="data/spam.csv",
        limit=100,
        model="gemini-2.0-flash"
    )
"""

from .spam_judge import SpamJudge, SpamPrediction
from .dataset_loader import SpamDatasetLoader
from .evaluate_spam import (
    run_spam_evaluation,
    correctness_evaluator,
    spam_detection_evaluator,
    ham_preservation_evaluator,
    confidence_calibration_evaluator,
)

__all__ = [
    "SpamJudge",
    "SpamPrediction",
    "SpamDatasetLoader",
    "run_spam_evaluation",
    "correctness_evaluator",
    "spam_detection_evaluator",
    "ham_preservation_evaluator",
    "confidence_calibration_evaluator",
]
