"""
Spam Judge - LLM-as-Judge for Spam Classification

This module implements an LLM-as-judge evaluator that uses Gemini
to classify emails as spam or ham (legitimate).
"""

import sys
from pathlib import Path
from typing import TypedDict, Literal
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm import get_llm, invoke_structured


class SpamPrediction(TypedDict):
    """Structured output for spam classification."""
    classification: Literal["spam", "ham"]
    confidence: float  # 0.0 to 1.0
    reasoning: str


SPAM_JUDGE_PROMPT = """You are an expert email spam classifier. Your task is to analyze email content and determine whether it is SPAM or HAM (legitimate email).

## Classification Guidelines:

### SPAM indicators:
- Unsolicited commercial messages (buy now, limited offer, act fast)
- Phishing attempts (verify your account, confirm your identity)
- Nigerian prince / inheritance scams
- Lottery or prize notifications you didn't enter
- Dubious financial offers (make money fast, investment opportunities)
- Adult content or dating solicitations
- Fake urgency ("Your account will be closed!")
- Poor grammar/spelling combined with suspicious requests
- Requests for personal information (passwords, credit cards, SSN)
- Unknown senders with generic greetings

### HAM (legitimate) indicators:
- Personal correspondence from known contacts
- Business communications with context
- Newsletters you subscribed to
- Order confirmations from known retailers
- Legitimate service notifications
- Professional communications with proper signatures
- Meeting invitations or calendar items
- Normal workplace communications

## Output Format:
Provide your classification with confidence (0.0-1.0) and brief reasoning.
- High confidence (0.8-1.0): Clear spam/ham indicators present
- Medium confidence (0.5-0.8): Some indicators but not definitive
- Low confidence (0.0-0.5): Ambiguous content, could go either way

Be decisive but calibrated in your confidence scores."""


@dataclass
class SpamJudge:
    """
    LLM-as-Judge for spam classification using Gemini.

    This evaluator uses Gemini to classify emails as spam or ham,
    providing structured predictions with confidence scores.
    """

    model: str = "gemini-2.0-flash"
    temperature: float = 0.0

    def __post_init__(self):
        """Initialize the LLM after dataclass initialization."""
        self.llm = get_llm(model=self.model, temperature=self.temperature)

    def classify(self, email_text: str) -> SpamPrediction:
        """
        Classify a single email as spam or ham.

        Args:
            email_text: The email content to classify

        Returns:
            SpamPrediction with classification, confidence, and reasoning
        """
        # Truncate very long emails to avoid token limits
        max_chars = 8000
        if len(email_text) > max_chars:
            email_text = email_text[:max_chars] + "\n\n[... truncated ...]"

        user_message = f"""Classify the following email as SPAM or HAM:

---EMAIL START---
{email_text}
---EMAIL END---

Provide your classification."""

        return invoke_structured(
            self.llm,
            system_prompt=SPAM_JUDGE_PROMPT,
            user_message=user_message,
            output_schema=SpamPrediction,
        )

    def evaluate(
        self,
        email_text: str,
        ground_truth: Literal["spam", "ham"]
    ) -> dict:
        """
        Evaluate the judge's classification against ground truth.

        Args:
            email_text: The email content to classify
            ground_truth: The actual label (spam or ham)

        Returns:
            Dictionary with prediction details and correctness
        """
        prediction = self.classify(email_text)

        is_correct = prediction["classification"] == ground_truth

        return {
            "email_preview": email_text[:200] + "..." if len(email_text) > 200 else email_text,
            "ground_truth": ground_truth,
            "prediction": prediction["classification"],
            "confidence": prediction["confidence"],
            "reasoning": prediction["reasoning"],
            "is_correct": is_correct,
        }


def demo():
    """Quick demo of the spam judge."""
    judge = SpamJudge()

    # Test with a spam example
    spam_email = """
    Subject: URGENT: You've Won $1,000,000!!!

    Dear Lucky Winner,

    Congratulations! You have been selected as the winner of our international lottery!
    You have won ONE MILLION DOLLARS ($1,000,000.00 USD).

    To claim your prize, please send us your:
    - Full name
    - Bank account number
    - Social Security Number
    - Copy of your passport

    Act NOW! This offer expires in 24 hours!

    Best regards,
    International Lottery Commission
    """

    # Test with a ham example
    ham_email = """
    Subject: Team meeting tomorrow at 2pm

    Hi team,

    Just a reminder that we have our weekly standup tomorrow at 2pm in the main conference room.

    Please come prepared to share:
    - What you worked on this week
    - Any blockers you're facing
    - Plans for next week

    See you there!

    Best,
    Sarah
    """

    print("=" * 60)
    print("SPAM JUDGE DEMO")
    print("=" * 60)

    print("\n[Testing SPAM email]")
    result = judge.evaluate(spam_email, "spam")
    print(f"Ground truth: {result['ground_truth']}")
    print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.2f})")
    print(f"Correct: {result['is_correct']}")
    print(f"Reasoning: {result['reasoning']}")

    print("\n" + "-" * 60)

    print("\n[Testing HAM email]")
    result = judge.evaluate(ham_email, "ham")
    print(f"Ground truth: {result['ground_truth']}")
    print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.2f})")
    print(f"Correct: {result['is_correct']}")
    print(f"Reasoning: {result['reasoning']}")


if __name__ == "__main__":
    demo()
