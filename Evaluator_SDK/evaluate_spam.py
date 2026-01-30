"""
Spam Evaluation Script using LangSmith SDK

This script runs the full evaluation pipeline using LangSmith:
1. Load the spam dataset from Kaggle
2. Create a LangSmith dataset
3. Use Gemini as LLM-as-judge to classify emails
4. Run evaluation with LangSmith's evaluate() function

Usage:
    python -m Evaluator_SDK.evaluate_spam --csv data/spam.csv --limit 100

Or from Python:
    from Evaluator_SDK import run_spam_evaluation
    results = run_spam_evaluation(csv_path="data/spam.csv", limit=100)
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langsmith import Client, evaluate
from langsmith.schemas import Example, Run
from langsmith.evaluation import EvaluationResult

from Evaluator_SDK.spam_judge import SpamJudge, SpamPrediction
from Evaluator_SDK.dataset_loader import SpamDatasetLoader


def create_langsmith_dataset(
    client: Client,
    loader: SpamDatasetLoader,
    dataset_name: str,
    limit: int = 50,
    balanced: bool = True,
) -> str:
    """
    Create or update a LangSmith dataset from the spam data.

    Args:
        client: LangSmith client
        loader: Spam dataset loader
        dataset_name: Name for the LangSmith dataset
        limit: Maximum samples to include
        balanced: Whether to balance spam/ham classes

    Returns:
        Dataset ID
    """
    # Check if dataset exists
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"Using existing dataset: {dataset_name}")
        return dataset.id
    except Exception:
        pass

    # Create new dataset
    print(f"Creating new dataset: {dataset_name}")
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Spam/Ham email classification dataset from Kaggle"
    )

    # Add examples
    examples = []
    for sample in loader.iterate(limit=limit, balanced=balanced):
        examples.append({
            "inputs": {"email_text": sample.text},
            "outputs": {"label": sample.label},
        })

    client.create_examples(
        inputs=[ex["inputs"] for ex in examples],
        outputs=[ex["outputs"] for ex in examples],
        dataset_id=dataset.id,
    )

    print(f"Created dataset with {len(examples)} examples")
    return dataset.id


def spam_classifier_target(inputs: dict) -> dict:
    """
    Target function that classifies emails using Gemini.

    This is the function being evaluated by LangSmith.

    Args:
        inputs: Dict with 'email_text' key

    Returns:
        Dict with prediction details
    """
    judge = SpamJudge(model="gemini-2.0-flash")
    prediction = judge.classify(inputs["email_text"])

    return {
        "prediction": prediction["classification"],
        "confidence": prediction["confidence"],
        "reasoning": prediction["reasoning"],
    }


def correctness_evaluator(run: Run, example: Example) -> EvaluationResult:
    """
    Evaluator that checks if prediction matches ground truth.

    Args:
        run: The LangSmith run containing the prediction
        example: The dataset example with ground truth

    Returns:
        Evaluation result with score and feedback
    """
    predicted = run.outputs.get("prediction", "") if run.outputs else ""
    expected = example.outputs.get("label", "") if example.outputs else ""

    is_correct = predicted == expected

    return EvaluationResult(
        key="correctness",
        score=1.0 if is_correct else 0.0,
        comment=f"Predicted: {predicted}, Expected: {expected}"
    )


def spam_detection_evaluator(run: Run, example: Example) -> EvaluationResult:
    """
    Evaluator for spam detection (true positive rate).

    Args:
        run: The LangSmith run containing the prediction
        example: The dataset example with ground truth

    Returns:
        Evaluation result
    """
    predicted = run.outputs.get("prediction", "") if run.outputs else ""
    expected = example.outputs.get("label", "") if example.outputs else ""

    # Only evaluate spam detection for actual spam emails
    if expected != "spam":
        return EvaluationResult(key="spam_detected", score=None)

    is_spam_detected = predicted == "spam"

    return EvaluationResult(
        key="spam_detected",
        score=1.0 if is_spam_detected else 0.0,
        comment="Spam correctly identified" if is_spam_detected else "Spam missed (false negative)"
    )


def ham_preservation_evaluator(run: Run, example: Example) -> EvaluationResult:
    """
    Evaluator for ham preservation (true negative rate).

    Args:
        run: The LangSmith run containing the prediction
        example: The dataset example with ground truth

    Returns:
        Evaluation result
    """
    predicted = run.outputs.get("prediction", "") if run.outputs else ""
    expected = example.outputs.get("label", "") if example.outputs else ""

    # Only evaluate ham preservation for actual ham emails
    if expected != "ham":
        return EvaluationResult(key="ham_preserved", score=None)

    is_ham_preserved = predicted == "ham"

    return EvaluationResult(
        key="ham_preserved",
        score=1.0 if is_ham_preserved else 0.0,
        comment="Ham correctly identified" if is_ham_preserved else "Ham marked as spam (false positive)"
    )


def confidence_calibration_evaluator(run: Run, example: Example) -> EvaluationResult:
    """
    Evaluator for confidence calibration.

    High confidence on correct predictions and low confidence on incorrect
    predictions indicates well-calibrated confidence scores.

    Args:
        run: The LangSmith run containing the prediction
        example: The dataset example with ground truth

    Returns:
        Evaluation result
    """
    predicted = run.outputs.get("prediction", "") if run.outputs else ""
    expected = example.outputs.get("label", "") if example.outputs else ""
    confidence = run.outputs.get("confidence", 0.5) if run.outputs else 0.5

    is_correct = predicted == expected

    # Good calibration: high confidence when correct, low when incorrect
    if is_correct:
        # Reward high confidence on correct predictions
        score = confidence
    else:
        # Reward low confidence on incorrect predictions
        score = 1.0 - confidence

    return EvaluationResult(
        key="confidence_calibration",
        score=score,
        comment=f"Confidence: {confidence:.2f}, Correct: {is_correct}"
    )


def run_spam_evaluation(
    csv_path: str,
    limit: int = 50,
    model: str = "gemini-2.0-flash",
    balanced: bool = True,
    experiment_prefix: str = "spam-eval",
    dataset_name: str = None,
    verbose: bool = True,
):
    """
    Run spam evaluation using LangSmith SDK.

    Args:
        csv_path: Path to the spam dataset CSV
        limit: Maximum number of samples to evaluate
        model: Gemini model to use
        balanced: Whether to balance spam/ham classes
        experiment_prefix: Prefix for the LangSmith experiment
        dataset_name: Name for the LangSmith dataset (auto-generated if None)
        verbose: Whether to print progress

    Returns:
        LangSmith evaluation results
    """
    if verbose:
        print("=" * 60)
        print("SPAM EVALUATION - LangSmith + Gemini LLM-as-Judge")
        print("=" * 60)
        print(f"\nModel: {model}")
        print(f"Dataset: {csv_path}")
        print(f"Sample limit: {limit}")
        print(f"Balanced: {balanced}")

    # Initialize LangSmith client
    client = Client()

    # Load the spam dataset
    loader = SpamDatasetLoader(csv_path=csv_path)
    try:
        loader.load()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return None

    if verbose:
        stats = loader.stats
        print(f"\nDataset stats:")
        print(f"  Total samples: {stats['total']}")
        print(f"  Spam: {stats['spam']} ({stats['spam_ratio']:.1%})")
        print(f"  Ham: {stats['ham']} ({1 - stats['spam_ratio']:.1%})")

    # Create LangSmith dataset
    if dataset_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = f"spam-classification-{timestamp}"

    dataset_id = create_langsmith_dataset(
        client=client,
        loader=loader,
        dataset_name=dataset_name,
        limit=limit,
        balanced=balanced,
    )

    if verbose:
        print(f"\nRunning evaluation with LangSmith...")
        print(f"Experiment prefix: {experiment_prefix}")
        print("-" * 60)

    # Create the target function with the specified model
    def target_fn(inputs: dict) -> dict:
        judge = SpamJudge(model=model)
        prediction = judge.classify(inputs["email_text"])
        return {
            "prediction": prediction["classification"],
            "confidence": prediction["confidence"],
            "reasoning": prediction["reasoning"],
        }

    # Run evaluation using LangSmith
    results = evaluate(
        target_fn,
        data=dataset_name,
        evaluators=[
            correctness_evaluator,
            spam_detection_evaluator,
            ham_preservation_evaluator,
            confidence_calibration_evaluator,
        ],
        experiment_prefix=experiment_prefix,
        max_concurrency=4,
    )

    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print("\nView detailed results in LangSmith dashboard:")
        print("https://smith.langchain.com")

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate spam classification using LangSmith + Gemini LLM-as-judge"
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the spam dataset CSV file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of samples to evaluate (default: 50)"
    )
    parser.add_argument(
        "--model",
        default="gemini-2.0-flash",
        help="Gemini model to use (default: gemini-2.0-flash)"
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Don't balance spam/ham classes"
    )
    parser.add_argument(
        "--experiment-prefix",
        default="spam-eval",
        help="Prefix for LangSmith experiment name"
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Name for LangSmith dataset (auto-generated if not specified)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    run_spam_evaluation(
        csv_path=args.csv,
        limit=args.limit,
        model=args.model,
        balanced=not args.no_balance,
        experiment_prefix=args.experiment_prefix,
        dataset_name=args.dataset_name,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
