"""
Dataset Loader for Spam Classification Evaluation

This module provides utilities for loading and processing the Kaggle
spam mails dataset for evaluation.

Dataset source: https://www.kaggle.com/datasets/venky73/spam-mails-dataset
"""

import os
import csv
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator, Literal


@dataclass
class EmailSample:
    """A single email sample with its label."""
    text: str
    label: Literal["spam", "ham"]
    index: int = 0


@dataclass
class SpamDatasetLoader:
    """
    Loader for the Kaggle spam mails dataset.

    The dataset is expected to be a CSV file with columns:
    - 'text' or 'Message' or similar: the email content
    - 'label' or 'Category' or similar: spam/ham classification

    Usage:
        loader = SpamDatasetLoader(csv_path="path/to/spam.csv")
        for sample in loader.iterate(limit=100):
            print(sample.label, sample.text[:50])
    """

    csv_path: str
    text_column: str = None  # Auto-detected if None
    label_column: str = None  # Auto-detected if None
    _samples: list = field(default_factory=list, repr=False)
    _loaded: bool = field(default=False, repr=False)

    def _detect_columns(self, headers: list[str]) -> tuple[str, str]:
        """Auto-detect text and label column names."""
        # Common text column names
        text_candidates = ['text', 'message', 'email', 'content', 'body', 'mail']
        # Common label column names
        label_candidates = ['label', 'category', 'class', 'spam', 'type', 'target']

        text_col = None
        label_col = None

        headers_lower = [h.lower().strip() for h in headers]

        for candidate in text_candidates:
            for i, h in enumerate(headers_lower):
                if candidate in h:
                    text_col = headers[i]
                    break
            if text_col:
                break

        for candidate in label_candidates:
            for i, h in enumerate(headers_lower):
                if candidate in h:
                    label_col = headers[i]
                    break
            if label_col:
                break

        # Fallback: assume first column is label, second is text
        if not text_col and len(headers) >= 2:
            text_col = headers[1]
        if not label_col and len(headers) >= 1:
            label_col = headers[0]

        return text_col, label_col

    def _normalize_label(self, label: str) -> Literal["spam", "ham"]:
        """Normalize label to 'spam' or 'ham'."""
        label_lower = label.lower().strip()
        if label_lower in ['spam', '1', 'true', 'yes']:
            return "spam"
        return "ham"

    def load(self) -> "SpamDatasetLoader":
        """
        Load the dataset from CSV.

        Returns:
            self for method chaining
        """
        if self._loaded:
            return self

        path = Path(self.csv_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.csv_path}\n"
                f"Please download from: https://www.kaggle.com/datasets/venky73/spam-mails-dataset\n"
                f"And place the CSV file at the specified path."
            )

        self._samples = []

        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            # Try to detect delimiter
            sample = f.read(4096)
            f.seek(0)

            # Detect delimiter
            if '\t' in sample and sample.count('\t') > sample.count(','):
                delimiter = '\t'
            else:
                delimiter = ','

            reader = csv.DictReader(f, delimiter=delimiter)
            headers = reader.fieldnames

            if not headers:
                raise ValueError("CSV file appears to be empty or malformed")

            # Detect or use specified columns
            text_col = self.text_column
            label_col = self.label_column

            if not text_col or not label_col:
                detected_text, detected_label = self._detect_columns(headers)
                text_col = text_col or detected_text
                label_col = label_col or detected_label

            print(f"Using columns: text='{text_col}', label='{label_col}'")
            print(f"Available columns: {headers}")

            for idx, row in enumerate(reader):
                text = row.get(text_col, "")
                label_raw = row.get(label_col, "ham")

                if not text or not text.strip():
                    continue

                self._samples.append(EmailSample(
                    text=text.strip(),
                    label=self._normalize_label(label_raw),
                    index=idx,
                ))

        self._loaded = True
        print(f"Loaded {len(self._samples)} samples from {self.csv_path}")

        # Print label distribution
        spam_count = sum(1 for s in self._samples if s.label == "spam")
        ham_count = len(self._samples) - spam_count
        print(f"Distribution: {spam_count} spam, {ham_count} ham")

        return self

    def iterate(
        self,
        limit: int = None,
        shuffle: bool = True,
        balanced: bool = True,
        seed: int = 42,
    ) -> Iterator[EmailSample]:
        """
        Iterate over dataset samples.

        Args:
            limit: Maximum number of samples to return
            shuffle: Whether to shuffle the samples
            balanced: Whether to balance spam/ham classes
            seed: Random seed for reproducibility

        Yields:
            EmailSample objects
        """
        if not self._loaded:
            self.load()

        samples = self._samples.copy()

        if balanced:
            # Separate by class
            spam_samples = [s for s in samples if s.label == "spam"]
            ham_samples = [s for s in samples if s.label == "ham"]

            # Balance classes
            min_count = min(len(spam_samples), len(ham_samples))

            rng = random.Random(seed)
            rng.shuffle(spam_samples)
            rng.shuffle(ham_samples)

            spam_samples = spam_samples[:min_count]
            ham_samples = ham_samples[:min_count]

            samples = spam_samples + ham_samples

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(samples)

        if limit:
            samples = samples[:limit]

        yield from samples

    def get_sample(self, index: int) -> EmailSample:
        """Get a specific sample by index."""
        if not self._loaded:
            self.load()
        return self._samples[index]

    def __len__(self) -> int:
        """Return total number of samples."""
        if not self._loaded:
            self.load()
        return len(self._samples)

    @property
    def stats(self) -> dict:
        """Return dataset statistics."""
        if not self._loaded:
            self.load()

        spam_count = sum(1 for s in self._samples if s.label == "spam")
        ham_count = len(self._samples) - spam_count

        return {
            "total": len(self._samples),
            "spam": spam_count,
            "ham": ham_count,
            "spam_ratio": spam_count / len(self._samples) if self._samples else 0,
        }


def demo():
    """Demo the dataset loader with a sample file."""
    print("SpamDatasetLoader Demo")
    print("=" * 50)
    print("\nTo use the loader, download the dataset from:")
    print("https://www.kaggle.com/datasets/venky73/spam-mails-dataset")
    print("\nThen initialize the loader with the CSV path:")
    print("  loader = SpamDatasetLoader(csv_path='path/to/spam.csv')")
    print("  loader.load()")
    print("  for sample in loader.iterate(limit=10):")
    print("      print(sample.label, sample.text[:50])")


if __name__ == "__main__":
    demo()
