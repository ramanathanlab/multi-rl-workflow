from random import random

from multirl.models import Sequence


def score_sequences(sequences: list[Sequence]) -> list[float]:
    """Generate a priority score for each sequence

    Args:
        sequences: List of sequences to evaluate
    Returns:
        Scores for each sequence
    """

    return [random() for _ in sequences]
