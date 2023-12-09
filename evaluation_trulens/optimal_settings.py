import csv
import math
import numpy as np

from typing import List, Tuple, Any

from numpy import ndarray

# Order: groundedness, context, answer
WEIGHTS = [2, 3, 1]


def get_answer_richness(measurements: List[float]) -> float:
    """
    Compute the answer richness metric for a given setting.

    :param measurements: The groundedness, context, and answer values
    :return: The answer richness value
    """

    assert len(WEIGHTS) == len(measurements), 'The lengths of the metrics and the weight vector must be equal.'
    total = 0.0

    for w, x in zip(WEIGHTS, measurements):
        total += w * x

    return 1.0 / (1 + math.exp(-total))


def get_optimal_chunk(file_name: str) -> tuple[float, ndarray[int], list[float]]:
    """
    Find the setting that leads to optimal richness in terms of chunking.

    :param file_name: The file containing TruLens RAG triad measurements for different settings
    :return: The richness score, optimal setting index, and the corresponding measurements
    """

    richness = []
    triads = []

    with open(file_name, 'r') as in_file:
        csv_reader = csv.reader(in_file)
        # Exclude the header
        next(csv_reader)

        for a_row in csv_reader:
            values = [float(x) for x in a_row]
            triads.append(values)
            richness.append(get_answer_richness(values))

    max_idx = np.argmax(richness)

    return richness[max_idx], max_idx, triads[max_idx]


def get_optimal_top_k(file_name: str) -> tuple[float, ndarray[int], list[float]]:
    """
    Find the setting that leads to optimal richness in terms of top-k values.

    :param file_name: The file containing TruLens RAG triad measurements for different settings
    :return: The optimal setting index and the corresponding measurements
    """

    richness = []
    triads = []

    with open(file_name, 'r') as in_file:
        csv_reader = csv.reader(in_file)
        # Exclude the header
        next(csv_reader)

        for a_row in csv_reader:
            values = [float(x) for x in a_row]
            triads.append(values)
            richness.append(get_answer_richness(values))

    max_idx = np.argmax(richness)

    return richness[max_idx], max_idx, triads[max_idx]


if __name__ == '__main__':
    rich, idx, rag_triad = get_optimal_chunk('chunk_measurements_from_dashboard.csv')
    print(f'{rich=}; optimal chunk settings obtained for index: {idx}; the corresponding RAG triad: {rag_triad}')

    rich, idx, rag_triad = get_optimal_top_k('top_k_cutoff_measurements_from_dashboard.csv')
    print(f'{rich=}; optimal top-k, cut-off setting obtained for index: {idx}; the corresponding RAG triad: {rag_triad}')
