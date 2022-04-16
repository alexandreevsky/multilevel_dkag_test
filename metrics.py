from typing import List, Tuple

import numpy as np
import torch


def precision_at_k(hits: np.ndarray, k: int) -> np.ndarray:
    return hits[:, :k].mean(axis=1)


def recall_at_k(hits: np.ndarray, k: int) -> np.ndarray:
    return hits[:, :k].sum(axis=1) / hits.sum(axis=1)


def dcg_at_k(hits: np.ndarray, k) -> np.ndarray:
    hits_k = hits[:, :k]
    return np.sum((2 ** hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)


def ndcg_at_k(hits: np.ndarray, k: int) -> np.ndarray:

    dcg = dcg_at_k(hits, k)
    sorted_hits = np.flip(np.sort(hits))

    idcg = dcg_at_k(sorted_hits, k)
    idcg[idcg == 0] = np.inf

    return dcg / idcg


def calculate_metrics_at_k(k: int,
                           scores: torch.Tensor,
                           interactions_list: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    interactions_matrix = np.zeros((scores.size(0), scores.size(1)))
    for idx, interactions in enumerate(interactions_list):
        interactions_matrix[idx, interactions] = 1

    _, sorted_items = torch.sort(scores, descending=True)
    sorted_items = sorted_items.numpy()

    binary_hits = np.zeros_like(interactions_matrix)
    for idx, items in enumerate(sorted_items):
        binary_hits[idx, :] = interactions_matrix[idx, items]

    precision = precision_at_k(binary_hits, k)
    recall = recall_at_k(binary_hits, k)
    ndcg = ndcg_at_k(binary_hits, k)

    return precision, recall, ndcg
