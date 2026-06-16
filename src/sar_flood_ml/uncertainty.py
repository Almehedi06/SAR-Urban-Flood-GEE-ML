from __future__ import annotations

import numpy as np


def vote_entropy(probabilities: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Entropy helper for future bootstrap/model-ensemble uncertainty maps."""
    probs = np.clip(probabilities, eps, 1.0)
    return -np.sum(probs * np.log(probs), axis=0)
