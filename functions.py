import numpy as np

def cost(desired: np.ndarray, predicted: np.ndarray) -> float:
    return np.sum((desired - predicted)**2)