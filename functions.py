import numpy as np
import math

def cost(desired: np.ndarray, predicted: np.ndarray) -> float:
    return np.sum((desired - predicted)**2)

def costDerivative(desired: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return 2*(predicted - desired)

def Sigmoid(x) -> float:
    return 1/(1 + math.exp(-x))

def SigmoidDerivative(x) -> float:
    return Sigmoid(x)*(1 - Sigmoid(x))