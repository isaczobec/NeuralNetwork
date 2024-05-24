import numpy as np
import math

# normal cost functions

def cost(desired: np.ndarray, predicted: np.ndarray) -> float:
    return np.sum((desired - predicted)**2)

def costDerivative(desired: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return 2*(predicted - desired)

# functions for calculating regularized cost derivatives

# if the regularization terms is simply adding all the absolute values of all weights and biases together
def GetRegularizedCostDerivativeTerm(
    value: float,
    exponent: float = 1,
    regularizationMultiplier: float = 1,
) -> float:
    
    if value > 0:
        return (exponent*math.abs(value)**(exponent-1)) * regularizationMultiplier
    else:
        return -(exponent*math.abs(value)**(exponent-1)) * regularizationMultiplier

# squashing functions

def Sigmoid(x) -> float:
    return 1/(1 + math.exp(-x))

def SigmoidDerivative(x) -> float:
    return Sigmoid(x)*(1 - Sigmoid(x))

def ReLU(x) -> float:
    return max(0, x)

def ReLUDerivative(x) -> float:
    return 1 if x > 0 else 0

