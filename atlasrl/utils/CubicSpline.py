import numpy as np
import matplotlib.pyplot as plt

def cube(alpha: float, beta: float, gamma: float, delta: float, t: float):
    return alpha * t ** 3 + beta * t ** 2 + gamma * t + delta

def cubeDeriv(alpha: float, beta: float, gamma: float, delta: float, t: float):
    return 3 * alpha * t ** 2 + 2 * beta * t + gamma

def cubeDerivDeriv(alpha: float, beta: float, gamma: float, delta: float, t: float):
    return 6 * alpha * t + 2 * beta

def solveCubicInterpol(qZero: float, qZeroDot: float, qT: float, qTDot: float, t: float):
    qZero = np.array(qZero)
    qZeroDot = np.array(qZeroDot)
    qT = np.array(qT)
    qTDot = np.array(qTDot)
    alpha = (qTDot * t - 2 * qT + qZeroDot * t + 2 * qZero) / t ** 3
    beta = (3 * qT - qTDot * t - 2 * qZeroDot * t - 3 * qZero) /  t ** 2
    gamma = qZeroDot
    delta = qZero

    return (alpha, beta, gamma, delta)

def cubicInterpol(qZero: float, qZeroDot: float, qT: float, qTDot: float, t: float, currentT: float):
    return cube(*solveCubicInterpol(qZero, qZeroDot, qT, qTDot, t), currentT)

if __name__ == "__main__":
    T = np.arange(0, 0.5, 0.01)
    y = np.zeros_like(T)
    for (i, t) in enumerate(T):
        y[i] = cubicInterpol(0, 0, 0.1, 0.05, 0.5, t)
    plt.plot(T, y)
    plt.show()
