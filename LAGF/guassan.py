import numpy as np 
from scipy.interpolate import interp1d
import math 
from .trace_mass_process import Trace_mass_fill
from joblib import Parallel, delayed

def func_gaosi(x, miu, sigma, a):
    """
    x : np.array ;The horizontal coordinate of the Gauss curve eg: np.linspace(0,400,401)
    miu : int ;The center of x   eg: 200
    sigma : float32
    a : float32 ;The highest value of the peak

    return : y ;A curve of length equal to x

    """
    return a * np.exp(-((x - miu) ** 2) / 2 / sigma**2)


def goodness_fitting__(y_orignal, y_fitted):
    """
    Returns R^2 as goodness of fitting.
    """
    return 1 - (
        np.sum((y_fitted - y_orignal) ** 2)
        / (np.sum((y_orignal - np.mean(y_orignal)) ** 2) + 0.0001)
    )


