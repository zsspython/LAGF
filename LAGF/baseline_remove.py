from scipy.signal import find_peaks
import numpy as np
import math
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve

def WhittakerSmooth(x, w, lambda_, differences=1):
    X = np.matrix(x)
    m = X.size
    E = eye(m, format='csc')
    for i in range(differences):
        E = E[1:] - E[:-1]  # numpy.diff() does not work with sparse matrix. This is a workaround.
    W = diags(w, 0, shape=(m, m))
    A = csc_matrix(W + (lambda_ * E.T * E))
    B = csc_matrix(W * X.T)
    background = spsolve(A, B)
    return np.array(background)
    
def airPLS(x, lambda_=10, porder=1, itermax=30):
    m = x.shape[0]
    w = np.ones(m)
    for i in range(1, itermax + 1):
        z = WhittakerSmooth(x, w, lambda_, porder)
        d = x - z
        dssn = np.abs(d[d < 0].sum())
        if (dssn < 0.001 * (abs(x)).sum() or i == itermax):
            break
        if len(d[d < 0])!=0:
            w[d >= 0] = 0  # d>0 means that this point is part of a peak峰值, so its weight is set to 0 in order to ignore it
            w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
            w[0] = np.exp(i * (d[d < 0]).max() / dssn)
            w[-1] = w[0]
    return z

def AirPLS(data, lamb):
    data=data-airPLS(data,lamb)
    return data

def evaluate_baseline_correction(baseline_corrected_data,data):
    # 使用峰值数量作为评估指标
    peaks, _ = find_peaks(baseline_corrected_data, height=max(data)/100)
    return len(peaks)

def binary_search(data, low, high, tolerance = 0.5):
    while high - low > tolerance:
        mid = (low + high) / 2
        baseline_corrected_data = airPLS(data, mid)
        evaluation_metric = evaluate_baseline_correction(baseline_corrected_data, data)
        if evaluation_metric > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2

def optimize_lambda(data, initial_lambda = 10, max_iterations=10):
    best_lambda = initial_lambda
    lambda_values = [initial_lambda]
    
    for i in range(max_iterations):
        baseline_corrected_data = airPLS(data, lambda_values[-1])
        evaluation_metric = evaluate_baseline_correction(baseline_corrected_data, data)
        best_lambda = lambda_values[-1]
        
        # 根据评估指标更新lambda
        if evaluation_metric > 0:  # 假设评估指标大于0.5时需要增加lambda
            lambda_values.append(lambda_values[-1] * 1.5)  # 增加10%
        elif evaluation_metric ==0 and len(lambda_values) == 1:
            lambda_values.insert(0,1)
            break
        else:
            break
    best_lambda=binary_search(data,lambda_values[-2],lambda_values[-1])
    
    return best_lambda