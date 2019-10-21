import numpy as np


def trimmed_reshape(x, size):
    n = x.shape[0]
    s = n // size
    o = n % size
    r = n - o
    y = x[:r].reshape((size, s, -1))
    return y


def _all(fn, x, n=1, diagonal=None, mode='time'):
    '''
    Returns all pairwise correlations
    '''
    if mode == 'psd':
        x = np.fft.rfft(x)
        x = np.abs(x)

    if mode == 'csd':
        x = np.fft.rfft(x)

    if n != 1:
        j = len(x)//n
        x = x.iloc[:j*n].values.reshape((n, j, -1))
        x = x.mean(axis=0)

    cc = fn(x)
    if diagonal is None:
        i, j = np.tril_indices_from(cc, k=-1)
        result = cc[i, j]
    else:
        result = np.diagonal(cc, diagonal)

    if mode == 'csd':
        result = np.abs(result)

    return result


def all_cc(x, n=1, diagonal=None, mode='time'):
    '''
    Returns all pairwise correlations
    '''
    return _all(np.corrcoef, x, n, diagonal, mode)


def all_cov(x, n=1, diagonal=None, mode='time'):
    '''
    Returns all pairwise correlations
    '''
    return _all(np.cov, x, n, diagonal, mode)
