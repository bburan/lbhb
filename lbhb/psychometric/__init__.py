import numpy as np
from scipy.stats import norm
import pandas as pd
import pylab as pl


def _get_trace_values(trace, effect):
    a = trace.get_values('α {}'.format(effect))
    b = trace.get_values('β {}'.format(effect))
    l = trace.get_values('λ {}'.format(effect))
    g = trace.get_values('γ {}'.format(effect))
    return a, b, l, g


def trace_summary(trace, effect, effects):
    traces = _get_trace_values(trace, effect)
    names = 'alpha', 'beta', 'lambda', 'gamma'
    effect_info = effects[effect]
    results = []
    for n, t in zip(names, traces):
        ci = mc.hpd(t)
        result = pd.DataFrame(ci,
                              index=effect_info['labels'],
                              columns=['ci_lb', 'ci_ub'])
        result['mean'] = np.mean(t, axis=0)
        results.append(result)
    return pd.concat(results, keys=names, names=['coefficient', effect])


def logit(x):
    return x/(1+x)


def inverse_logit(y):
    return -y/(y-1)


def glogit(x, y):
    return x/(1+x+y)


def compute_psi(x, a, b, g, l):
    return g+(1-g-l)/(1+np.exp(-(x-a)/b))


def compute_dprime(x, a, b, g, l, clip):
    psi = compute_psi(x, a, b, g, l)
    psi_clipped = np.clip(psi, clip, 1-clip)
    fa_clipped = np.clip(g, clip, 1-clip)
    return norm.ppf(psi_clipped) - norm.ppf(fa_clipped)


def compute_threshold(threshold, x, a, b, g, l, clip):
    dprime = compute_dprime(x, a, b, g, l, clip)
    interp = lambda y: np.interp([threshold], y, x.ravel(), left=-np.inf, right=np.inf)[0]
    if dprime.ndim == 1:
        return interp(dprime)
    elif dprime.ndim == 2:
        return np.apply_along_axis(interp, 0, dprime)
    else:
        raise ValueError('too many dimensions')


def dprime(x, k, n):
    p = k/n
    clip = 0.5/(n+1)
    p_clipped = np.clip(p, clip, 1-clip)
    return norm.ppf(p_clipped) - norm.ppf(p_clipped.iloc[0])


def compute_psi_ci(x_fit, trace, slice=Ellipsis):
    a, b, l, g = _get_trace_values(trace, slice)
    p_trace = compute_psi(x_fit[..., np.newaxis], a, b, g, l)
    ci_lower, ci_upper = np.percentile(p_trace, [2.5, 97.5], axis=1)
    return ci_lower, ci_upper


def compute_dprime_trace(x_fit, trace, clip, slice=Ellipsis):
    a, b, l, g = _get_trace_values(trace, slice)
    return compute_dprime(x_fit[..., np.newaxis], a, b, g, l, clip)


def compute_dprime_ci(x_fit, trace, clip, slice=Ellipsis):
    p_trace = compute_dprime_trace(x_fit, trace, clip, slice)
    ci_lower, ci_upper = np.percentile(p_trace, [2.5, 97.5], axis=1)
    return ci_lower, ci_upper


def compute_threshold_ci(threshold, x_fit, trace, clip, slice=Ellipsis):
    a, b, l, g = _get_trace_values(trace, slice)
    threshold = compute_threshold(threshold, x_fit[..., np.newaxis], a, b, g, l, clip)
    return np.percentile(threshold, [2.5, 97.5])


def plot_dprime(trace, data, effects):
    x_fit = np.log(np.arange(0.01, 1.05, 0.01))

    for i_row, row in enumerate(row_values):
        for i_col, col in enumerate(col_values):
            pass


    f, axes = pl.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows),
                          sharex=True, sharey=True)

    for i_row, row in enumerate(row_values):
        for i_col, col in enumerate(col_values):
            ax = axes[i_row, i_col]
            mask = (data[rows] == row) & (data[cols] == col)
            d = data.loc[mask]
            base_slice[row_ax] = r_map[row]
            base_slice[col_ax] = c_map[col]

            summary = trace_summary(trace, base_slice)
            a = summary.loc['alpha', 'mean']
            b = summary.loc['beta', 'mean']
            g = summary.loc['gamma', 'mean']
            l = summary.loc['lambda', 'mean']

            x = d['log_depth'].values
            n = d['size'].values
            k = d['sum'].values

            clip = max(0.5/(n+1))

            traces = compute_dprime_trace(x_fit, trace, clip, base_slice)
            samples = traces.shape[-1]
            skip = int(samples / 100)
            ax.plot(x_fit, traces[:, ::skip], 'k-', alpha=0.01)
            lb, ub = compute_dprime_ci(x_fit, trace, clip, base_slice)
            ax.fill_between(x_fit, lb, ub, edgecolor=None, facecolor='0.75')

            th = compute_threshold(1, x_fit, a, b, g, l, clip)
            th_lb, th_ub = compute_threshold_ci(1, x_fit, trace, clip, base_slice)
            ax.errorbar([th], [0], xerr=[[th_lb-th, th_ub-th]], fmt='ko')

            fitted_dprime = compute_dprime(x_fit, a, b, g, l, clip)
            ax.plot(x_fit, fitted_dprime, 'k-')

            d = dprime(x, k, n)
            ax.scatter(x, d, np.sqrt(n))

            ax.set_xlabel('SAM depth')
            ax.set_ylabel('d prime')
            ax.set_title('{} {}'.format(row, col))
