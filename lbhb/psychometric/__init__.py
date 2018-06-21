import itertools

import pystan
import pickle
from hashlib import md5

import numpy as np
from scipy.stats import norm
import pandas as pd
import pylab as pl
import seaborn as sns


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


def compute_threshold(x, a, b, g, l, threshold, clip):
    x_i = x.ravel()
    trace = compute_dprime(x, a, b, g, l, clip)
    interp = lambda y: np.interp(threshold, y, x_i, left=-np.inf, right=np.inf)
    return np.apply_along_axis(interp, 0, trace)


def dprime(p):
    n = len(p)
    clip = 0.5/(n+1)
    p_clipped = np.clip(p, clip, 1-clip)
    return norm.ppf(p_clipped) - norm.ppf(p_clipped[0])


def get_traces(fit, variables=None, levels=None):
    if variables is None:
        variables = []
    if levels is None:
        levels = []

    traces = fit.extract()
    for var in variables:
        for level in levels:
            loc = traces[f'{var}_loc'][..., np.newaxis]
            scale = traces[f'{var}_{level}_scale'][..., np.newaxis]
            delta = traces[f'{var}_{level}_delta']
            traces[f'{var}_{level}'] = loc + delta * scale

    return traces


def _get_trace(x, traces, level, fn, *args, **kw):
    a = traces[f'a_{level}']
    gt = traces[f'gt_{level}']
    g = inverse_logit(np.exp(gt))
    b = traces['b'][..., np.newaxis]
    l = traces['l'][..., np.newaxis]
    x_fit = x[..., np.newaxis, np.newaxis]
    return fn(x_fit, a, b, g, l, *args, **kw)


def get_psi_trace(x, traces, level):
    return _get_trace(x, traces, level, compute_psi)


def get_dprime_trace(x, traces, level, clip=0.05):
    return _get_trace(x, traces, level, compute_dprime, clip)


def get_threshold_trace(x, traces, level, threshold=1, clip=0.05, fillna=False):
    th_trace = _get_trace(x, traces, level, compute_threshold, threshold, clip)
    if fillna:
        m = ~np.isfinite(th_trace)
        th_trace[m] = np.random.choice(x.ravel(), m.sum())
    return th_trace


def get_subplot_iter(n_labels):
    n = int(np.ceil(np.sqrt(n_labels)))
    f, axes = pl.subplots(n, n, sharex=True, sharey=True, figsize=(2*n, 2*n))
    ax_iter = itertools.chain(*axes)
    return f, axes, ax_iter


def plot_level_ci(x, trace, labels, raw_data=None):
    lb, ub = np.percentile(trace, [2.5, 97.5], axis=1)
    mean = np.mean(trace, axis=1)

    n_labels = len(labels)
    f, axes, ax_iter = get_subplot_iter(n_labels)

    for ax, (i, l) in zip(ax_iter, enumerate(labels)):
        p, = ax.plot(x, mean[:, i], label=f'{l}')
        ax.fill_between(x, lb[:, i], ub[:, i],
                        facecolor='0.75', alpha=0.5, label='95% CI')
        if raw_data is not None:
            print(raw_data.shape, i)
            d = raw_data.iloc[:, i].dropna()
            if len(d) >= 2:
                d_x = d.index.astype('f')
                ax.plot(d_x, dprime(d.values), 'ko')

        ax.set_title(f'{l}')
        #ax.set_title(f'Masker {c} dB SPL')

    for ax in axes[:, 0]:
        ax.set_ylabel("$d'$")
    for ax in axes[-1, :]:
        ax.set_xlabel('AM depth')
    pl.tight_layout()


def plot_level(x, trace, labels):
    mean = np.mean(trace, axis=1)
    n_labels = len(labels)
    f, ax = pl.subplots(1, 1, figsize=(6, 4))
    colors = pl.cm.viridis(np.linspace(0, 1, n_labels))
    ax.set_prop_cycle(color=colors)
    for i, l in enumerate(labels):
        p, = ax.plot(x, mean[:, i], label=f'{l}')
    ax.legend(loc='upper left')
    ax.set_xlabel('AM depth')
    ax.set_ylabel("$d'$")


def plot_threshold(th_trace, labels):
    f, ax = pl.subplots(1, 1, figsize=(6, 4))
    lb, ub = np.percentile(th_trace, [2.5, 97.5], axis=0)
    mean = np.mean(th_trace, axis=0)
    yerr = np.vstack((mean-lb, ub-mean))
    i = np.arange(lb.shape[-1])
    ax.errorbar(i, mean, yerr, fmt='ko-')
    ax.xaxis.set_ticks(i)
    ax.xaxis.set_ticklabels(f'{l}' for l in labels)
    ax.axis(ymin=0, ymax=1)
    ax.set_ylabel('AM depth')


def plot_threshold_posterior(th_trace, labels):
    n_labels = len(labels)
    f, axes, ax_iter = get_subplot_iter(n_labels)

    for ax, (i, l) in zip(ax_iter, enumerate(labels)):
        t = th_trace[:, i]
        sns.kdeplot(t, ax=ax, shade=True)
        lb, ub = np.percentile(t, [2.5, 97.5])
        mean = np.mean(t)
        ax.axvline(mean, c='k')
        ax.axvline(lb, c='k', ls=':')
        ax.axvline(ub, c='k', ls=':')

    ax.axis(xmin=0, xmax=1)
    for ax in axes[:, 0]:
        ax.set_ylabel("Estimated PDF")
    for ax in axes[-1, :]:
        ax.set_xlabel('AM depth')
    pl.tight_layout()


def summarize_level(x, traces, level, labels=None, raw_data=None,
                    data_level=None):
    if raw_data is not None:
        d = raw_data.groupby([data_level, 'depth'])[['size', 'sum']].sum()
        d['p'] = d['sum']/d['size']
        d = d['p'].unstack(data_level)
    else:
        d = None

    x_log = np.log(x)
    trace = get_dprime_trace(x_log, traces, level)
    th_log_trace = get_threshold_trace(x_log, traces, level, fillna=True)
    th_trace = np.exp(th_log_trace)

    if labels is None:
        labels = np.arange(trace.shape[-1])

    plot_level_ci(x, trace, labels, d)
    plot_level(x, trace, labels)
    plot_threshold(th_trace, labels)
    plot_threshold_posterior(th_trace, labels)


def summarize_chains(fit):
    fit_summary = fit.summary()
    rows = fit_summary['c_summary_rownames']
    cols = fit_summary['c_summary_colnames']
    chain_summary = []
    for c in np.rollaxis(fit_summary['c_summary'], 2):
        df = pd.DataFrame(c, index=rows, columns=cols)
        chain_summary.append(df)
    return pd.concat(chain_summary, keys=range(len(df)), names=['chain'])


def summarize(fit):
    fit_summary = fit.summary()
    rows = fit_summary['summary_rownames']
    cols = fit_summary['summary_colnames']
    return pd.DataFrame(fit_summary['summary'], index=rows, columns=cols)


def CachedStanModel(model_file, model_name=None, **kwargs):
    """Use just as you would `stan`"""
    with open(model_file, 'rb') as fh:
        model_code = fh.read().decode('utf8')
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    if model_name is None:
        cache_fn = 'cached-model-{}.pkl'.format(code_hash)
    else:
        cache_fn = 'cached-{}-{}.pkl'.format(model_name, code_hash)
    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except:
        sm = pystan.StanModel(model_code=model_code)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    return sm
