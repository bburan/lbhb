import numpy as np


def _global_gain_spec(n_targets):
    gain_mean = np.zeros(n_targets)
    gain_sd = np.ones(n_targets)
    template = {
        'fn': 'lbhb.analysis.rdt.modules.global_gain',
        'fn_kwargs': {},
        'prior': {
            'gain': ('Normal', {'mean': gain_mean, 'sd': gain_sd}),
        }
    }
    return template


def _relative_gain_spec(n_targets):
    gain_mean = np.zeros(n_targets)
    gain_sd = np.ones(n_targets)
    template = {
        'fn': 'lbhb.analysis.rdt.modules.relative_gain',
        'fn_kwargs': {},
        'prior': {
            'fg_gain': ('Normal', {'mean': gain_mean, 'sd': gain_sd}),
            'bg_gain': ('Normal', {'mean': gain_mean, 'sd': gain_sd}),
        }
    }
    return template


def rdtgain(kw):
    _, mode, n_targets = kw.split('.')
    n_targets = int(n_targets)
    if mode == 'global':
        return _global_gain_spec(n_targets)
    elif mode == 'relative':
        return _relative_gain_spec(n_targets)



def rdtwc(kw):
    from nems.plugins import default_keywords
    kw = kw[3:]
    ms = default_keywords.wc(kw)
    del ms['fn_kwargs']['i']
    del ms['fn_kwargs']['o']
    ms['fn'] = 'lbhb.analysis.rdt.weight_channels.gaussian'
    return ms


def rdtfir(kw):
    from nems.plugins import default_keywords
    kw = kw[3:]
    ms = default_keywords.fir(kw)
    del ms['fn_kwargs']['i']
    del ms['fn_kwargs']['o']
    ms['fn'] = 'lbhb.analysis.rdt.fir.basic'
    return ms
