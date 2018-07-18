from copy import deepcopy

import numpy as np

from nems.keywords import defaults as kw_registry

def dual_lvl(rec, level):
    fn = lambda x: x + level
    return [
        rec['bg_pred'].transform(fn, 'bg_pred'),
        rec['fg_pred'].transform(fn, 'fg_pred'),
    ]


def single_stream_merge_sf(rec, gain):
    is_repeating = rec['repeating'].as_continuous()[0].astype('bool')
    t_map = rec['target_id_map'].as_continuous()[0].astype('i')

    gain_mapped = gain[t_map]
    gain = np.zeros(t_map.shape, dtype=np.double)

    # Set gains for repeating portions of ss and ds
    gain[is_repeating] = gain_mapped[is_repeating]

    # Convert to scaling factor
    sf = np.exp(gain)

    return sf


def single_stream_merge(rec, gain):
    sf = single_stream_merge_sf(rec, gain)
    bg = rec['bg_pred'].as_continuous()
    fg = rec['fg_pred'].as_continuous()
    pred = (bg + fg) * sf
    pred_signal = rec['bg_pred']._modified_copy(pred, name='pred')
    return [pred_signal]


def create_single_stream_merge(n_targets):
    name = 'rdt:mss{}'.format(n_targets)
    gain_mean = np.zeros(n_targets)
    gain_sd = np.ones(n_targets)
    template = {
        'id': 'RDTglobalgain',
        'fn': 'lbhb.analysis.rdt.modules.single_stream_merge',
        'prior': {
            'gain': ('Normal', {'mean': gain_mean, 'sd': gain_sd}),
        }
    }
    return template


def stream_merge_sf(rec, fg_gain, bg_gain):
    is_repeating = rec['repeating'].as_continuous()[0].astype('bool')
    t_map = rec['target_id_map'].as_continuous()[0].astype('i')

    fg_gain_mapped = fg_gain[t_map]
    bg_gain_mapped = bg_gain[t_map]
    fg_gain = np.zeros(t_map.shape, dtype=np.double)
    bg_gain = np.zeros(t_map.shape, dtype=np.double)

    # Set gains for repeating portions of ss and ds
    fg_gain[is_repeating] = fg_gain_mapped[is_repeating]
    bg_gain[is_repeating] = bg_gain_mapped[is_repeating]

    # Convert to scaling factor
    bg_sf = np.exp(bg_gain)
    fg_sf = np.exp(fg_gain)

    return fg_sf, bg_sf


def stream_merge(rec, fg_gain, bg_gain):
    fg_sf, bg_sf = stream_merge_sf(rec, fg_gain, bg_gain)
    bg = rec['bg_pred'].as_continuous()
    fg = rec['fg_pred'].as_continuous()

    pred = bg * bg_sf + fg * fg_sf
    pred_signal = rec['bg_pred']._modified_copy(pred, name='pred')
    return [pred_signal]


def create_stream_merge(n_targets):
    name = 'rdt:ms{}'.format(n_targets)
    gain_mean = np.zeros(n_targets)
    gain_sd = np.ones(n_targets)
    template = {
        'id': 'RDTstreamgain',
        'fn': 'lbhb.analysis.rdt.modules.stream_merge',
        'prior': {
            'fg_gain': ('Normal', {'mean': gain_mean, 'sd': gain_sd}),
            'bg_gain': ('Normal', {'mean': gain_mean, 'sd': gain_sd}),
        }
    }
    return template


def create_modelspec(recording, wcg_n, fir_n, dexp, mode='dual'):
    modelspec = []

    wc = deepcopy(kw_registry[f'wcg18x{wcg_n}'])
    wc['id'] = f'RDTwcg18x{wcg_n}'
    wc['fn'] = 'lbhb.analysis.rdt.weight_channels.gaussian'
    wc['fn_kwargs'].pop('i', None)
    wc['fn_kwargs'].pop('o', None)
    modelspec.append(wc)

    fir = deepcopy(kw_registry[f'fir{wcg_n}x{fir_n}'])
    fir['id'] = f'RDTfir{wcg_n}x{fir_n}'
    fir['fn'] = 'lbhb.analysis.rdt.fir.basic'
    fir['fn_kwargs'].pop('i', None)
    fir['fn_kwargs'].pop('o', None)
    modelspec.append(fir)

    if mode == 'dual':
        n_targets = recording.meta['n_targets']
        ms = create_stream_merge(n_targets)
        modelspec.append(ms)
    elif mode == 'single':
        n_targets = recording.meta['n_targets']
        ms = create_single_stream_merge(n_targets)
        modelspec.append(ms)
    else:
        raise ValueError('Unsupported mode')

    lvl = deepcopy(kw_registry['lvl1'])
    lvl['id'] = 'lvl1'
    modelspec.append(lvl)

    if dexp:
        dexp = deepcopy(kw_registry['dexp1'])
        dexp['id'] = 'dexp1'
        modelspec.append(dexp)

    return modelspec
