import numpy as np

from nems.keywords import defaults as kw_registry

def dual_lvl(rec, level):
    fn = lambda x: x + level
    return [
        rec['bg_pred'].transform(fn, 'bg_pred'),
        rec['fg_pred'].transform(fn, 'fg_pred'),
    ]


def stream_merge_sf(rec, ss_gain, fg_gain, bg_gain):
    is_repeating = rec['repeating'].as_continuous()[0].astype('bool')
    is_ds = rec['dual_stream'].as_continuous()[0].astype('bool')
    t_map = rec['target_id_map'].as_continuous()[0].astype('i')

    ss_gain_mapped = ss_gain[t_map]
    fg_gain_mapped = fg_gain[t_map]
    bg_gain_mapped = bg_gain[t_map]

    fg_gain = np.zeros(t_map.shape, dtype=np.double)
    bg_gain = np.zeros(t_map.shape, dtype=np.double)

    m_ds_repeating = is_repeating & is_ds
    m_ss_repeating = is_repeating & (~is_ds)

    # Set gains for repeating portions of ss and ds
    fg_gain[m_ss_repeating] = ss_gain_mapped[m_ss_repeating]
    fg_gain[m_ds_repeating] = fg_gain_mapped[m_ds_repeating]
    bg_gain[m_ds_repeating] = bg_gain_mapped[m_ds_repeating]

    # Convert to scaling factor
    bg_sf = np.exp(bg_gain)
    fg_sf = np.exp(fg_gain)

    # Ensure that there is no contribution of background stream (there shouldn't
    # be, but just in case a prior module added some sort of shift of the
    # background prediction away from 0 such as levelshift).
    bg_sf[~is_ds] = 0

    return fg_sf, bg_sf


def stream_merge(rec, ss_gain, fg_gain, bg_gain):
    fg_sf, bg_sf = stream_merge_sf(rec, ss_gain, fg_gain, bg_gain)
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
        'id': 'RDTgain',
        'fn': 'lbhb.analysis.rdt.modules.stream_merge',
        'prior': {
            'ss_gain': ('Normal', {'mean': gain_mean, 'sd': gain_sd}),
            'fg_gain': ('Normal', {'mean': gain_mean, 'sd': gain_sd}),
            'bg_gain': ('Normal', {'mean': gain_mean, 'sd': gain_sd}),
        }
    }
    return template


def create_modelspec(recording, wcg_n, fir_n, pre_shift):
    modelspec = []

    wc = kw_registry[f'wcg18x{wcg_n}'].copy()
    wc['id'] = f'RDTwcg18x{wcg_n}'
    wc['fn'] = 'lbhb.analysis.rdt.weight_channels.gaussian'
    wc['fn_kwargs'].pop('i', None)
    wc['fn_kwargs'].pop('o', None)
    modelspec.append(wc)

    fir = kw_registry[f'fir{wcg_n}x{fir_n}'].copy()
    fir['id'] = f'RDTfir{wcg_n}x{fir_n}'
    fir['fn'] = 'lbhb.analysis.rdt.fir.basic'
    fir['fn_kwargs'].pop('i', None)
    fir['fn_kwargs'].pop('o', None)
    modelspec.append(fir)

    if pre_shift:
        lvl = kw_registry['lvl1'].copy()
        lvl['id'] = 'RDTlvl1'
        lvl['fn'] = 'lbhb.analysis.rdt.modules.dual_lvl'
        lvl['fn_kwargs'].pop('i', None)
        lvl['fn_kwargs'].pop('o', None)
        modelspec.append(lvl)

    n_targets = recording.meta['n_targets']
    ms = create_stream_merge(n_targets)
    modelspec.append(ms)

    lvl = kw_registry['lvl1'].copy()
    lvl['id'] = 'lvl1'
    modelspec.append(lvl)

    dexp = kw_registry['dexp1'].copy()
    dexp['id'] = 'dexp1'
    modelspec.append(dexp)

    return modelspec
