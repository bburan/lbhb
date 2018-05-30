import numpy as np

from nems.keywords import defaults as keywords


def dual_lvl(rec, level):
    fn = lambda x: x + level
    return [
        rec['bg_pred'].transform(fn, 'bg_pred'),
        rec['fg_pred'].transform(fn, 'fg_pred'),
    ]


def merge_streams(rec, gain):
    # Gain is expected to be a n_target x 3 array. First column is single-stream
    # gain, second column is background and third is forgeground.
    is_repeating = rec['repeating'].as_continuous()[0]
    is_ds = rec['dual_stream'].as_continuous()[0].astype('i')
    t_map = rec['target_id_map'].as_continuous()[0].astype('i')
    bg = rec['bg_pred'].as_continuous()
    fg = rec['fg_pred'].as_continuous()

    # Pull out the correct gain for the target and zero out if not repeatnig
    r_gain = gain[t_map].T * is_repeating

    # Pull out the correct gain term (from first row if single-stream, second
    # row if dual stream).
    bg_gain = r_gain[0]
    bg_gain[is_ds] = r_gain[1, is_ds]

    # Pull out the correct gain term (from first row if single-stream, third
    # row if dual stream).
    fg_gain = r_gain[0]
    fg_gain[is_ds] = r_gain[2, is_ds]

    pred = bg*np.exp(bg_gain) + fg*np.exp(fg_gain)

    # Now, package up the signal nicely
    pred_signal = rec['bg_pred']._modified_copy(pred, name='pred')
    return [pred_signal]


def create_ms(n_targets):
    name = 'rdt:ms{}'.format(n_targets)
    gain_mean = np.zeros((n_targets, 3))
    gain_sd = np.ones((n_targets, 3))
    template = {
        'id': 'MS',
        'fn': 'rdt.modules.merge_streams',
        'prior': {
            'gain': ('Normal', {'mean': gain_mean, 'sd': gain_sd})
        }
    }
    return template


from nems.keywords import defaults as kw_registry

def create_modelspec(recording, n_substreams):
    wc = kw_registry[f'wcg18x{n_substreams}']
    wc['id'] = f'RDTwcg18x{n_substreams}'
    wc['fn'] = 'rdt.weight_channels.gaussian'
    wc['fn_kwargs'].pop('i', None)
    wc['fn_kwargs'].pop('o', None)

    fir = kw_registry[f'fir{n_substreams}x10']
    fir['id'] = f'RDTfir{n_substreams}x10'
    fir['fn'] = 'rdt.fir.basic'
    fir['fn_kwargs'].pop('i', None)
    fir['fn_kwargs'].pop('o', None)

    lvl = kw_registry['lvl1']
    lvl['id'] = f'RDTlvl1'
    lvl['fn'] = 'rdt.modules.dual_lvl'
    lvl['fn_kwargs'].pop('i', None)
    lvl['fn_kwargs'].pop('o', None)

    n_targets = recording.meta['n_targets']
    ms = create_ms(n_targets)

    dexp = kw_registry['dexp1']

    return [wc, fir, lvl, ms, dexp]
