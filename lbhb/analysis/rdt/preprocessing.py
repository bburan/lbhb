import re
import numpy as np
import pandas as pd
from nems.epoch import epoch_difference, epoch_union
from nems.epoch import epoch_difference, epoch_intersection


def select_balanced_targets(epochs, rng):
    pattern = 'target_(\d+)_(random|repeating)_(dual|single)'
    epoch_counts = epochs.groupby('name').size()
    epoch_counts = epoch_counts.filter(regex=pattern)

    pattern = re.compile(pattern)
    result = []
    for name, count in epoch_counts.iteritems():
        target, phase, stream = pattern.match(name).groups()
        result.append({
            'target': target,
            'phase': phase,
            'stream': stream,
            'count': count,
        })
    result = pd.DataFrame(result).set_index(['target', 'stream', 'phase'])
    result = result.unstack('phase')

    keep = []
    for (target, stream), counts in result.iterrows():
        random, repeating = counts.values
        repeating_epoch_name = f'target_{target}_repeating_{stream}'
        m_repeating = epochs['name'] == repeating_epoch_name
        random_epoch_name = f'target_{target}_random_{stream}'
        m_random = epochs['name'] == random_epoch_name

        keep.append(epochs.loc[m_random])
        if random >= repeating:
            keep.append(epochs.loc[m_repeating])
            continue
        else:
            subset = epochs.loc[m_repeating].sample(random, random_state=rng)
            keep.append(subset)

    keep = pd.concat(keep, ignore_index=True)[['start', 'end']]
    return keep


def split_est_val(recording, balance_phase=False):
    rng = np.random.RandomState(0)

    epochs = recording.epochs
    est_times, val_times = get_est_val_times_by_sequence(recording, rng)

    if balance_phase:
        target_times = select_balanced_targets(epochs, rng)

        m = epochs['name'].str.contains('^repeating$')
        repeating_times = epochs.loc[m, ['start', 'end']].values

        # Remove the repeating phase from the dataset
        est_times = epoch_difference(est_times, repeating_times)
        # Now, add back in selected targets from repeating phase
        est_times = epoch_union(est_times, target_times)

        # Remove the repeating phase from the dataset
        val_times = epoch_difference(val_times, repeating_times)
        # Now, add back in selected targets from repeating phase
        val_times = epoch_union(val_times, target_times)

    return est_times, val_times


def get_est_val_times_by_sequence(recording, rng):
    epochs = recording.epochs
    m = epochs['name'].str.match('^SEQUENCE')
    sequences = epochs.loc[m, 'name'].unique()

    s_map = {}
    for s in sequences:
        tid = recording['target_id_map'].extract_epoch(s).ravel()[0]
        tid = int(tid)
        is_ds = recording['dual_stream'].extract_epoch(s).ravel()[0]
        is_ds = bool(is_ds)
        s_map.setdefault((tid, is_ds), []).append(s)

    val_epochs, est_epochs = [], []
    for v in s_map.values():
        rng.shuffle(v)
        val_size = round(len(v) * 0.2)
        v_epochs, e_epochs = v[:val_size], v[val_size:]
        val_epochs.extend(v_epochs)
        est_epochs.extend(e_epochs)

    # This returns the times of the validation and estimation sequences
    m = epochs['name'].apply(lambda x: x in val_epochs)
    val_times = epochs.loc[m][['start', 'end']].values
    m = epochs['name'].apply(lambda x: x in est_epochs)
    est_times = epochs.loc[m][['start', 'end']].values

    return est_times, val_times


def shuffle_streams(recording):
    fg = recording['fg'].as_continuous().copy()
    bg = recording['bg'].as_continuous().copy()
    i_all = np.arange(fg.shape[-1])
    n = round(fg.shape[-1]/2)
    np.random.shuffle(i_all)
    i_switch = i_all[:n]
    fg[:, i_switch], bg[:, i_switch] = bg[:, i_switch], fg[:, i_switch]

    s = recording['fg']
    recording['fg'] = s._modified_copy(fg)
    recording['bg'] = s._modified_copy(bg)
    return recording

def select_times(recording, subset, random_only=True, dual_only=True):
    '''
    Parameters
    ----------
    recording : nems.recording.Recording
        The recording object.
    subset : Nx2 array
        Epochs representing the selected subset (e.g., from an est/val split).
    random_only : bool
        If True, return only the repeating portion of the subset
    dual_only : bool
        If True, return only the dual stream portion of the subset
    '''
    epochs = recording['stim'].epochs

    m_dual = epochs['name'] == 'dual'
    m_repeating = epochs['name'] == 'repeating'
    m_trial = epochs['name'] == 'TRIAL'

    dual_epochs = epochs.loc[m_dual, ['start', 'end']].values
    repeating_epochs = epochs.loc[m_repeating, ['start', 'end']].values
    trial_epochs = epochs.loc[m_trial, ['start', 'end']].values

    subset = trial_epochs.copy()

    if random_only:
        subset = epoch_difference(subset, repeating_epochs)

    if dual_only:
        subset = epoch_intersection(subset, dual_epochs)

    return recording.select_times(subset)
