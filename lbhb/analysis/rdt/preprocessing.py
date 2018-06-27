import re
import numpy as np
import pandas as pd
from nems.epoch import epoch_difference, epoch_union


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

    val = recording.select_times(val_times)
    est = recording.select_times(est_times)
    return est, val


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

