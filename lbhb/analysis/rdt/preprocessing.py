import numpy as np


def split_est_val_by_sequence(recording):

    rng = np.random.RandomState(0)
    epochs = recording.epochs
    m = epochs['name'].str.match('^SEQUENCE')
    sequences = recording.epochs.loc[m, 'name'].unique()

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

    m = epochs['name'].apply(lambda x: x in val_epochs)
    val_times = epochs.loc[m][['start', 'end']].values
    m = epochs['name'].apply(lambda x: x in est_epochs)
    est_times = epochs.loc[m][['start', 'end']].values
    val = recording.select_times(val_times)
    est = recording.select_times(est_times)
    return est, val
