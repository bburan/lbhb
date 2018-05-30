import numpy as np


def split_est_val_by_sequence(recording):
    epochs = recording.epochs
    m = epochs['name'].str.match('^SEQUENCE_')
    sequences = recording.epochs.loc[m, 'name'].unique()
    np.random.shuffle(sequences)
    val_size = round(len(sequences) * 0.2)
    val_epochs, est_epochs = sequences[:val_size], sequences[val_size:]
    m = epochs['name'].apply(lambda x: x in val_epochs)
    val_times = epochs.loc[m][['start', 'end']].values
    m = epochs['name'].apply(lambda x: x in est_epochs)
    est_times = epochs.loc[m][['start', 'end']].values
    val = recording.select_times(val_times)
    est = recording.select_times(est_times)
    return est, val
