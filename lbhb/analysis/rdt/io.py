import re
import pickle
import pandas as pd
import numpy as np
import os.path

from nems.recording import Recording
from nems.epoch import merge_epoch

from joblib import Memory

tmpdir = '/tmp/nems'
if not os.path.exists(tmpdir):
    os.makedirs(tmpdir, exist_ok=True)
memory = Memory(cachedir=tmpdir)


def get_sequence_epochs(epochs):
    epochs = epochs.copy()
    extract_token_id = lambda x: x.split(',')[1].strip()
    join_token_id = lambda x: ':'.join(x)

    epochs.loc[:, 'trial'] = None
    t = epochs['name'] == 'TRIAL'
    epochs.loc[:, 'trial'] = t.cumsum()-1
    m = epochs.name.str.startswith('Stim')
    subset = epochs.loc[m].copy()

    subset.loc[:, 'name'] = subset['name'].apply(extract_token_id)

    sequence = subset.groupby('trial')['name'] \
        .apply(join_token_id) \
        .astype('category')
    sequence = sequence.reset_index()
    sequence.loc[:, 'id'] = sequence['name'].values.codes
    sequence_to_trial_map = sequence.groupby('id').trial.unique()

    new_epochs = []
    for sequence_id, trial_ids in sequence_to_trial_map.items():
        new_epoch_name = f'SEQUENCE_{sequence_id}'
        for trial_id in trial_ids:
            m = epochs['name'] == f'TRIAL{trial_id}'
            if m.sum() != 1:
                raise ValueError('More than one match for trial')
            epoch = epochs.loc[m].iloc[0]
            new_epoch = epoch[['start', 'end']].copy()
            new_epoch['name'] = new_epoch_name
            new_epoch.name = None
            new_epochs.append(new_epoch)

    return pd.DataFrame(new_epochs)


def mark_first_reference(epochs, prestim_silence=0.5):
    def _mark_first_reference(row, stim_starts):
        if ('Reference' in row['name']) and (row['start'] in stim_starts):
            row['name'] += ', first'
        return row
    m = epochs['name'] == 'TRIAL'
    stim_starts = epochs.loc[m, 'start'].values + prestim_silence
    return epochs.apply(_mark_first_reference, axis=1, args=(stim_starts,))


def extract_stream_type(df):
    m = df['name'] == 'TRIAL'
    df['trial'] = m.cumsum()
    result = []
    for _, group in df.groupby('trial'):
        m = group['name'].str.startswith('Stim')
        g = group.loc[m]
        has_plus = g['name'].str.contains('\+')
        if np.all(has_plus):
            stream_type = 'dual'
        elif np.all(~has_plus):
            stream_type = 'single'
        else:
            raise ValueError('stream type issue')
        series = pd.Series({
            'name': stream_type,
            'start': group['start'].min(),
            'end': group['end'].max(),
        })
        result.append(series)
    return pd.DataFrame(result, columns=['name', 'start', 'end'])


def extract_sample_ids(df, stream):
    result = []
    for _, row in df.iterrows():
        if row['name'].startswith('Stim'):
            sample_id = row['name'].split(',')[1].strip()
            if '+' in sample_id:
                if stream == 'fg':
                    id = sample_id.split('+')[0]
                else:
                    id = sample_id.split('+')[1]
                series = pd.Series({
                    'name': id,
                    'start': row['start'],
                    'end': row['end'],
                })
                result.append(series)
            elif stream == 'fg':
                series = pd.Series({
                    'name': sample_id,
                    'start': row['start'],
                    'end': row['end'],
                })
                result.append(series)
    return pd.DataFrame(result, columns=['name', 'start', 'end'])


def extract_repeating(df):
    m = df['name'].apply(lambda x: 'TargetRep' in x)
    epoch = df.loc[m, ['start', 'end']].values
    epoch = merge_epoch(epoch)
    return pd.DataFrame({
        'name': 'repeating',
        'start': epoch[:, 0],
        'end': epoch[:, 1],
    }, columns=['name', 'start', 'end'])


def extract_random(df):
    m = df['name'].apply(lambda x: 'Reference' in x)
    epoch = df.loc[m, ['start', 'end']].values
    delta = epoch[:, 1]-epoch[:, 0]
    duration = np.unique(np.round(delta, 2))
    if len(duration) != 1:
        raise ValueError('Cannot find sample duration')
    # Shift by one sample duration. This means that the reference phase will
    # exclude the very first reference but include the very first target (but
    # not the repeating target).
    epoch += duration[0]
    return pd.DataFrame({
        'name': 'reference',
        'start': epoch[:, 0],
        'end': epoch[:, 1],
    }, columns=['name', 'start', 'end'])


@memory.cache
def reformat_RDT_recording(recording):
    new_epochs = []
    epochs = recording['stim'].epochs

    state = recording['state']
    state.chans = ['repeating', 'single_stream', 'target_id']

    repeating = state.loc['repeating'].as_continuous()[0]
    target_id = state.loc['target_id'].as_continuous()[0]
    if np.any(target_id[repeating == 1] == 0):
        # This is a sanity check to ensure that we always know the *correct*
        # target when it matters (e.g., during the repeating phase).
        raise ValueError('Data missing')

    fg_samples = extract_sample_ids(epochs, stream='fg')
    bg_samples = extract_sample_ids(epochs, stream='bg')
    new_epochs.append(fg_samples.copy())
    new_epochs.append(bg_samples.copy())

    fg_samples['name'] = fg_samples['name'].apply(lambda x: f'fg_{x}')
    bg_samples['name'] = bg_samples['name'].apply(lambda x: f'bg_{x}')
    new_epochs.append(fg_samples.copy())
    new_epochs.append(bg_samples.copy())

    recording['stim1'].epochs = fg_samples
    recording['stim2'].epochs = bg_samples

    samples = pd.concat(new_epochs, ignore_index=True)
    samples.sort_values('start', inplace=True)

    # Save this to the state to facilitate analysis.
    state.epochs = samples
    target_id = state.loc['target_id']

    new_epochs.append(epochs)
    new_epochs.append(get_sequence_epochs(epochs))

    # Find out the targets

    x = target_id.as_continuous()[0]
    targets = np.unique(x[x != 0])

    # Tag the epochs as target and whether it's current or not.
    for i, target in enumerate(targets):
        target_name = '{:02d}'.format(int(target))
        m = samples['name'].str.match(target_name)
        e = samples.loc[m].copy()
        e['name'] = 'target_{}'.format(i)
        new_epochs.append(e)

        # extract all occurances of the target across all trials. find out what
        # the actual (intended) target was for each particular trial.
        x = target_id.extract_epoch(target_name)[:, 0, 0]
        m = x == target
        c = e.loc[m].copy()
        c['name'] = 'current_target'
        new_epochs.append(c)

        c = e.loc[~m].copy()
        c['name'] = 'other_target'
        new_epochs.append(c)

    repeating = extract_repeating(epochs)
    new_epochs.append(repeating)

    random = extract_random(epochs)
    new_epochs.append(random)

    stream_type = extract_stream_type(epochs)
    new_epochs.append(stream_type)

    epochs = pd.concat(new_epochs, ignore_index=True)
    epochs.drop_duplicates(inplace=True)
    epochs.sort_values('start', inplace=True)

    state.epochs = epochs
    recording['repeating'] = state.loc['repeating']
    tid = state.loc['target_id'].as_continuous().astype('i')
    recording['target_id'] = state._modified_copy(tid)

    ss = state.loc['single_stream'].as_continuous().astype('bool')
    ds = ~ss

    recording['single_stream'] = state._modified_copy(ss)
    recording['dual_stream'] = state._modified_copy(ds)

    target_map = dict((t, i) for i, t in enumerate(targets))
    target_map[0] = 0
    x = target_id.as_continuous()
    x_mapped = np.vectorize(target_map.get, otypes=[np.float])(x)

    recording['target_id_map'] = target_id._modified_copy(x_mapped)
    recording['target_id_map'].meta = {'target_map': target_map}
    recording.meta = {'n_targets': len(targets)}

    recording['fg'] = recording['stim1']
    recording['bg'] = recording['stim2']
    del recording.signals['stim1']
    del recording.signals['stim2']
    del recording.signals['state']

    resp = recording['resp']
    recording['resp'] = resp._modified_copy(resp._data[..., :-1])

    epochs = recording.epochs[['name', 'start', 'end']]
    epochs.index = np.arange(len(epochs))

    # This fixes a bug in BAPHY->NEMS conversion of timestamps (not all trials
    # from the same sequence are identical length and may be off by 0.001).
    # This should be fine for RDT data because all relevant epochs are
    # in units of 0.5, 0.15, 0.3, 0.25 sec.
    epochs['start'] = epochs['start'].round(2)
    epochs['end'] = epochs['end'].round(2)

    for s in recording.signals.values():
        s.epochs = epochs
        s._data = s._data.astype('double')

    return recording


@memory.cache
def load_recording(batch, cell, reformat=True):
    from nems.recording import Recording
    url = 'http://hyrax.ohsu.edu:3000/baphy/{}/{}?' \
        'rasterfs=100&includeprestim=1&stimfmt=ozgf&chancount=18' \
        '&pupil=0&stim=1&pertrial=1&runclass=RDT'
    url = url.format(batch, cell)
    recording = Recording.load_url(url)
    if reformat:
        recording = reformat_RDT_recording(recording)
    return recording


def remove_nan(recording):
    i = np.isnan(recording['fg'].as_continuous())
    new_signals = {}
    for name, signal in recording.items():
        new_data = signal.as_continuous()[:, i]
        new_signals[name] = signal._modified_copy(new_data)
    return Recording(new_signals)


if __name__ == '__main__':
    recording = load_recording('269', 'zee021e-c1')

    #import itertools
    #import pylab as pl

    #epochs = recording['fg'].extract_epoch('03')
    #f, axes = pl.subplots(4, 4)
    #for i, ax in zip(range(16), itertools.chain(*axes)):
    #    ax.imshow(epochs[i])

    #epochs = recording['bg'].extract_epoch('03')
    #f, axes = pl.subplots(4, 4)
    #for i, ax in zip(range(16), itertools.chain(*axes)):
    #    ax.imshow(epochs[i])

    #epochs = recording['resp'].extract_epoch('03')
    #f, axes = pl.subplots(4, 4)
    #for i, ax in zip(range(16), itertools.chain(*axes)):
    #    ax.plot(epochs[i, 0])

    #pl.show()
