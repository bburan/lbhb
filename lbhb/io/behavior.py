import shutil
import os.path

import pandas as pd
import bcolz

from . import util


def load_trial_log(dirname):
    try:
        tl_dirname = os.path.join(dirname, 'trial_log')
        trial_log = util.fix_object_columns(tl_dirname)
        experiment_info = util.parse_filename(dirname)
        for k, v in experiment_info.items():
            trial_log[k] = v
        trial_log.index.name = 'trial'
        trial_log.reset_index(inplace=True)
        trial_log.set_index(['datetime', 'animal', 'trial'], inplace=True)
        return trial_log
    except Exception as e:
        m = 'Error parsing {}'.format(dirname)
        raise IOError(m) from e


def load_event_log(dirname):
    el_dirname = os.path.join(dirname, 'event_log')
    event_log = util.fix_object_columns(el_dirname)
    experiment_info = util.parse_filename(dirname)
    for k, v in experiment_info.items():
        event_log[k] = v
    event_log.index.name = 'id'
    event_log.reset_index(inplace=True)
    event_log.set_index(['datetime', 'animal', 'id'], inplace=True)
    return event_log


def remove_empty_experiments(experiments):
    valid_experiments = []
    for experiment in experiments[:]:
        tl_dirname = os.path.join(experiment, 'trial_log')
        table = bcolz.ctable(rootdir=tl_dirname)
        if len(table) == 0:
            shutil.rmtree(experiment)
        else:
            valid_experiments.append(experiment)
    return valid_experiments


def load_sam_behavior(experiments):
    trial_logs = [load_trial_log(e) for e in experiments]
    data = pd.concat(trial_logs)
    data.sort_index(inplace=True)

    for c in data.columns:
        if c.startswith('background_continuous'):
            c_other = c.replace('continuous_', '')
            m = data[c].notnull()
            data.loc[m, c_other] = data.loc[m, c]
            data.drop(columns=c, inplace=True)

    rename_map = {}
    for c in data.columns:
        if c.startswith('background_bandlimited'):
            rename_map[c] = c.replace('background_bandlimited_noise', 'masker')
        if c.startswith('target_bandlimited_noise'):
            rename_map[c] = c.replace('target_bandlimited_noise', 'target')

    data.rename(columns=rename_map, inplace=True)

    data = data.reset_index()
    data['date'] = data['datetime'].map(lambda x: x.date())
    data['yes'] = data['response'].map({'no response': False, 'poke': False, 'reward': True})
    data['masker_level'] = pd.Categorical(data.masker_level)
    data['depth'] = data['target_SAM_depth']
    n_depths = data.groupby(['animal', 'date'])['target_SAM_depth'].nunique()

    data_clean = data.copy()
    for animal, date in n_depths[n_depths < 5].index.get_values():
        data_clean = data_clean.query('~(animal == @animal and date == @date)')

    def count_sessions(df):
        unique_dates = df['date'].unique()
        unique_dates.sort()
        mapping = {d: i for i, d in enumerate(unique_dates)}
        df['session'] = df['date'].map(mapping)
        return df

    return data_clean.groupby('animal').apply(count_sessions)
