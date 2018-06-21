import shutil
import os.path

import matplotlib as mp
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as pl
import bcolz
from psi.data.io.bcolz_tools import load_ctable_as_df

from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from . import util


from joblib import Memory
tmpdir = '/tmp/lbhb'
if not os.path.exists(tmpdir):
    os.makedirs(tmpdir, exist_ok=True)
memory = Memory(cachedir=tmpdir)


def load_trial_log(dirname):
    try:
        tl_dirname = os.path.join(dirname, 'trial_log')
        trial_log = bcolz.ctable(rootdir=tl_dirname).todataframe()
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


def _load_trial_logs(experiments, progressbar=True):
    trial_logs = []
    experiment_info = []
    iterator = tqdm(experiments) if progressbar else experiments
    for e in iterator:
        tl_dirname = os.path.join(e, 'trial_log')
        tl = load_ctable_as_df(tl_dirname)
        trial_logs.append(tl)
        ei = util.parse_filename(e)
        experiment_info.append(ei)
    return trial_logs, experiment_info


def load_trial_logs(experiments, progressbar=True):
    trial_logs, experiment_info = _load_trial_logs(experiments, progressbar)
    experiment_info = pd.DataFrame(experiment_info)
    trial_log = pd.concat(trial_logs,
                          keys=np.arange(len(trial_logs)),
                          names=['experiment', 'trial'],
                          sort=True)
    return trial_log.join(experiment_info, on=['experiment'])


def load_sam_behavior(experiments, progressbar=True):
    data = load_trial_logs(experiments)

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

    #data = data.reset_index()
    data['date'] = data['datetime'].map(lambda x: x.date())
    data['yes'] = data['response'].map({'no response': False, 'poke': False, 'reward': True})
    data['masker_level'] = pd.Categorical(data.masker_level, ordered=True)
    data['depth'] = pd.Categorical(data['target_SAM_depth'], ordered=True)
    n_depths = data.groupby(['animal', 'date'])['depth'] \
        .nunique().rename('n_depths')
    data = data.join(n_depths, on=['animal', 'date'])

    def count_sessions(df):
        unique_dates = df['date'].unique()
        unique_dates.sort()
        mapping = {d: i for i, d in enumerate(unique_dates)}
        df['session'] = df['date'].map(mapping)
        return df

    return data.groupby('animal').apply(count_sessions)


def summarize_masker_level(animal_data):

    def summarize(df):
        ml_trials = df.groupby(['masker_level', 'depth'])[['size', 'sum']].sum()
        ml_trials['mean'] = ml_trials['sum']/ml_trials['size']
        return ml_trials

        strings = []
        for ml, ml_data in ml_trials.groupby('masker_level'):
            n_trials = ml_data.sum()
            n_depths = len(ml_data)
            depths = []
            for (_, depth), trials in ml_data.iteritems():
                depths.append(f'{depth} ({trials})')
            depths = ', '.join(depths)
            #depths = ':'.join(str(v) for v in ml_data.index.get_level_values('depth'))
            #strings.append(f'{ml} ({n_trials}/{n_depths}) {depths}')
            strings.append(f'{ml} {depths}')
        return ', '.join(strings)
        #return ', '.join(f'{ml} ({trials})' for ml, trials \
        #                 in ml_trials.iteritems() if trials > 0)

    x = animal_data.groupby(['date', 'masker_level', 'depth'])['yes'] \
        .agg(['size', 'sum']) \
        .reset_index()

    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    return x.groupby('date').apply(summarize)
    #return x.groupby([
    #    x['datetime'].dt.year.rename('year'),
    #    x['datetime'].dt.weekofyear.rename('week'),
    #    x['datetime'].dt.weekday_name.rename('day'),
    #]).apply(summarize) \
    #    .unstack('day') \
    #    .reindex(weekday_names, axis=1)


def heatmap_masker_level(animal_data, metric='size', section_size=5):
    summary = summarize_masker_level(animal_data)[metric]
    summary = summary.unstack('masker_level').unstack('depth')
    n_rows, n_cols = summary.shape
    figsize = n_cols / 28 * 15, n_rows / 3 * 1
    figure = pl.figure(figsize=figsize)

    # Create pretty labels
    masker_levels, depths = zip(*summary.columns.values)
    masker_levels = np.unique(masker_levels)
    depths = np.unique(depths)
    i = np.argmin(np.abs(depths - depths.mean()))
    middle_depth = depths[i]

    labels = []
    for ml, d in summary.columns.values:
        label = f'{d}'
        labels.append(label)

    fmt = '.0f' if metric == 'size' else '0.2f'
    ax = sns.heatmap(summary, annot=True, fmt=fmt, xticklabels=labels,
                     cbar=False, cmap='gray_r')

    # Add some reference symbols to help visualize results
    n_depths = len(depths)
    transform = mp.transforms.blended_transform_factory(ax.transData,
                                                        ax.transAxes)
    for i in range(len(masker_levels)+1):
        rect = mp.patches.Rectangle((i*n_depths, 0), 1, 1, transform=transform,
                                    facecolor='0.5', alpha=0.25)
        ax.add_patch(rect)
        rect = mp.patches.Rectangle((i*n_depths+n_depths-1, 0), 1, 1,
                                    transform=transform, facecolor='0.5',
                                    alpha=0.25)
        ax.add_patch(rect)
        ax.axvline(i*n_depths, color='white', lw=3)
        ax.axvline(i*n_depths, color='black', lw=1)

    for i, ml in enumerate(masker_levels):
        ax.text(i*n_depths + n_depths/2, 1.0, f'{ml} dB SPL masker',
                transform=transform, ha='center', va='bottom')

    n_sections = len(summary) // section_size
    for i in range(n_sections+1):
        ax.axhline(i*section_size, color='white', lw=3)
        ax.axhline(i*section_size, color='black', lw=1)

    ax.set_ylabel('Date')
    ax.set_xlabel('AM depth')
    return figure


def generate_report(data, filename):
    pdf = PdfPages(filename)
    for animal, animal_data in data.groupby('animal'):
        f_trials = heatmap_masker_level(animal_data, 'size')
        f_trials.suptitle(animal)
        f_performance = heatmap_masker_level(animal_data, 'mean')
        f_performance.suptitle(animal)
        pdf.savefig(f_trials)
        pdf.savefig(f_performance)
    pdf.close()
    pl.close('all')
