import shutil
import os.path

import matplotlib as mp
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
import pylab as pl
import bcolz

from psi.data.io import Recording

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


def _load_trial_logs(experiments, progressbar=True, catch_errors=False):
    trial_logs = []
    experiment_info = []
    iterator = tqdm(experiments) if progressbar else experiments
    errors = []
    for e in iterator:
        try:
            recording = Recording(e)
            trial_logs.append(recording.trial_log)
            ei = util.parse_filename(e)
            experiment_info.append(ei)
        except Exception as exc:
            if not catch_errors:
                raise
            errors.append((e, exc))

    if catch_errors:
        return trial_logs, experiment_info, errors

    return trial_logs, experiment_info


def load_trial_logs(experiments, progressbar=True):
    trial_logs, experiment_info = _load_trial_logs(experiments, progressbar)
    experiment_info = pd.DataFrame(experiment_info)
    trial_log = pd.concat(trial_logs,
                          keys=np.arange(len(trial_logs)),
                          names=['experiment', 'trial'],
                          sort=True)
    trial_log = trial_log.join(experiment_info, on=['experiment'])
    return trial_log


def find_bad_experiments(experiments, progressbar=True):
    _, _, excs = _load_trial_logs(experiments, progressbar, catch_errors=True)
    return excs


def load_sam_behavior(experiments, progressbar=True):
    data = load_trial_logs(experiments)

    for c in data.columns:
        if c.startswith('background_continuous'):
            c_other = c.replace('continuous_', '')
            print(c, c_other)
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
    data['depth'] = data['target_SAM_depth']
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


def summarize_masker_level(animal_data, variable='dbi', nogo_value=-27,
                           reindex_dates=True):
    grouping = ['date', 'masker_level', 'target_tone_frequency', variable]
    x = animal_data.groupby(grouping)['yes'].agg(['size', 'sum', 'mean'])
    fa = x['mean'].xs(nogo_value, level=variable).rename('fa')
    x = x.join(fa, on=fa.index.names)
    x['d_prime'] = x[['mean', 'fa']].clip(lower=0.05, upper=0.95) \
        .apply(norm.ppf).eval('mean-fa')

    if reindex_dates:
        dates = pd.DatetimeIndex(x.index.get_level_values('date'))
        lb, ub = dates.min(), dates.max()
        lb_mon = lb - pd.Timedelta(days=lb.dayofweek)
        ub_fri = ub + pd.Timedelta(days=4-ub.dayofweek)
        date_range = pd.bdate_range(lb_mon, ub_fri, name='date')
        x = x.unstack(grouping[1:]) \
            .reindex(date_range) \
            .stack(grouping[1:], dropna=False)
    return x


def heatmap_masker_level(animal_data, metrics=['size', 'd_prime'],
                         variable='dbi', metric_heatmap_kw=None, **heatmap_kw):
    summary = summarize_masker_level(animal_data, variable)[metrics]
    n_date = len(summary.index.unique('date'))
    n_var = len(summary.index.unique(variable))
    n_cols = len(summary) / n_date / n_var

    height = 10 * n_date / 40
    width = 10 * n_cols * n_var * len(metrics) / 40

    figure, axes = pl.subplots(1, 2, figsize=(width, height), sharex=False,
                               sharey=False)

    fmt = {'size': '.0f', 'd_prime': '.1f'}

    if metric_heatmap_kw is None:
        metric_heatmap_kw = {}

    for ax, metric in zip(axes, metrics):
        s = summary[metric] \
            .unstack('masker_level') \
            .unstack('target_tone_frequency') \
            .unstack(variable)
        masker_levels, target_frequency, depths = zip(*s.columns.values)
        masker_levels = np.unique(masker_levels)

        kw = heatmap_kw.copy()
        kw.update(metric_heatmap_kw.get(metric, {}))
        sns.heatmap(s, ax=ax, annot=True,  fmt=fmt[metric],
                    yticklabels=s.index.format(),
                    xticklabels=s.columns.format(), cbar=False,
                    cmap='RdYlGn', **kw)

        n_sections = len(s) // 5
        for i in range(n_sections+1):
            ax.axhline(i*5, color='white', lw=3)
            ax.axhline(i*5, color='black', lw=1)

        c = s.T.groupby(s.columns.names[:-1]).size().values
        for i in np.cumsum(c):
            ax.axvline(i, color='white', lw=3)
            ax.axvline(i, color='black', lw=1)

        ax.set_ylabel('Date')
        ax.set_xlabel('AM depth')

    return figure


def generate_report(data, filename, variable='depth'):
    metric_kw = {
        'size': dict(vmin=0, vmax=50, center=20),
        'd_prime': dict(vmin=-1, vmax=4, center=1),
    }
    pdf = PdfPages(filename)
    for animal, animal_data in data.groupby('animal'):
        f_trials = heatmap_masker_level(animal_data,
                                        metric_heatmap_kw=metric_kw,
                                        square=True)
        f_trials.suptitle(animal)
        pdf.savefig(f_trials)
    pdf.close()
    pl.close('all')
