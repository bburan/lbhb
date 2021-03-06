'''
Interface for reading tone-evoked ABR data generated by psiexperiment

This supports merging across aggregate files. Right now there's no support for
reading from the raw (i.e., untrial_filtered) data. Maybe eventually. It wouldn't be
hard to add, but would need support for trial_filter specs as well as ensuring that we
pull out enough pre/post samples for proper trial_filtering.
'''
import ast
import functools
from pathlib import Path
import os.path
import shutil
import re
from glob import glob

from tqdm import tqdm
import bcolz
import numpy as np
import pandas as pd
from scipy import signal

from . import util

from psi.data.io import abr


MAXSIZE = 1024


def load_n(base_folder):
    n = []
    keys = []
    names = ['start', 'end', 'filter_lb', 'filter_ub']
    for filename in base_folder.glob('*number of epochs.csv'):
        n.append(pd.read_csv(filename))
        match = P_N_EPOCHS_FILE_PATTERN.match(filename.name).groupdict()
        match['start'] = float(match['start'])*1e-3
        match['end'] = float(match['end'])*1e-3
        parse = lambda x: float(x) if x is not None else 0
        match['filter_lb'] = parse(match['filter_lb'])
        match['filter_ub'] = parse(match['filter_ub'])
        keys.append(tuple(match[k] for k in names))
    result = pd.concat(n, keys=keys, names=names)
    return result.set_index(['frequency', 'level'])['0'].rename('n')


def load_analysis(base_folder, actual_frequencies=None):
    search_pattern = os.path.join(base_folder, '*-analyzed.txt')
    result = [load_abr_analysis(f) for f in glob(search_pattern)]

    names = ['analyzer', 'start', 'end', 'filter_lb', 'filter_ub', 'frequency']
    freq, th, info, data = zip(*result)

    # Some versions of the ABR wave analysis program have a bug where the
    # frequency is rounded to the nearest tenth place. Fix this by mapping back
    # to the original frequencies in the file.
    if actual_frequencies is not None:
        actual_frequencies = np.array(list(actual_frequencies))
        freq = np.array(freq)
        d = freq[..., np.newaxis] - actual_frequencies
        i = np.abs(d).argmin(axis=1)
        freq = actual_frequencies[i]

    keys = []
    for f, i in zip(freq, info):
        key = tuple(i[n] for n in names[:-1]) + (f,)
        keys.append(key)

    index = pd.MultiIndex.from_tuples(keys, names=names)
    threshold = pd.Series(th, index=index, name='threshold')
    peaks = pd.concat(data, keys=keys, names=names, sort=False)
    peaks.sort_index(inplace=True)
    return threshold, peaks


def cleanup_analysis(result, info, simplify, analyzer=None):
    if analyzer is not None:
        result = result.xs(analyzer, level='analyzer')
    if not simplify:
        return result
    remove = []
    for level in result.index.names:
        if level not in info:
            values = result.index.get_level_values(level)
            if len(set(values)) == 1:
                remove.append(level)
    result.reset_index(remove, drop=True, inplace=True)
    return result


class ABRDataset:

    def __init__(self, experiments):
        self.experiments = experiments

    @classmethod
    def from_folder(cls, folder):
        folder = Path(folder)
        experiments = [ABRExperiment(f) for f in folder.glob('*abr')]
        return cls(experiments)

    def get_n(self, concat_columns, simplify=True):
        keys = self.get_info(concat_columns, 'list')
        n = [e.n for e in self.experiments]
        result = pd.concat(n, keys=keys, names=concat_columns)
        return cleanup_analysis(result, concat_columns, simplify)

    def get_thresholds(self, concat_columns, simplify=True, analyzer=None):
        keys = self.get_info(concat_columns, 'list')
        thresholds = [e.all_thresholds for e in self.experiments]
        result = pd.concat(thresholds, keys=keys, names=concat_columns)
        return cleanup_analysis(result, concat_columns, simplify, analyzer)

    def get_waves(self, concat_columns, simplify=True, analyzer=None,
                  compute_metrics=False, multiplier=1):
        keys = self.get_info(concat_columns, 'list')
        waves = [e.all_waves for e in self.experiments]
        result = pd.concat(waves, keys=keys, names=concat_columns)
        result = cleanup_analysis(result, concat_columns, simplify, analyzer)

        cols = [c for c in result.columns if 'amplitude' in c]
        cols.extend(['baseline', 'baseline_std'])
        result[cols] *= multiplier

        if not compute_metrics:
            return result

        cols = []
        for w in range(1, 6):
            if f'p{w}_amplitude' in result.columns:
                expr = f'p{w}_amplitude-n{w}_amplitude'
                result[f'w{w}_amplitude'] = result.eval(expr)
                result[f'w{w}_latency'] = result[f'p{w}_latency']
                expr = f'p{w}_latency-n{w}_latency'
                result[f'pn{w}_latency'] = result.eval(expr)
                cols.append(f'w{w}_amplitude')
                cols.append(f'w{w}_latency')
                cols.append(f'pn{w}_latency')
                expr = f'p{w}_amplitude-baseline'
                result[f'w{w}_amplitude_re_baseline'] = result.eval(expr)
                cols.append(f'w{w}_amplitude_re_baseline')

        return result[cols]

    def get_epochs_processed(self, concat_columns, **kwargs):
        keys = self.get_info(concat_columns, 'list')
        epochs = [e.get_average_epochs_processed(**kwargs) for e in self.experiments]
        return pd.concat(epochs, keys=keys, names=concat_columns)

    def get_epochs(self, concat_columns, **kwargs):
        keys = self.get_info(concat_columns, 'list')
        epochs = [e.get_mean_epochs(**kwargs) for e in self.experiments]
        return pd.concat(epochs, keys=keys, names=concat_columns)

    def get_epochs_filtered(self, concat_columns, **kwargs):
        keys = self.get_info(concat_columns, 'list')
        epochs = [e.get_mean_epochs_filtered(**kwargs) for e in self.experiments]
        return pd.concat(epochs, keys=keys, names=concat_columns)

    def get_inear_calibrations(self, concat_columns):
        keys = self.get_info(concat_columns, 'list')
        calibrations = [e.inear_calibration for e in self.experiments]
        return pd.concat(calibrations, keys=keys, names=concat_columns)

    def find_experiments(self, **kwargs):
        if 'date' in kwargs:
            if isinstance(kwargs['date'], pd.Timestamp):
                kwargs['date'] = kwargs['date'].date()
        cols = kwargs.keys()
        vals = tuple(kwargs.values())
        matches = []
        for e in self.experiments:
            if e.get_info(cols) == vals:
                matches.append(e)
        return matches

    def get_experiment(self, **kwargs):
        experiments = self.find_experiments(**kwargs)
        if len(experiments) > 1:
            raise ValueError('More than one match')
        elif len(experiments) < 1:
            raise ValueError('No experiment matching criteria')
        else:
            return experiments[0]

    def get_info(self, columns, flavor='dataframe', include_experiments=False):
        info = [e.get_info(columns) for e in self.experiments]
        if include_experiments:
            info = [list(i) + [e] for i, e in zip(info, self.experiments)]
            columns = list(columns) + ['experiment']
        if flavor == 'dataframe':
            return pd.DataFrame(info, columns=columns)
        elif flavor == 'index':
            return pd.MultiIndex.from_tuples(info, names=columns)
        elif flavor == 'list':
            return info

    def __getitem__(self, slice):
        return self.experiments[slice]

    def __iter__(self):
        return iter(self.experiments)


def clear_cache():
    for method_name in dir(ABRExperiment):
        method = getattr(ABRExperiment, method_name)
        if isinstance(method, property):
            method = method.fget
        if hasattr(method, 'cache_clear'):
            method.cache_clear()


class ABRExperiment:

    def __init__(self, base_folder):
        self._base_folder = Path(base_folder)
        self._fh = abr.load(base_folder)

    @property
    @functools.lru_cache(maxsize=MAXSIZE)
    def n(self):
        return load_n(self._base_folder)

    def get_info(self, columns, verify_integrity=False, flavor='tuple'):
        info = util.parse_filename(self._base_folder)

        # Check to see if we need to actually load the trial log. If not, keep
        # it simple and fast.
        result = []
        for c in columns:
            # special hack for ordering
            if c == 'ordering':
                ordering = self._fh.erp_metadata[c].iloc[0]
                if ordering == 'interleaved':
                    l1, l2 = self._fh.erp_metadata['level'].iloc[:2]
                    ordering = 'plateau' if l1 == l2 else 'ramp'
                result.append(ordering)
            elif c not in info:
                data = self._fh.erp_metadata[c]
                if verify_integrity and len(np.unique(data)) != 1:
                    m = f'Column {c} has more than one unique value'
                    raise ValueError(m)
                result.append(data.iloc[0])
            else:
                result.append(info[c])

        if flavor == 'tuple':
            return tuple(result)
        elif flavor == 'series':
            return pd.Series(result, index=columns)
        else:
            raise ValueError(f'Unsupported flavor {flavor}')

    @property
    @functools.lru_cache(maxsize=MAXSIZE)
    def frequencies(self):
        return {*self._fh.erp_metadata['frequency']}

    @property
    @functools.lru_cache(maxsize=MAXSIZE)
    def analyzed_data(self):
        return load_analysis(self._base_folder, self.frequencies)

    @property
    @functools.lru_cache(maxsize=MAXSIZE)
    def all_thresholds(self):
        # Need to make sure that *all* frequencies are represented. This helps
        # us catch data for which we do not have the full set analyzed.
        thresholds = self.analyzed_data[0] \
            .unstack('frequency') \
            .reindex(self.frequencies, axis='columns') \
            .stack(dropna=False) \
            .rename('threshold')
        return thresholds.sort_index()

    @property
    @functools.lru_cache(maxsize=MAXSIZE)
    def thresholds(self):
        return self.all_thresholds.groupby('frequency').mean()

    @property
    @functools.lru_cache(maxsize=MAXSIZE)
    def all_waves(self):
        # Need to make sure that *all* frequencies are represented. This helps
        # us catch data for which we do not have the full set analyzed.
        waves = self.analyzed_data[1].copy()
        waves.columns = waves.columns.rename('measure')
        waves = waves.stack() \
            .unstack('frequency') \
            .reindex(self.frequencies, axis='columns') \
            .stack(dropna=False) \
            .unstack('measure')
        waves.index = waves.index.swaplevel('level', 'frequency')
        waves = waves.sort_index()

        gain, = self.get_info(['amplifier_gain'])
        cols = [c for c in waves.columns if 'amplitude' in c]
        cols.extend(['baseline', 'baseline_std'])
        waves[cols] /= gain
        return waves

    @property
    @functools.lru_cache(maxsize=MAXSIZE)
    def waves(self):
        return self.all_waves.groupby(['frequency', 'level']).mean()

    @property
    @functools.lru_cache(maxsize=MAXSIZE)
    def inear_calibration(self):
        csv_file = self._fh.base_path / 'target_microphone_calibration.csv'
        data = pd.read_csv(csv_file, index_col=0).set_index(['frequency'])
        return data[['norm_spl']]

    def get_average_epochs_processed(self, matched=True):
        '''
        Returns epochs as processed for ABR analysis
        '''
        choices = self._fh.base_path.glob('*average waveforms.csv')
        if matched:
            choices = [c for c in choices if 'matched' in str(c)]
        else:
            choices = [c for c in choices if 'matched' not in str(c)]
        if len(choices) != 1:
            raise ValueError('Cannot choose from options')
        data = pd.read_csv(choices[0], header=[0, 1], index_col=0).T
        levels = [l.astype('f') for l in data.index.levels]
        data.index = data.index.set_levels(levels)
        return data

    def _get_epochs(self, fn, apply_reject, limit_averages, *args, **kwargs):
        info = ['amplifier_gain', 'reject_threshold', 'averages']
        gain, reject, averages = self.get_info(info)
        epochs = fn(*args, **kwargs)
        if apply_reject:
            mask = (epochs <= reject).all(axis=1)
            epochs = epochs.loc[mask]
        if limit_averages:
            raise NotImplementedError
        return epochs / gain

    def get_epochs(self, apply_reject=True, limit_averages=True, **kwargs):
        return self._get_epochs(self._fh.get_epochs, apply_reject=apply_reject,
                                limit_averages=limit_averages, **kwargs)

    def get_epochs_filtered(self, apply_reject=True, limit_averages=True,
                            **kwargs):
        return self._get_epochs(self._fh.get_epochs_filtered,
                                apply_reject=apply_reject,
                                limit_averages=limit_averages, **kwargs)

    def get_random_segments(self, *args, **kwargs):
        return self._fh.get_random_segments(*args, **kwargs)

    def get_random_segments_filtered(self, *args, **kwargs):
        return self._fh.get_random_segments_filtered(*args, **kwargs)

    def _get_mean(self, fn, *args, **kwargs):
        columns = ['frequency', 'level', 'polarity']
        epochs = fn(*args, columns=columns, **kwargs)
        reject_threshold, = self.get_info(['reject_threshold'])
        m = np.abs(epochs) <= reject_threshold
        m = m.all(axis=1)
        return epochs.loc[m].groupby(['frequency', 'level', 'polarity']).mean() \
            .groupby(['frequency', 'level']).mean()

    def get_mean_epochs_filtered(self, *args, **kwargs):
        return self._get_mean(self.get_epochs_filtered, *args, **kwargs)

    def get_mean_epochs(self, *args, **kwargs):
        return self._get_mean(self.get_epochs, *args, **kwargs)


MERGE_PATTERN = \
    r'\g<date>-* ' \
    r'\g<experimenter> ' \
    r'\g<animal> ' \
    r'\g<ear> ' \
    r'\g<note> ' \
    r'\g<experiment>*'


PROCESSED_FILE_PATTERN_BASE = \
    r'ABR (?P<start>-?\d+\.\d+)ms ' + \
    r'to (?P<end>-?\d+\.\d+)ms ' + \
    r'(with (?P<filter_lb>\d+)Hz ' + \
    r'to (?P<filter_ub>\d+)Hz filter )?' \
    r'(?P<suffix>[\w\s]+)?'

N_EPOCHS_FILE_PATTERN = PROCESSED_FILE_PATTERN_BASE + \
    'number of epochs.csv'


ABR_ANALYZED_FILE_PATTERN = PROCESSED_FILE_PATTERN_BASE + \
    r'average waveforms' + \
    r'-(?P<frequency>\d+\.\d+)kHz-' + \
    r'((?P<analyzer>\w+)-)?' + \
    r'analyzed.txt'


P_ABR_ANALYZED_FILE_PATTERN = re.compile(ABR_ANALYZED_FILE_PATTERN)
P_N_EPOCHS_FILE_PATTERN = re.compile(N_EPOCHS_FILE_PATTERN)


def load_abr_analysis(filename, parse_filename=True):
    rename = {
        'Level': 'level',
        '1msec Avg': 'baseline',
        '1msec StDev': 'baseline_std',
        'P1 Latency': 'p1_latency',
        'P1 Amplitude': 'p1_amplitude',
        'N1 Latency': 'n1_latency',
        'N1 Amplitude': 'n1_amplitude',
        'P2 Latency': 'p2_latency',
        'P2 Amplitude': 'p2_amplitude',
        'N2 Latency': 'n2_latency',
        'N2 Amplitude': 'n2_amplitude',
        'P3 Latency': 'p3_latency',
        'P3 Amplitude': 'p3_amplitude',
        'N3 Latency': 'n3_latency',
        'N3 Amplitude': 'n3_amplitude',
        'P4 Latency': 'p4_latency',
        'P4 Amplitude': 'p4_amplitude',
        'N4 Latency': 'n4_latency',
        'N4 Amplitude': 'n4_amplitude',
        'P5 Latency': 'p5_latency',
        'P5 Amplitude': 'p5_amplitude',
        'N5 Latency': 'n5_latency',
        'N5 Amplitude': 'n5_amplitude',
    }

    th_match = re.compile('Threshold \(dB SPL\): ([\w.-]+)')
    freq_match = re.compile('Frequency \(kHz\): ([\d.]+)')
    with open(filename) as fh:
        for line in fh:
            # Parse the threshold string
            if line.startswith('Threshold'):
                th_string = th_match.search(line).group(1)
                if th_string == 'None':
                    th = -np.inf
                elif th_string == 'inf':
                    th = np.inf
                elif th_string == '-inf':
                    th = -np.inf
                else:
                    th = float(th_string)

            if line.startswith('Frequency'):
                freq = float(freq_match.search(line).group(1))*1e3

            if line.startswith('NOTE'):
                break

        data = pd.io.parsers.read_csv(fh, sep='\t')
        data.rename(columns=rename, inplace=True)

    if parse_filename:
        base, head = os.path.split(filename)
        info = P_ABR_ANALYZED_FILE_PATTERN.match(head).groupdict()
        if info['analyzer'] is None:
            info['analyzer'] = ''
        info['start'] = float(info['start'])*1e-3
        info['end'] = float(info['end'])*1e-3
        parse = lambda x: float(x) if x is not None else 0
        info['filter_lb'] = parse(info['filter_lb'])
        info['filter_ub'] = parse(info['filter_ub'])
        del info['frequency']
    else:
        info = None

    m = data['level'] >= th
    data = data.loc[m]

    keep_cols = list(rename.values())
    keep = [c for c in data.columns if c in keep_cols]
    data = data[keep] \
        .set_index('level', verify_integrity=True) \
        .sort_index()
    return freq, th, info, data


def load_abr_analysis_for_experiments(abr_experiments, progressbar=True):
    abr_data = []
    filenames = []
    for filename in abr_experiments['filename']:
        search_pattern = os.path.join(filename, '*-analyzed.txt')
        filenames.extend(glob(search_pattern))

    iterator = tqdm(filenames) if progressbar else filenames
    peaks = [load_abr_analysis(filename) for filename in iterator]
    return pd.concat(peaks)
