import pandas as pd
import tables
import ast
from glob import glob
import re
import os.path
import datetime as dt
import numpy as np
import tables


EXPERIMENT_FILE_PATTERN = \
    r'(?P<datetime>\d{8}-\d{4}) ' \
    r'(?P<experimenter>\w+) ' \
    r'(?P<animal>.*) ' \
    r'(?P<ear>(right|left)) ' \
    r'((?P<note>[-\(\)\.\s\w]+?) (merged )?)?' \
    r'(?P<experiment>(ABR|DPOAE|HRTF))\.hdf5'


P_EXPERIMENT_FILE = re.compile(EXPERIMENT_FILE_PATTERN)


ANALYZED_FILE_PATTERN = EXPERIMENT_FILE_PATTERN + \
    r'-(?P<frequency>\d+\.\d+)kHz-' + \
    r'((?P<analyzer>\w+)-)?' + \
    r'analyzed.txt'


P_ANALYZED_FILE = re.compile(ANALYZED_FILE_PATTERN)



def load_animal_info():

    def load_google_sheet(key, sheet_id):
        url_template = 'https://docs.google.com/spreadsheets/d/{key}/export' \
            '?&format=csv'
            #'?gid={sheet_id}&format=csv'
        url = url_template.format(key=key, sheet_id=sheet_id)
        return pd.read_csv(url, parse_dates=['exposure date', 'DOB'])

    column_names = {
        'animal': 'animal',
        'DOB': 'DOB',
        'exposure date': 'exposure_date',
        'exposure level': 'exposure_level',
        'S': 'sex',
        'exclude R': 'exclude_right',
        'exclude L': 'exclude_left',
    }

    key = '1NtpLAA9xcJSgHmUCys0dZc4lQcy_7_mICgQ2Rt_MqYA'
    sheet_id = 0
    sheet = load_google_sheet(key, sheet_id)
    sheet['animal'] = sheet.apply(lambda x: '{} {}'.format(x['group'], x['marker']), axis=1)
    sheet['exposure date'] = sheet['exposure date'].map(lambda x: x.date())
    sheet['DOB'] = sheet['DOB'].map(lambda x: x.date())

    sheet['exclude R'] = sheet['exclude R'].map({np.nan: False, 'Y': True})
    sheet['exclude L'] = sheet['exclude L'].map({np.nan: False, 'Y': True})
    sheet.rename(columns=column_names, inplace=True)
    return sheet.loc[:, column_names.values()]


def parse_filename(filename):
    match = P_FILE.match(os.path.basename(filename)).groupdict()
    match['datetime'] = dt.datetime.strptime(match['datetime'], '%Y%m%d-%H%M')
    match['filename'] = filename
    return pd.Series(match)


def _load_analysis(filename, fix_missing_threshold=False):
    rename = {
        'Level': 'level',
        '1msec Avg': 'baseline',
        '1msec StDev': 'baseline_std',
        'P1 Latency': 'p1_latency',
        'P1 Amplitude': 'p1_amplitude',
        'N1 Latency': 'n1_latency',
        'N1 Amplitude': 'n1_amplitude',
        'Frequency': 'frequency',
        'threshold': 'threshold',
    }

    th_match = re.compile('Threshold \(dB SPL\): ([\w.]+)')
    freq_match = re.compile('Frequency \(kHz\): ([\d.]+)')
    with open(filename) as fh:
        text = fh.read()
        th = ast.literal_eval(th_match.search(text).group(1))
        freq = float(freq_match.search(text).group(1))
        for i, l in enumerate(text.split('\n')):
            if l.startswith('Level'):
                break

    info = P_ANALYZED_FILE.match(os.path.basename(filename)).groupdict()
    data = pd.io.parsers.read_csv(filename, sep='\t', skiprows=i)
    data.rename(columns=rename, inplace=True)
    data['frequency'] = freq*1e3

    data['threshold'] = th
    if th is None and fix_missing_threshold:
        data['threshold'] = data['level'].min()

    data = data[list(rename.values())]
    data['analyzer'] = info['analyzer']
    return data


def load_analyzed_abr_data(abr_experiments, fix_missing_thresholds=False):
    abr_data = []
    for _, row in abr_experiments.reset_index().iterrows():
        with tables.open_file(row['filename']) as fh:
            tl = pd.DataFrame(fh.root.trial_log.read())
            for k, v in row.iteritems():
                tl[k] = v

        peaks = []
        for analyzed_filename in glob(row['filename'] + '*analyzed.txt'):
            p = _load_analysis(analyzed_filename, fix_missing_thresholds)
            peaks.append(p)

        peaks = pd.concat(peaks)
        data = pd.merge(tl, peaks, on=['frequency', 'level'], how='outer')
        abr_data.append(data)

    return pd.concat(abr_data)


def merge_cochlear_files(filenames, new_filename):
    if os.path.exists(new_filename):
        raise IOError('Output file already exists')
    with tables.open_file(new_filename, 'w') as fh_new:
        trial_logs = []
        waveforms = []
        fs = []
        for filename in filenames:
            with tables.open_file(filename) as fh:
                trial_logs.append(fh.root.trial_log.read())
                waveforms.append(fh.root.waveforms.read())
                fs.append(fh.root.waveforms._v_attrs.fs)
        if len(np.unique(fs)) != 1:
            mesg = 'Cannot merge data collected with different sampling rates'
            raise ValueError(mesg)
        trial_logs = np.concatenate(trial_logs, axis=0)
        waveforms = np.concatenate(waveforms, axis=0)
        fh_new.create_table('/', 'trial_log', trial_logs)
        w_node = fh_new.create_array('/', 'waveforms', waveforms)
        w_node._v_attrs['fs'] = fs[0]


def load_hrtf_experiments(experiments):
    '''
    Load set of HRTF experiments and tag with informaton stored in dataframe
    '''
    hrtf_data = []
    for _, row in experiments.iterrows():
        hrtf = load_hrtf(row['filename']).reset_index()
        for k, v in row.iteritems():
            if k not in ('filename', 'experiment'):
                hrtf[k] = v
        hrtf_data.append(hrtf)
    return pd.concat(hrtf_data, ignore_index=True)


def load_hrtf(filename):
    '''
    Load single HRTF calibration file and return normalized SPL
    '''
    with tables.open_file(filename) as fh:
        frequency = fh.root.frequency.read()
        signal_psd = 20*np.log10(fh.root.sig_psd.read())
        speaker_psd = fh.root.speaker_spl.read()
        output_gain = fh.root._v_attrs['output_gain']
        freq_ub = fh.root._v_attrs['freq_ub']
        freq_lb = fh.root._v_attrs['freq_lb']
        norm_spl = speaker_psd-signal_psd-output_gain
        series = pd.Series(norm_spl, index=frequency).loc[freq_lb:freq_ub]
        series.index.name = 'frequency'
        series.name = 'normalized_spl'
        return series


def find_experiments(base_path, experiment):
    pattern =  os.path.join(base_path, '*{}.hdf5'.format(experiment))
    info = [parse_filename(f) for f in glob(pattern)]
    return pd.DataFrame(info)


def parse_filename(filename):
    info = P_EXPERIMENT_FILE.match(os.path.basename(filename)).groupdict()
    info['datetime'] = dt.datetime.strptime(info['datetime'], '%Y%m%d-%H%M')
    info['date'] = info['datetime'].date()
    info['filename'] = filename
    return info
