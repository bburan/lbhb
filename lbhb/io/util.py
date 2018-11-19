from glob import glob
import re
import shutil
import os.path
import datetime as dt

import bcolz
import numpy as np
import pandas as pd


def fix_object_columns(rootdir):
    '''
    Fix object columns in a ctable

    Since object columns significantly slow down loading of ctables, we want to
    convert them to a native type if possible. Due to a possible bug in bcolz,
    the code simplys loads the ctable into a dataframe, converts the columns,
    deletes the original table and recreates it from the dataframe.
    '''
    table = bcolz.ctable(rootdir=rootdir)
    df = table.todataframe()
    modified = False

    # When reading into a dataframe, unicode columns are converted to object
    # columns. We need to use the ctable dtype descriptor, not the DataFrame
    # dtype descriptor to determine whether we need to perform a conversion.
    for column_name in table.cols:
        if table.dtype[column_name].type is np.bytes_:
            pass
        elif table.dtype[column_name].type is np.object_:
            pass
        else:
            continue

        modified = True
        value = df[column_name].iloc[0]
        new_type = type(value)
        if new_type == bytes:
            df[column_name] = df[column_name].str.decode('utf8')
        else:
            df[column_name] = df[column_name].astype(new_type)

    if modified:
        shutil.rmtree(rootdir)
        bcolz.ctable.fromdataframe(df, rootdir=rootdir)

    return df


EAR_EXPERIMENTS = [
    'abr',
    'dpoae',
    'HRTF',
    'calibration',
]


ANIMAL_EXPERIMENTS = [
    'appetitive_gonogo_food',
]


DATETIME_PATTERN = r'(?P<datetime>\d{8}-\d{4,6}) '


EAR_FILE_PATTERN = \
    DATETIME_PATTERN + \
    r'(?P<experimenter>\w+) ' \
    r'(?P<animal>.*) ' \
    r'(?P<ear>(right|left))\s+' \
    r'((?P<note>[-\(\)\.\s\w]+?) (merged )?)?' + \
    r'(?P<experiment>({}))(\.hdf5)?'.format('|'.join(EAR_EXPERIMENTS))


# This is a bit tricky since there's ambiguity in the animal filenames. Need to
# come up with a better filename format.
# r'(?P<animal>[\.\w\s]+) '
ANIMAL_FILE_PATTERN = \
    DATETIME_PATTERN + \
    r'(?P<experimenter>\w+) ' \
    r'(?P<animal>[\.\w]+\s+\w+) ' \
    r'((?P<note>[-\(\)\.\s\w]+) (merged )?)?' + \
    r'(?P<experiment>({}))?'.format('|'.join(ANIMAL_EXPERIMENTS))


ABR_ANALYZED_FILE_PATTERN = EAR_FILE_PATTERN + \
    r'-(?P<frequency>\d+\.\d+)kHz-' + \
    r'((?P<analyzer>\w+)-)?' + \
    r'analyzed.txt'


P_EAR_FILE = re.compile(EAR_FILE_PATTERN, flags=re.IGNORECASE)
P_ANIMAL_FILE = re.compile(ANIMAL_FILE_PATTERN, flags=re.IGNORECASE)
P_ABR_ANALYZED_FILE = re.compile(ABR_ANALYZED_FILE_PATTERN, flags=re.IGNORECASE)


def parse_filename(filename, include_filename=True):
    basename = os.path.basename(filename)
    for pattern in (P_ABR_ANALYZED_FILE, P_EAR_FILE, P_ANIMAL_FILE):
        match = pattern.match(basename)
        if match is not None:
            break
    else:
        raise ValueError(f'Unrecognized filename {basename}')

    info = match.groupdict()
    dt_format = '%Y%m%d-%H%M' if len(info['datetime']) == 13 else '%Y%m%d-%H%M%S'
    info['datetime'] = dt.datetime.strptime(info['datetime'], dt_format)
    info['date'] = info['datetime'].date()
    if include_filename:
        info['filename'] = filename
    return info


def find_experiments(base_path, experiment):
    pattern =  os.path.join(base_path, '*{}*'.format(experiment))
    info = [parse_filename(f) for f in glob(pattern)]
    return pd.DataFrame(info)
