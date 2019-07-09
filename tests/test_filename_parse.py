from lbhb.io.util import parse_filename


filenames = [
    '20180824-1403 Sean 180412.0 tail right interleaved 20 abr',
    '20180824-1403 Sean 180412.0 right right interleaved 20 abr',
    '20180824-1403 Sean 180412.0 right interleaved 20 abr',
    '20180824-1403 BK 180412 right abr',
    '20180802-1005 BK 1354 left  abr',
    '20180413-122138 Sean 170704.0 4 2nd day reconditioning appetitive_gonogo_food',
    '20180802-1109 Brad M human appetitive_gonogo_food',
]

expected = [
    dict(experimenter='Sean', animal='180412.0 tail', ear='right', note='interleaved 20', experiment='abr'),
    dict(experimenter='Sean', animal='180412.0 right', ear='right', note='interleaved 20', experiment='abr'),
    dict(experimenter='Sean', animal='180412.0', ear='right', note='interleaved 20', experiment='abr'),
    dict(experimenter='BK', animal='180412', ear='right', note=None, experiment='abr'),
    dict(experimenter='BK', animal='1354', ear='left', note=None, experiment='abr'),
    dict(experimenter='Sean', animal='170704.0 4', note='2nd day reconditioning', experiment='appetitive_gonogo_food'),
    dict(experimenter='Brad', animal='M human', note=None, experiment='appetitive_gonogo_food'),
]


def test_filename_parse():
    for f, e in zip(filenames, expected):
        print(f)
        r = parse_filename(f)
        for k, v in e.items():
            assert r[k] == v
