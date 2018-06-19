import os

import pylab as pl
import numpy as np
import itertools

from lbhb.analysis import rdt

import nems.analysis.api
import nems.fitters.api
import nems.preprocessing
import nems.plots.api
import nems.initializers
import nems.metrics.corrcoef
import nems.analysis.test_prediction
import nems.xforms
import nems.plots.file

from nems_db import db


def do_fit(batch, cell, wcg_n, fir_n, shuffle_phase, shuffle_stream):
    recording = rdt.io.load_recording(batch, cell, True, True)
    modelspec = rdt.modules.create_modelspec(recording, wcg_n, fir_n, False)
    model_name = '_'.join(m['id'] for m in modelspec)

    if shuffle_phase:
        model_name = f'shufflePhase_{model_name}'
        recording['repeating'] = recording['repeating'].shuffle_time()

    if shuffle_stream:
        model_name = f'shuffleStream_{model_name}'
        recording['dual_stream'] = recording['dual_stream'].shuffle_time()

    est, val = rdt.preprocessing.split_est_val_by_sequence(recording)

    prefit_modelspec = nems.initializers.prefit_to_target(
        est,
        modelspec,
        nems.analysis.api.fit_basic,
        target_module='stream_merge',
        fitter=nems.fitters.api.scipy_minimize,
        fit_kwargs={'options': {'ftol': 1e-7, 'maxiter': 1000}}
    )

    final_modelspecs = nems.analysis.api.fit_basic(
        est,
        prefit_modelspec,
        fitter=nems.fitters.api.scipy_minimize,
        fit_kwargs={'options': {'ftol': 1e-9}}
    )

    est_pred, val_pred = nems.analysis.api \
        .generate_prediction(est, val, final_modelspecs)

    final_modelspecs = nems.analysis.api \
        .standard_correlation(est_pred, val_pred, final_modelspecs)

    context = {'est': est_pred, 'val': val_pred, 'modelspecs': final_modelspecs}
    figure = nems.plots.api.quickplot(context, epoch='TRIAL');
    figures = [figure]

    destination = f'/auto/data/nems_db/results/{batch}/{cell}/{model_name}/'
    figurefile = os.path.join(destination, 'figure.0000.png')

    final_modelspecs[0][0]['meta'].update({
        'batch': batch,
        'cellid': cell,
        'modelname': model_name,
        'loader': 'custom',
        'fitkey': 'custom',
        'recording': 'custom',
        'username': 'nems',
        'labgroup': 'lbhb',
        'public': 1,
        'githash': 'TODO',
        'modelpath': destination,
        'figurefile': figurefile,
    })

    if not os.path.exists(destination):
        os.makedirs(destination)

    try:
        gain_figures = rdt.plots.plot_gain(val, final_modelspecs[0], recording.meta['n_targets'])
        for i, f in enumerate(gain_figures):
            filename = os.path.join(destination, f'val_target_{i}.png')
            f.savefig(filename)
    except Exception as e:
        print(e)
        pass

    try:
        gain_figures = rdt.plots.plot_gain(est, final_modelspecs[0], recording.meta['n_targets'])
        for i, f in enumerate(gain_figures):
            filename = os.path.join(destination, f'est_target_{i}.png')
            f.savefig(filename)
    except Exception as e:
        print(e)
        pass

    gain_figures = rdt.plots.plot_gain(recording, final_modelspecs[0], recording.meta['n_targets'])
    for i, f in enumerate(gain_figures):
        filename = os.path.join(destination, f'recording_target_{i}.png')
        f.savefig(filename)

    # TODO Should eventually be handled by save_analysis
    figure_bytes = [nems.plots.file.fig2BytesIO(f) for f in figures]

    nems.xforms.save_analysis(destination, recording,
                              modelspecs=final_modelspecs, xfspec=['custom'],
                              log='None', figures=figure_bytes)

def main():
    import argparse

    qid = os.environ.get('QUEUEID', None)
    parser = argparse.ArgumentParser(description='Fit cell from batch to model')
    parser.add_argument('batch', type=str, help='Batch ID containing data')
    parser.add_argument('cell', type=str, help='Cell ID to fit')
    parser.add_argument('--wcg_n', type=int, help='wcg rank', default=2)
    parser.add_argument('--fir_n', type=int, help='FIR ntaps', default=15)
    #parser.add_argument('--pre-level-shift', action='store_true', help='Apply level shift before RDT gain?')
    parser.add_argument('--shuffle-phase', action='store_true', help='Shuffle phase')
    parser.add_argument('--shuffle-stream', action='store_true', help='Shuffle stream')
    parser.add_argument('model', type=str, help='Model name (ignored)')
    parser.add_argument('queue', type=str, help='Queue ID', nargs='?', default=qid)

    args = parser.parse_args()
    if args.queue is not None:
        db.update_job_start(args.queue)
        nems.utils.progress_fun = db.update_job_tick

    do_fit(args.batch, args.cell, args.wcg_n, args.fir_n, args.shuffle_phase,
           args.shuffle_stream)

    if args.queue is not None:
        db.update_job_complete(args.queue)


if __name__ == '__main__':
    main()
