import os

import pylab as pl
import numpy as np
import itertools
from functools import partial

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
from nems.epoch import epoch_difference, epoch_intersection

from nems_db import db


def do_fit(batch, cell, wcg_n, fir_n, shuffle_phase, shuffle_stream):

    recording = rdt.io.load_recording(batch, cell, True, True)
    est_times, val_times = rdt.preprocessing.split_est_val(recording, False)

    modelspec = rdt.modules.create_modelspec(recording, wcg_n, fir_n, True, 'dual')
    modelspec = nems.priors.set_mean_phi(modelspec)
    model_name = '_'.join(m['id'] for m in modelspec)

    if shuffle_phase:
        model_name = f'shufflePhase_{model_name}'
        recording['repeating'] = recording['repeating'].shuffle_time()

    if shuffle_stream:
        model_name = f'shuffleStream_{model_name}'
        recording = rdt.preprocessing.shuffle_streams(recording)

    strf_kw = f'wcg18x{wcg_n}_fir{wcg_n}x{fir_n}_lvl1_dexp1'
    strf_modelspec = nems.initializers.from_keywords(strf_kw)

    epochs = recording['stim'].epochs

    m_dual = epochs['name'] == 'dual'
    m_repeating = epochs['name'] == 'repeating'
    m_trial = epochs['name'] == 'TRIAL'

    dual_epochs = epochs.loc[m_dual, ['start', 'end']].values
    repeating_epochs = epochs.loc[m_repeating, ['start', 'end']].values
    trial_epochs = epochs.loc[m_trial, ['start', 'end']].values

    # Remove repeating epochs from dual and fit
    prefit_epochs = epoch_difference(trial_epochs, repeating_epochs)
    prefit_epochs = epoch_intersection(est_times, prefit_epochs)
    prefit_recording = recording.select_times(prefit_epochs)

    est_epochs = epoch_intersection(est_times, dual_epochs)
    val_epochs = epoch_intersection(val_times, dual_epochs)
    est_recording = recording.select_times(est_epochs)
    val_recording = recording.select_times(val_epochs)

    prefit_modelspec = nems.initializers.prefit_to_target(
        prefit_recording,
        strf_modelspec,
        nems.analysis.api.fit_basic,
        target_module='levelshift',
        fit_kwargs={'options': {'ftol': 1e-4, 'maxiter': 1000}},
    )

    strf_modelspec, = nems.analysis.api.fit_basic(
        est_recording,
        prefit_modelspec,
        fitter=nems.fitters.api.scipy_minimize,
        fit_kwargs={'options': {'ftol': 1e-8, 'maxiter': 1000}},
    )

    # Copy over fitted phi
    modelspec[0]['phi'] = strf_modelspec[0]['phi'].copy()
    modelspec[1]['phi'] = strf_modelspec[1]['phi'].copy()
    modelspec[3]['phi'] = strf_modelspec[2]['phi'].copy()
    modelspec[4]['phi'] = strf_modelspec[3]['phi'].copy()

    mapper = partial(nems.fitters.mappers.simple_vector, subset=[2])
    final_modelspecs = nems.analysis.api.fit_basic(
        est_recording,
        modelspec,
        fitter=nems.fitters.api.scipy_minimize,
        fit_kwargs={'options': {'ftol': 1e-20, 'maxiter': 5000}},
        mapper=mapper,
    )

    est_pred, val_pred = nems.analysis.api \
        .generate_prediction(est_recording, val_recording, final_modelspecs)

    final_modelspecs = nems.analysis.api \
        .standard_correlation(est_pred, val_pred, final_modelspecs)

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

    nems.xforms.save_analysis(destination, recording,
                              modelspecs=final_modelspecs, xfspec=['custom'],
                              log='None', figures=[])

    context = {'est': est_pred, 'val': val_pred, 'modelspecs': final_modelspecs}
    f = nems.plots.api.quickplot(context);
    f.savefig(os.path.join(destination, 'quickplot.png'))


def main():
    import argparse

    qid = os.environ.get('QUEUEID', None)
    parser = argparse.ArgumentParser(description='Fit cell from batch to model')
    parser.add_argument('batch', type=str, help='Batch ID containing data')
    parser.add_argument('cell', type=str, help='Cell ID to fit')
    parser.add_argument('--wcg_n', type=int, help='wcg rank', default=2)
    parser.add_argument('--fir_n', type=int, help='FIR ntaps', default=15)
    parser.add_argument('--shuffle-phase', action='store_true', help='Shuffle phase')
    parser.add_argument('--shuffle-stream', action='store_true', help='Shuffle stream')
    parser.add_argument('model', type=str, help='Model name (ignored)', nargs='?')

    args = parser.parse_args()
    if qid is not None:
        db.update_job_start(qid)
        nems.utils.progress_fun = db.update_job_tick

    do_fit(args.batch, args.cell, args.wcg_n, args.fir_n, args.shuffle_phase,
           args.shuffle_stream)

    if qid is not None:
        db.update_job_complete(qid)


if __name__ == '__main__':
    main()
