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
from nems.epoch import epoch_difference

from nems_db import db



def do_fit(batch, cell, wcg_n, fir_n, dexp, shuffle_phase, shuffle_stream,
           balance_phase, two_step, mode):

    recording = rdt.io.load_recording(batch, cell, True, True)
    modelspec = rdt.modules.create_modelspec(recording, wcg_n, fir_n, dexp, mode)
    model_name = '_'.join(m['id'] for m in modelspec)

    if balance_phase:
        model_name = f'balancePhase_{model_name}'

    if shuffle_phase:
        model_name = f'shufflePhase_{model_name}'
        recording['repeating'] = recording['repeating'].shuffle_time()

    if shuffle_stream:
        model_name = f'shuffleStream_{model_name}'
        recording['dual_stream'] = recording['dual_stream'].shuffle_time()

    est, val = rdt.preprocessing.split_est_val(recording, balance_phase)

    if two_step:
        if not dexp:
            raise ValueError('Cannot fit two step without dexp')

        model_name = f'twoStep_{model_name}'

        # Extract epochs for first step of fit
        epochs = recording['stim'].epochs
        m = epochs['name'] == 'repeating'
        repeating_epochs = epochs.loc[m, ['start', 'end']].values
        m = epochs['name'] == 'TRIAL'
        trial_epochs = epochs.loc[m, ['start', 'end']].values
        fit_epochs = epoch_difference(trial_epochs, repeating_epochs)

        ms = f'wcg18x{wcg_n}_fir{wcg_n}x{fir_n}_lvl1_dexp1'
        strf_modelspec = nems.initializers.from_keywords(ms)

        initial_est = est.select_times(fit_epochs)

        step1_modelspec = nems.initializers.prefit_to_target(
            initial_est,
            strf_modelspec,
            nems.analysis.api.fit_basic,
            target_module='levelshift',
            fit_kwargs={'options': {'ftol': 1e-9, 'maxiter': 1000}},
        )

        step2_modelspec, = nems.analysis.api.fit_basic(
            est,
            step1_modelspec,
            fitter=nems.fitters.api.scipy_minimize,
            fit_kwargs={'options': {'ftol': 1e-10}},
        )

        # Copy over fitted phi
        modelspec[0]['phi'] = step2_modelspec[0]['phi'].copy()
        modelspec[1]['phi'] = step2_modelspec[1]['phi'].copy()
        modelspec[3]['phi'] = step2_modelspec[2]['phi'].copy()
        modelspec[4]['phi'] = step2_modelspec[3]['phi'].copy()

        mapper = partial(nems.fitters.mappers.simple_vector, subset=[2])
        final_modelspecs = nems.analysis.api.fit_basic(
            recording,
            modelspec,
            fitter=nems.fitters.api.scipy_minimize,
            fit_kwargs={'options': {'ftol': 1e-12, 'gtol': 1e-12}},
            mapper=mapper,
        )

    else:
        # Single fit
        if dexp:
            prefit_modelspec = nems.initializers.prefit_to_target(
                est,
                modelspec,
                nems.analysis.api.fit_basic,
                target_module='stream_merge',
                fitter=nems.fitters.api.scipy_minimize,
                fit_kwargs={'options': {'ftol': 1e-7, 'maxiter': 1000}}
            )
        else:
            prefit_modelspec = modelspec

        final_modelspecs = nems.analysis.api.fit_basic(
            est,
            prefit_modelspec,
            fitter=nems.fitters.api.scipy_minimize,
            fit_kwargs={'options': {'ftol': 1e-12, 'gtol': 1e-12}}
        )

    est_pred, val_pred = nems.analysis.api \
        .generate_prediction(est, val, final_modelspecs)

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
    parser.add_argument('--balance-phase', action='store_true', help='Balance targets across phase')
    parser.add_argument('--dexp', action='store_true', help='Use dexp?')
    parser.add_argument('--two-step', action='store_true', help='Two step fit?')
    parser.add_argument('--mode', type=str, default='dual', help='Fit mode')
    parser.add_argument('model', type=str, help='Model name (ignored)', nargs='?')

    args = parser.parse_args()
    if qid is not None:
        db.update_job_start(qid)
        nems.utils.progress_fun = db.update_job_tick

    do_fit(args.batch, args.cell, args.wcg_n, args.fir_n, args.dexp,
           args.shuffle_phase, args.shuffle_stream, args.balance_phase,
           args.two_step, args.mode)

    if qid is not None:
        db.update_job_complete(qid)


if __name__ == '__main__':
    main()
