from nems_db import db
import hashlib


def generate_modelname(wcg_n, fir_n, dexp, shuffle_phase, shuffle_stream,
                       balance_phase, two_step_fit, mode):
    mode = 'global' if mode == 'single' else 'stream'
    model_name = f'RDTwcg18x{wcg_n}_RDTfir{wcg_n}x{fir_n}_RDT{mode}gain_lvl1'

    if dexp:
        model_name = f'{model_name}_dexp1'

    if balance_phase:
        model_name = f'balancePhase_{model_name}'

    if shuffle_phase:
        model_name = f'shufflePhase_{model_name}'

    if shuffle_stream:
        model_name = f'shuffleStream_{model_name}'

    if two_step_fit:
        model_name = f'twoStep_{model_name}'

    return model_name


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Fit cell from batch to model')
    parser.add_argument('--wcg_n', type=int, help='wcg rank', default=2)
    parser.add_argument('--fir_n', type=int, help='FIR ntaps', default=15)
    #parser.add_argument('--shuffle-phase', action='store_true', help='Shuffle phase')
    #parser.add_argument('--shuffle-stream', action='store_true', help='Shuffle stream')
    parser.add_argument('--balance-phase', action='store_true', help='Balance targets across phase')
    parser.add_argument('--dexp', action='store_true', help='Use dexp?')
    parser.add_argument('--two-step', action='store_true', help='Two step fit?')
    #parser.add_argument('--mode', type=str, default='dual', help='Fit mode')
    args = parser.parse_args()

    template = '-m lbhb.analysis.rdt.do_fit --wcg_n {wcg_n} --fir_n {fir_n} --mode {mode}'
    executable_path = '/auto/users/bburan/bin/miniconda3/envs/nems-intel/bin/python'


    for shuffle_phase in (True, False):
        for shuffle_stream in (True, False):
            for mode in ('single', 'dual'):
                modelname = generate_modelname(args.wcg_n, args.fir_n,
                                               args.dexp, shuffle_phase,
                                               shuffle_stream,
                                               args.balance_phase,
                                               args.two_step, mode)

                script_path = template.format(wcg_n=args.wcg_n, fir_n=args.fir_n, mode=mode)
                if shuffle_phase:
                    script_path += ' --shuffle-phase'
                if shuffle_stream:
                    script_path += ' --shuffle-stream'
                if args.balance_phase:
                    script_path += ' --balance-phase'
                if args.dexp:
                    script_path += ' --dexp'
                if args.two_step:
                    script_path += ' --two-step'

                modelname = hashlib.sha1(modelname.encode('ascii')).hexdigest()

                for batch in (269, 273):
                    for cell in db.get_batch_cells(batch=batch)['cellid']:
                        db.enqueue_single_model(batch, cell, modelname,
                                                force_rerun=True, user='bburan',
                                                executable_path=executable_path,
                                                script_path=script_path)
                        print(f'Queued {cell}')


if __name__ == '__main__':
    main()
