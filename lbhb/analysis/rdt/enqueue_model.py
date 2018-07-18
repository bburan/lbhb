from nems_db import db
import hashlib


def generate_modelname(wcg_n, fir_n, shuffle_phase, shuffle_stream):
    model_name = f'RDTwcg18x{wcg_n}_RDTfir{wcg_n}x{fir_n}_RDTstreamgain_lvl1_dexp'

    if shuffle_phase:
        model_name = f'shufflePhase_{model_name}'

    if shuffle_stream:
        model_name = f'shuffleStream_{model_name}'

    return model_name


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Fit cell from batch to model')
    parser.add_argument('--wcg_n', type=int, help='wcg rank', default=2)
    parser.add_argument('--fir_n', type=int, help='FIR ntaps', default=15)
    args = parser.parse_args()

    template = '-m lbhb.analysis.rdt.do_fit --wcg_n {wcg_n} --fir_n {fir_n}'
    executable_path = '/auto/users/bburan/bin/miniconda3/envs/nems-intel/bin/python'


    for shuffle_phase in (True, False):
        for shuffle_stream in (True, False):
            modelname = generate_modelname(args.wcg_n, args.fir_n,
                                           shuffle_phase, shuffle_stream)

            script_path = template.format(wcg_n=args.wcg_n, fir_n=args.fir_n)
            if shuffle_phase:
                script_path += ' --shuffle-phase'
            if shuffle_stream:
                script_path += ' --shuffle-stream'

            modelname = hashlib.sha1(modelname.encode('ascii')).hexdigest()

            for batch in (269, 273):
                for cell in db.get_batch_cells(batch=batch)['cellid']:
                    db.enqueue_single_model(cell, batch, modelname,
                                            force_rerun=True, user='bburan',
                                            executable_path=executable_path,
                                            script_path=script_path)
                    print(f'Queued {cell}')


if __name__ == '__main__':
    main()
