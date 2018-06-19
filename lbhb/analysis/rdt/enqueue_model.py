executable_path = '/auto/users/bburan/bin/miniconda3/envs/nems-intel/bin/python'
script_path = '-m lbhb.analysis.rdt.do_fit --wcg_n 2 --fir_n 15 '
script_shuffle_path = '-m lbhb.analysis.rdt.do_fit --shuffle-phase --wcg_n 2 --fir_n 15 '

if __name__ == '__main__':
    from nems_db import db
    for batch in (269, 273):
        print('loading cells')
        cells = db.get_batch_cells(batch=batch)
        print('loaded cells')
        for cell in cells['cellid']:
            db.enqueue_single_model(batch, cell, 'RDT_ha', force_rerun=True,
                                    user='bburan', executable_path=executable_path,
                                    script_path=script_path)
            db.enqueue_single_model(batch, cell, 'RDT_h0', force_rerun=True,
                                    user='bburan', executable_path=executable_path,
                                    script_path=script_shuffle_path)
            print(f'Queued {cell}')
