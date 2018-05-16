import tables as tb
import pandas as pd
import numpy as np
from scipy import signal


class ABRFile:

    def __init__(self, filename):
        self._filename = filename
        self._fh = tb.open_file(filename)
        self.trial_log = pd.DataFrame(self._fh.root.trial_log[:])
        self.fs = self._fh.root.waveforms._v_attrs['fs']

    def get_epochs(self, trial_filter, signal_filter=None):
        masks = [self.trial_log[k] == v for k, v in trial_filter.items()]
        masks = np.all(masks, axis=0)
        waveforms = []
        for i in np.flatnonzero(masks):
            w = self._fh.root.waveforms[i]
            waveforms.append(w)

        waveforms =  np.concatenate(waveforms, axis=0)

        if signal_filter is not None:
            filter_lb, filter_ub = signal_filter
            Wn = filter_lb/self.fs, filter_ub/self.fs
            b, a = signal.iirfilter(N=2, Wn=Wn, btype='band', ftype='butter')
            waveforms = signal.filtfilt(b, a, waveforms, axis=1)
        return waveforms
