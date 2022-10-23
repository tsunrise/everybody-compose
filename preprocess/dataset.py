from torch.utils.data import IterableDataset

from preprocess.prepare import generate_sequences, prepare_raw_beats_notes

import torch.utils.data
import math
import numpy as np

class BeatsRhythmsDataset(IterableDataset):
    def __init__(self, mono = True, max_files = None, seq_len = 64, save_freq = 128):
        self.beats_files, self.notes_files = prepare_raw_beats_notes(mono, max_files, False, progress_save_freq=save_freq)
        self.seq_len = seq_len

    def __iter__(self):
        # serial version
        # for beats, notes in zip(self.beats_files, self.notes_files):
        #     # here: beats and notes represents beats and notes of one midi file
        #     yield from generate_sequences(beats, notes, self.seq_len)

        # support for multiple workers using static task assignment (not the most efficient one but good enough for now)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            for beats, notes in zip(self.beats_files, self.notes_files):
                # here: beats and notes represents beats and notes of one midi file
                yield from generate_sequences(beats, notes, self.seq_len, one_hot=True)
        else:
            lo, hi = 0, len(self.beats_files)
            per_worker = int(math.ceil((hi - lo) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            worker_lo, worker_hi = lo + worker_id * per_worker, min(lo + (worker_id + 1) * per_worker, hi)
            for idx in range(worker_lo, worker_hi):
                beats, notes = self.beats_files[idx], self.notes_files[idx]
                yield from generate_sequences(beats, notes, self.seq_len, one_hot=True)

def collate_fn(batch):
    X, y = zip(*batch)
    X = np.array(X)
    y = np.array(y)
    return X, y
