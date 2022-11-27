from torch.utils.data import Dataset
from preprocess.constants import DATASETS_CONFIG_PATH
from preprocess.prepare import download_midi_files, parse_melody_to_beats_notes, parse_midi_to_melody

from preprocess.fetch import download

import numpy as np
import toml
import warnings
import pickle
from tqdm import tqdm
import csv
from utils.data_paths import DataPaths
from dataclasses import dataclass

PREPROCESS_SAVE_FREQ = 32

@dataclass
class MetaData:
    canonical_composer: str
    canonical_title: str
    split: str
    midi_filename: str

def _processed_name(dataset: str, dataset_type:str):
    return f"processed_{dataset}_{dataset_type}.pkl"

class BeatsRhythmsDataset(Dataset):
    def __init__(self, seq_len, seed = 12345, initial_note: int = 60):
        self.seq_len = seq_len
        self.beats_list = []
        self.notes_list = []
        self.metadata_list = []
        self.name_to_idx = {}
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.initial_note = initial_note
        self.dataset = ""
        self.dataset_type = ""

    def load(self, dataset = "mastero", mono=True, force_prepare = False):
        assert mono, "Only mono is supported for now"
        paths = DataPaths()
        dataset_type = "mono" if mono else "chords"
        processed_path = paths.prepared_data_dir / _processed_name(dataset, dataset_type)
        progress_path = paths.cache_dir / f"progress_{dataset}_{dataset_type}.pkl"
        self.dataset = dataset
        self.dataset_type = dataset_type

        ## Check if we have processed data
        ### Locally processed data
        if processed_path.exists() and not force_prepare:
            print(f"Found processed data at {processed_path}.")
            with open(processed_path, "rb") as f:
                state_dict = pickle.load(f)
            self.load_processed(state_dict)
            return
        ### Remotely processed data
        config = toml.load(DATASETS_CONFIG_PATH)["datasets"][dataset_type][dataset]
        if "processed" in config and not force_prepare:
            prepared = download(f"processed_{dataset}_{dataset_type}.pkl", config["processed"])
            if prepared is None:
                raise ValueError("Failed to download prepared dataset")
            with open(prepared, "rb") as f:
                state_dict = pickle.load(f)
                self.load_processed(state_dict)
                return
        
        ## Preprocessing
        midi_files, num_files = download_midi_files(dataset, config["midi"])
        metadata_path = download(f"metadata_{dataset}.csv", config["metadata"])
        assert metadata_path is not None, "Failed to download metadata"
        metadata = {}
        with open(metadata_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                metadata[row["midi_filename"]] = MetaData(
                    canonical_composer=row["canonical_composer"],
                    canonical_title=row["canonical_title"],
                    split=row["split"],
                    midi_filename=row["midi_filename"],
                )

        skip = 0
        if progress_path.exists():
            with open(progress_path, "rb") as f:
                state_dict = pickle.load(f)
            self.load_processed(state_dict)
            skip = len(self.metadata_list)
            print(f"Resuming from {skip} files")
        bar = tqdm(total=num_files, desc = "Processing MIDI files")
        warnings_cnt, errors_cnt, saved = 0, 0, 0
        for filename, io in midi_files:
            if skip > 0:
                skip -= 1
                bar.update(1)
                continue
            beats, notes = None, None
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
            try:
                melody, _ = parse_midi_to_melody(io)
                beats, notes = parse_melody_to_beats_notes(melody)
            except Warning:
                warnings_cnt += 1
                bar.set_description(f"Parsing MIDI files ({warnings_cnt} warns, {errors_cnt} errors)", refresh=True)
            except KeyboardInterrupt:
                self.save_processed_to_file(progress_path)
                print(f"KeyboardInterrupt detected, saving progress and exit")
                exit()
            except Exception:
                errors_cnt += 1
                bar.set_description(f"Parsing MIDI files ({warnings_cnt} warns, {errors_cnt} errors)", refresh=True)
            if "truncate" in config:
                filename = filename[len(config["truncate"]):]
            if beats is not None and notes is not None:
                self.beats_list.append(beats)
                self.notes_list.append(notes) 
                self.metadata_list.append(metadata[filename])
                self.name_to_idx[filename] = len(self.metadata_list) - 1
            bar.update(1)
            if len(self.metadata_list) % PREPROCESS_SAVE_FREQ == 0:
                self.save_processed_to_file(progress_path)
                saved = len(self.metadata_list)
            bar.set_postfix(warns=warnings_cnt, errors=errors_cnt, saved=saved)
        bar.close()


    def __len__(self):
        return len(self.metadata_list)

    def __getitem__(self, idx):
        lo = self.rng.integers(0, len(self.beats_list[idx]) - self.seq_len)
        hi = lo + self.seq_len

        beats = self.beats_list[idx][lo:hi]
        notes = self.notes_list[idx][lo:hi]
        # for teacher forcing, we need to shift the notes right by one
        notes_shifted = np.roll(notes, 1)
        if lo == 0:
            notes_shifted[0] = self.initial_note
        else:
            notes_shifted[0] = self.notes_list[idx][lo - 1]
        return {
            "beats": beats.astype(np.float32),
            "notes": notes.astype(np.int32).ravel(),
            "notes_shifted": notes_shifted.astype(np.int32).ravel(),
        }

    def save_processed(self) -> dict:
        return {
            "beats_list": self.beats_list,
            "notes_list": self.notes_list,
            "metadata_list": self.metadata_list,
            "name_to_idx": self.name_to_idx,
        }

    def save_processed_to_file(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.save_processed(), f)

    def save_processed_to_cache(self):
        paths = DataPaths()
        processed_path = paths.prepared_data_dir / _processed_name(self.dataset, self.dataset_type)
        with open(processed_path, "wb") as f:
            pickle.dump(self.save_processed(), f)

    def load_processed(self, state_dict):
        self.beats_list = state_dict["beats_list"]
        self.notes_list = state_dict["notes_list"]
        self.metadata_list = state_dict["metadata_list"]
        self.name_to_idx = state_dict["name_to_idx"]

    def gather(self, indices):
        dataset = BeatsRhythmsDataset(self.seq_len, self.seed, self.initial_note)
        dataset.beats_list = [self.beats_list[i] for i in indices]
        dataset.notes_list = [self.notes_list[i] for i in indices]
        dataset.metadata_list = [self.metadata_list[i] for i in indices]
        dataset.name_to_idx = {v.midi_filename: i for i, v in enumerate(dataset.metadata_list)}
        return dataset

    def subset_remove_short(self):
        """
        Remove short sequences from the dataset
        """
        indices = [i for i in range(len(self)) if len(self.beats_list[i]) >= self.seq_len]
        return self.gather(indices)
    
    def train_val_split(self, seed=0, val_ratio=0.1):
        rng = np.random.default_rng(seed)
        indices = np.arange(len(self.metadata_list))
        rng.shuffle(indices)
        val_size = int(len(self.metadata_list) * val_ratio)
        train_indices = indices[val_size:]
        dev_indices = indices[:val_size]
        return self.gather(train_indices), self.gather(dev_indices)

    def subset(self, max_len):
        rng = np.random.default_rng(0)
        indices = np.arange(len(self.metadata_list))
        rng.shuffle(indices)
        indices = indices[:max_len]
        return self.gather(indices)

    def to_stream(self, idx):
        from utils.render import convert_to_note_seq
        beats = self.beats_list[idx]
        notes = self.notes_list[idx]
        return convert_to_note_seq(beats, notes)

    def to_midi(self, idx, midi_path):
        from note_seq.midi_io import note_sequence_to_midi_file
        stream = self.to_stream(idx)
        note_sequence_to_midi_file(stream, midi_path)


# def collate_fn(batch):
#     X, y, y_prev = zip(*batch)
#     X = np.array(X)
#     y = np.array(y)
#     y_prev = np.array(y_prev)
#     return X, y, y_prev
