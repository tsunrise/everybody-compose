from torch.utils.data import Dataset
from preprocess.constants import DATASETS_CONFIG_PATH
from preprocess.prepare import download_midi_files, parse_midi_to_input_and_labels
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

class BeatsRhythmsDataset(Dataset):
    def __init__(self, seq_len, split_seed = 42, seed = 12345, initial_note: int = 60):
        self.seq_len = seq_len
        self.split_seed = split_seed
        self.beats_list = []
        self.notes_list = []
        self.metadata_list = []
        self.name_to_idx = {}
        self.rng = np.random.default_rng(seed)
        self.initial_note = initial_note

    def load(self, dataset = "mastero", mono=True, force_prepare = False):
        assert mono, "Only mono is supported for now"
        paths = DataPaths()
        dataset_type = "mono" if mono else "chords"
        processed_path = paths.prepared_data_dir / f"processed_{dataset}_{dataset_type}.pkl"
        progress_path = paths.cache_dir / f"progress_{dataset}_{dataset_type}.pkl"

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
        if "prepared" in config and not force_prepare:
            prepared = download(f"prepared_{dataset}_{dataset_type}.pkl", config["prepared"])
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
        warnings_cnt, errors_cnt = 0, 0
        for filename, io in midi_files:
            if skip > 0:
                skip -= 1
                bar.update(1)
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
            try:
                beats, notes = parse_midi_to_input_and_labels(io)
                self.beats_list.append(beats)
                self.notes_list.append(notes) 
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
            self.metadata_list.append(metadata[filename])
            self.name_to_idx[filename] = len(self.metadata_list) - 1
            bar.update(1)
            bar.set_postfix(warns=warnings_cnt, errors=errors_cnt)
            if len(self.metadata_list) % PREPROCESS_SAVE_FREQ == 0:
                self.save_processed_to_file(progress_path)
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
            "beats": beats,
            "notes": notes,
            "notes_shifted": notes_shifted,
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

    def load_processed(self, state_dict):
        self.beats_list = state_dict["beats_list"]
        self.notes_list = state_dict["notes_list"]
        self.metadata_list = state_dict["metadata_list"]
        self.name_to_idx = state_dict["name_to_idx"]

    def generate_midi(self, idx, path = "output.mid"):
        raise NotImplementedError()


# def collate_fn(batch):
#     X, y, y_prev = zip(*batch)
#     X = np.array(X)
#     y = np.array(y)
#     y_prev = np.array(y_prev)
#     return X, y, y_prev
