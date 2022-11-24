from typing import IO, Tuple
import zipfile

import music21
import numpy as np

from preprocess.fetch import download

def download_midi_files(dataset: str, midi_url: str):
    """Get an iterator over all MIDI files bytestreams.
    
    Returns:
        - `iterator`: An iterator over all MIDI files bytestreams.
        - `num_files`: The number of MIDI files.
    """
    archive_path = download(f"{dataset}.zip", midi_url)
    if archive_path is None:
        raise RuntimeError("Failed to download the dataset.")
    # get number of midi files
    total = 0
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        for info in zip_ref.infolist():
            if info.filename.endswith(".mid") or info.filename.endswith(".midi"):
                    total += 1
    # iterate over midi files
    def _iter():
        remaining = total
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            for info in zip_ref.infolist():
                if info.filename.endswith(".mid") or info.filename.endswith(".midi"):
                    yield info.filename, zip_ref.open(info)
                    remaining -= 1
                    if remaining == 0:
                        return
    return _iter(), total


def _parse_midi_to_notes_durations(midi_file: IO[bytes], mono: bool=True):
    """Parse a MIDI file into a list of Note and Duration objects.
    Args:
        midi_file: A MIDI file bytestream.
        mono: Whether to convert the MIDI file to monophonic.
    Returns:
        A tuple of (notes, durations) where notes is a list of (integers: if mono==True) or (lists of integers: if mono==False)
        representing the MIDI note numbers and durations is a list of floats
        representing the durations of each note in seconds.
        notes == None means there is a rest with no pitch attached to it.
    """
    parsed = music21.converter.parse(midi_file.read())
    flattened = parsed.chordify().flat
    
    notes = []
    durations = []

    for e in flattened:
        if isinstance(e, music21.chord.Chord):
            if mono:
                # only use highest pitch
                notes.append(max(e.pitches).midi)
            else:
                notes.append([n.midi for n in e.pitches])
            durations.append(e.duration.quarterLength)

        elif isinstance(e, music21.note.Note):
            notes.append(e.pitch.midi)
            durations.append(e.duration.quarterLength)

        elif isinstance(e, music21.note.Rest):
            notes.append(None)
            durations.append(e.duration.quarterLength)
        
    return notes, durations

def parse_midi_to_input_and_labels(midi_file: IO[bytes], mono: bool=True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a MIDI file into a sequence of (prev_rest_time, duration).
    Args:
        midi_file: A MIDI file bytestream.
        mono: Whether to convert the MIDI file to monophonic.
    Returns:
        - A numpy array of shape (num_notes, 2). Each row represents a beat. The first column
          is the rest time since last beat, and the second column is the duration of the note.
        - A numpy array of shape (num_notes, 1). Each row represents a beat. The column
          is the MIDI note number.
    """
    notes, durations = _parse_midi_to_notes_durations(midi_file, mono)
    beats, output_notes = [], []
    prev_rest = 0
    for i in range(len(notes)):
        if notes[i] is None:
            prev_rest += durations[i]
        else:
            beats.append([prev_rest, durations[i]])
            output_notes.append(notes[i])
            prev_rest = 0
    beats, output_notes = np.array(beats).astype(float), np.array(output_notes).reshape(-1,1)
    return beats, output_notes


# def generate_sequences(beats: np.ndarray, notes: np.ndarray, seq_length: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
#     """
#         Convert beats and notes into a sequence of training data using sliding window.
#         Args:
#             - `beats`: numpy array of shape (num_notes, 2). Each numpy array represents the beats of one midi file.
#             - `notes`: numpy array of shape (num_notes, 1). Each numpy array represents the notes in one midi file.
#             - `seq_length`: An integer represent the beat sequence length of each training example.
#         Yields:
#             - X: numpy array of shape (seq_length, 2). Each row represents a sequence of beats.
#             - y: numpy array of shape (seq_length, ), which is the expected note sequence that the network should predict given the beat sequence.
#     """
#     for i in range(0, len(notes) - seq_length):
#         X = beats[i:i + seq_length]
#         y = notes[i:i + seq_length].reshape(-1,)
#         yield X, y

# def generate_sequences_and_shifted(beats: np.ndarray, notes: np.ndarray, seq_length: int, initial_note: int = 0) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
#     """
#         Convert beats and notes into a sequence of training data using sliding window.
#         Args:
#             - `beats`: numpy array of shape (num_notes, 2). Each numpy array represents the beats of one midi file.
#             - `notes`: numpy array of shape (num_notes, 1). Each numpy array represents the notes in one midi file.
#             - `seq_length`: An integer represent the beat sequence length of each training example.
#         Yields:
#             - X: numpy array of shape (seq_length, 2). Each row represents a sequence of beats.
#             - y: numpy array of shape (seq_length, ), which is the expected note sequence that the network should predict given the beat sequence.
#             - y_shifted: numpy array of shape (seq_length, ), where y_shifted[i] = y[i-1], and y_shifted[0] = initial_note if this is the first yield.
#     """
#     notes_shifted = np.roll(notes, 1)
#     notes_shifted[0] = initial_note
#     for i in range(0, len(notes) - seq_length):
#         X = beats[i:i + seq_length]
#         y = notes[i:i + seq_length].reshape(-1,)
#         y_shifted = notes_shifted[i:i + seq_length].reshape(-1,)
#         yield X, y, y_shifted
    

# def _prepared_file_name(mono: bool = True, max_files: int = -1) -> str:
#     """Get the name of the prepared file."""
#     mono_str = "mono" if mono else "chord"
#     mx_file_str = f"{max_files}_files" if max_files != -1 else "all_files"
#     return f"prepared_{mono_str}_{mx_file_str}.pkl"

# def _save_progress(obj, file_name: os.PathLike):
#     """Save the progress of the preparation."""
#     with open(file_name, "wb") as f:
#         pickle.dump(obj, f)

# def _load_progress(file_name: os.PathLike):
#     """Load the progress of the preparation."""
#     with open(file_name, "rb") as f:
#         return pickle.load(f)

# def load_prepared_dataset(file_name: os.PathLike):
#     """Load prepared dataset."""
#     progress =  _load_progress(file_name)
#     return progress["beats_list"], progress["notes_list"]

# def prepare_raw_beats_notes(mono: bool=True, max_files: int = -1, override: bool=False, progress_save_freq: int=100) -> Tuple[List[np.ndarray], List[np.ndarray]]:
#     if not mono:
#         raise NotImplementedError("Polyphonic music is not supported yet.")

#     file_name = _prepared_file_name(mono, max_files)
#     progress_name = f"{file_name}.progress"
#     paths = DataPaths()
#     file_path = paths.prepared_data_dir / file_name
#     progress_path = paths.prepared_data_dir / progress_name
#     if os.path.exists(file_path) and not override:
#         print(f"Found prepared data at {file_path}.")
#         progress = _load_progress(file_path)
#         return progress["beats_list"], progress["notes_list"]

#     midi_iterator, count = download_midi_files(max_files)
#     beats_list, notes_list = [], []
#     warnings_cnt = 0
#     errors_cnt = 0
#     skip = 0
#     if os.path.exists(progress_path):
#         progress = _load_progress(progress_path)
#         beats_list, notes_list = progress["beats_list"], progress["notes_list"]
#         skip = len(beats_list)
#         print("Resuming from previous progress...")

#     bar = tqdm(total=count, desc="Parsing MIDI files", unit="file", colour="blue")
#     for midi_file in midi_iterator:
#         if skip > 0:
#             skip -= 1
#             bar.update(1)
#             continue
#         with warnings.catch_warnings():
#             warnings.filterwarnings("error")
#             try:
#                 beats, notes = parse_midi_to_input_and_labels(midi_file)
#                 beats_list.append(beats)
#                 notes_list.append(notes)
#             except Warning:
#                 warnings_cnt += 1
#                 bar.set_description(f"Parsing MIDI files ({warnings_cnt} warns, {errors_cnt} errors)", refresh=True)
#                 bar.update(1)
#                 continue
#             except KeyboardInterrupt:
#                 print(f"KeyboardInterrupt detected, saving progress and exit")
#                 _save_progress({"beats_list": beats_list, "notes_list": notes_list}, progress_path)
#                 exit()
#             except Exception:
#                 errors_cnt += 1
#                 bar.set_description(f"Parsing MIDI files ({warnings_cnt} warns, {errors_cnt} errors)", refresh=True)
#                 bar.update(1)
#                 continue
#         bar.update(1)
#         if len(beats_list) % progress_save_freq == 0:
#             _save_progress({"beats_list": beats_list, "notes_list": notes_list}, progress_path)

#     bar.close()

#     print(f"Saving prepared dataset to disk...")
#     _save_progress({"beats_list": beats_list, "notes_list": notes_list}, file_path)
#     return beats_list, notes_list