import os
from typing import IO, List, Optional, Tuple, Union
import zipfile

import music21
import numpy as np
import toml
from preprocess.constants import CACHE_DIR, DATASETS_CONFIG_PATH

from preprocess.fetch import download

from tqdm import tqdm

def download_midi_files(max_files = None):
    """Get an iterator over all MIDI files bytestreams.
    
    Returns:
        - `iterator`: An iterator over all MIDI files bytestreams.
        - `num_files`: The number of MIDI files.
    """
    config = toml.load(DATASETS_CONFIG_PATH)
    archives = [download(f"{filename}.zip", item["midi"]) for filename, item in config["datasets"].items()]
    # get number of midi files
    total = 0
    for archive_path in archives:
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            for info in zip_ref.infolist():
                if info.filename.endswith(".mid"):
                    total += 1
    if max_files is not None:
        total = min(total, max_files)
    # iterate over midi files
    def _iter():
        remaining = total
        for archive_path in archives:
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                for info in zip_ref.infolist():
                    if info.filename.endswith(".mid"):
                        yield zip_ref.open(info)
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
    beats, output_notes = np.array(beats), np.array(output_notes).reshape(-1,1)
    return beats, output_notes


def generate_sequences(beats_list: List[np.ndarray], notes_list: List[np.ndarray], seq_length: int, one_hot: bool=True) -> Tuple[np.ndarray, np.ndarray]:
    """
        Convert a list of beats and a list of notes into a sequence of training data using sliding window.
        Args:
            beats_list: A list contains numpy arrays of shape (num_notes, 2). Each numpy array represents the beats of one midi file.
            notes_list: A list contains numpy arrays of shape (num_notes, 1). Each numpy array represents the notes in one midi file.
            seq_length: An integer represent the beat sequence length of each training example.
        Returns:
            - A numpy array of shape (num_examples, seq_length, 2). Each row represents a sequence of beats.
            - A numpy array of shape (num_examples, seq_length, 128) if `one_hot` is True, or (num_examples, seq_length) if `one_hot` is False. Each row represents a sequence of note using one hot encoding,
                which is the expected note sequence that the network should predict given the beat sequence.
                128 represents the range of possible notes in the training data.
    """
    X_beats = []
    y_notes = []
    bar = tqdm(total=len(beats_list), desc="Generating sequences", colour="green")
    for beats, notes in zip(beats_list, notes_list):
        for i in range(0, len(notes) - seq_length):
            X_beats.append(beats[i:i + seq_length])
            note_sequence = notes[i:i + seq_length].reshape(-1,)
            if one_hot:
                y_notes.append(np.eye(128)[note_sequence])
            else:
                y_notes.append(note_sequence)
        bar.update(1)
    bar.close()
    return np.array(X_beats), np.array(y_notes)

def _prepared_file_name(seq_length: int, mono: bool = True, max_files: Optional[int] = None) -> str:
    """Get the name of the prepared file."""
    seq_length_str = f"seq{seq_length}"
    mono_str = "mono" if mono else "chord"
    mx_file_str = f"{max_files}_files" if max_files is not None else "all_files"
    return f"prepared_{seq_length_str}_{mono_str}_{mx_file_str}.npz"

def batch_one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert a batch of labels to one hot encoding."""
    return np.eye(num_classes)[labels]

def prepare_dataset(seq_length: int, mono: bool=True, max_files: Optional[int]=None, override: bool=False):
    """
    Fetch MIDI files and prepare the dataset. If the dataset has already been prepared, it will be loaded from disk.

    Args:
    - `seq_length`: The sequence length of each training example.
    - `mono`: Whether to convert the MIDI files to monophonic.
    - `max_files`: The maximum number of MIDI files to use. If None, all MIDI files will be used.
    - `override`: Whether to override the existing prepared dataset.

    Returns:
    - A numpy array of shape (num_examples, seq_length, 2). Each row represents a sequence of beats.
    - A numpy array of shape (num_examples, seq_length, 128). Each row represents a sequence of note using one hot encoding,
        which is the expected note sequence that the network should predict given the beat sequence.
        128 represents the range of possible notes in the training data.
    """
    if not mono:
        raise NotImplementedError("Polyphonic music is not supported yet.")

    file_name = _prepared_file_name(seq_length, mono, max_files)
    file_path = os.path.join(CACHE_DIR, file_name)
    if os.path.exists(file_path) and not override:
        print("Loading prepared dataset from disk...")
        with np.load(file_path, allow_pickle=True) as data:
            return data["X"], batch_one_hot(data["labels"], 128)
    elif not os.path.exists(CACHE_DIR):
        os.mkdir(CACHE_DIR)

    midi_iterator, count = download_midi_files(max_files)
    beats_list, notes_list = [], []
    bar = tqdm(total=count, desc="Parsing MIDI files", unit="files", colour="blue")
    for midi_file in midi_iterator:
        beats, notes = parse_midi_to_input_and_labels(midi_file)
        beats_list.append(beats)
        notes_list.append(notes)
        bar.update()
    bar.close()
    X, labels = generate_sequences(beats_list, notes_list, seq_length=seq_length, one_hot=False)
    np.savez_compressed(file_path, X=X, labels=labels)
    print(f"Saved prepared dataset to {file_path}")
    return X, batch_one_hot(labels, 128)