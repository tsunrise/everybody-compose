from typing import IO, Iterable, Tuple
import zipfile

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

def parse_midi_to_melody(midi_file: IO[bytes]):
    """
    Parse a MIDI file into a melody.
    Args:
        midi_file: A MIDI file bytestream.
    Returns:
        Generator as described below, number of notes
    yield:
        Tuple[start_time, end_time, pitch].
    """
    import note_seq.midi_io as midi_io
    import note_seq.melody_inference as melody_inference
    melody_inference.MAX_NUM_FRAMES = 100000
    ns = midi_io.midi_to_note_sequence(midi_file.read())
    with np.errstate(divide='ignore'):
        instrument_id = melody_inference.infer_melody_for_sequence(ns)
    def _gen():
        for note in ns.notes:
            if note.instrument == instrument_id:
                yield note.start_time, note.end_time, note.pitch
    return _gen(), len(ns.notes)

def convert_start_end_to_beats(start_time: np.ndarray, end_time: np.ndarray):
    """
    Convert start time and end time to beats.
    Args:
        start_time: array of shape (seq_length,)
        end_time: array of shape (seq_length,)
    Returns:
        beats: array of shape (seq_length, 2), where the first column is the rest time before current note and the second column is the current duration
    """
    # get the rest time since last beat
    prev_rest = np.zeros_like(start_time)
    prev_rest[1:] = start_time[1:] - end_time[:-1]
    prev_rest[0] = start_time[0]

    # get the duration of the note
    duration = end_time - start_time

    return np.stack([prev_rest, duration], axis=1)

def parse_melody_to_beats_notes(melody: Iterable[Tuple[float, float, int]]) -> Tuple[np.ndarray, np.ndarray]:
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

    start_time = []
    end_time = []
    pitch = []
    for s, e, p in melody:
        start_time.append(s)
        end_time.append(e)
        pitch.append(p)

    start_time = np.array(start_time)
    end_time = np.array(end_time)
    pitch = np.array(pitch)

    beats = convert_start_end_to_beats(start_time, end_time)
    labels = pitch.reshape(-1, 1)

    return beats, labels
