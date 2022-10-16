from typing import IO, List, Tuple, Union
import music21
import numpy as np
import fetch

def _parse_midi_to_notes_durations(midi_file: IO[bytes], mono: bool=True) -> Tuple[List[Union[int, List[int]]], List[float]]:
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
            notes.append(e.midi)
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


def generate_sequences(beats_list: List[np.ndarray], notes_list: List[np.ndarray], seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
        Convert a list of beats and a list of notes into a sequence of training data using sliding window.
        Args:
            beats_list: A list contains numpy arrays of shape (num_notes, 2). Each numpy array represents the beats of one midi file.
            notes_list: A list contains numpy arrays of shape (num_notes, 1). Each numpy array represents the notes in one midi file.
            seq_length: An integer represent the beat sequence length of each training example.
        Returns:
            - A numpy array of shape (num_examples, seq_length, 2). Each row represents a sequence of beats.
            - A numpy array of shape (num_examples, seq_length, 128). Each row represents a sequence of note using one hot encoding,
                which is the expected note sequence that the network should predict given the beat sequence.
                128 represents the range of possible notes in the training data.
    """
    X_beats = []
    y_notes = []
    for beats, notes in zip(beats_list, notes_list):
        for i in range(0, len(notes) - seq_length):
            X_beats.append(beats[i:i + seq_length])
            note_sequence = notes[i:i + seq_length].reshape(-1,)
            y_notes.append(np.eye(128)[note_sequence])
    return np.array(X_beats), np.array(y_notes)

if __name__ == "__main__":
    print("Sanity Check")
    midi_iterator = fetch.midi_iterators()
    midi_file = next(midi_iterator)
    beats, notes = parse_midi_to_input_and_labels(midi_file)
    print(f"Beats ({beats.shape}):\n{beats[0:20]}")
    print(f"Notes ({notes.shape}):\n{notes[0:20]}")
    '''
    beats_list, notes_list = [], []
    for midi_file in midi_iterator:
        beats, notes = parse_midi_to_input_and_labels(midi_file)
        beats_list.append(beats)
        notes_list.append(notes)
    X, y = generate_sequences(beats_list, notes_list, seq_length = 64)
    '''
    
    
    