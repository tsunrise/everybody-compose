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
    # TODO: @yixin
    raise NotImplementedError()



if __name__ == "__main__":
    print("Sanity Check")
    midi_iterator = fetch.midi_iterators()
    midi_file = next(midi_iterator)
    parse_midi_to_input_and_labels(midi_file)
    
    
    