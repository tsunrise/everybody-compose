# render to music21 stream
from note_seq.protobuf.music_pb2 import NoteSequence
import note_seq.midi_io as midi_io
import numpy as np
import torch
def convert_to_melody(beats, notes):
    """
    Convert beats and notes to melody.
    - `beats`: array of shape (seq_length, 2), where the first column is the rest time before current note and the second column is the current duration
    - `notes`: array of shape (seq_length,)
    Returns: Melody Array
    - `start_time`: array of shape (seq_length,)
    - `end_time`: array of shape (seq_length,)
    - `pitch`: array of shape (seq_length,)
    """
    if isinstance(beats, torch.Tensor):
        beats = beats.cpu().numpy()
    if isinstance(notes, torch.Tensor):
        notes = notes.cpu().numpy()
    
    num_notes = beats.shape[0]
    notes = notes.reshape(-1)
    # get the start time of each note
    start_time = np.zeros(num_notes)
    start_time[1:] = np.cumsum(np.sum(beats[:num_notes-1, :], axis=1))
    start_time = start_time + beats[:, 0]

    end_time = start_time + beats[:, 1]
    pitch = notes

    return start_time, end_time, pitch
    
def convert_to_note_seq(beats, notes):
    """
    Convert beats and notes to note_seq.
    - `beats`: array of shape (seq_length, 2), where the first column is the rest time before current note and the second column is the current duration
    - `notes`: array of shape (seq_length,)
    Returns: note_seq
    """
    start_time, end_time, pitch = convert_to_melody(beats, notes)
    seq = NoteSequence()
    seq.tempos.add().qpm = 120 # tempos is irrelevant here
    seq.total_time = end_time[-1]
    for i in range(len(start_time)):
        seq.notes.add(start_time=start_time[i], end_time=end_time[i], pitch=pitch[i], velocity=80) # velocity is irrelevant here
    return seq

def render_midi(beats, notes, midi_path):
    """
    Render beats and notes to MIDI file.
    - `beats`: array of shape (seq_length, 2), where the first column is the rest time before current note and the second column is the current duration
    - `notes`: array of shape (seq_length,)
    - `midi_path`: path to save the MIDI file
    """
    seq = convert_to_note_seq(beats, notes)
    midi_io.note_sequence_to_midi_file(seq, midi_path)