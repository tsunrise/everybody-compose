# render to music21 stream
import note_seq
import numpy as np
import torch
def convert_to_melody(beats, notes):
    """
    Convert two lists of notes and durations to a music21 stream
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

    # get the start time of each note
    start_time = np.zeros(num_notes)
    start_time[1:] = np.cumsum(np.sum(beats[:num_notes-1, :], axis=1))
    start_time = start_time + beats[:, 0]

    end_time = start_time + beats[:, 1]
    pitch = notes

    return start_time, end_time, pitch
    


    return s