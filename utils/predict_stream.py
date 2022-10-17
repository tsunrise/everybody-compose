import argparse

import music21
import numpy as np
import preprocess.fetch as fetch
import torch
from models.lstm import DeepBeats
from preprocess.dataset import (generate_sequences,
                                parse_midi_to_input_and_labels)


def write_to_midi(music, filename):
    """
    Write music21 stream to midi file
    """
    music.write('mid', fp=filename)

def convert_to_stream(notes, prev_rest, curr_durs):
    """
    Convert two lists of notes and durations to a music21 stream
    """
    s = music21.stream.Stream()
    for n, r, d in zip(notes, prev_rest, curr_durs):
        if r:
            a = music21.note.Rest()
            a.duration.quarterLength = float(r)
            s.append(a)
        a = music21.note.Note(n)
        a.duration.quarterLength = float(d)
        s.append(a)
    return s

def predict_notes_sequence(durs_seq, model, device):
    """
    Predict notes sequence given durations sequence
    """
    model.to(device)
    model.eval()
    prob_n = model(torch.from_numpy(durs_seq.astype(np.float32)).to(device)) # (1, seq_length, 128)
    prob_n = prob_n.cpu().detach().numpy()
    notes_seq = np.argmax(prob_n, -1) # (1, seq_length)
    notes_seq = notes_seq.squeeze(0) # (seq_length,)
    prev_rest_seq = durs_seq.squeeze(0)[:, 0] # (seq_length,)
    curr_durs_seq = durs_seq.squeeze(0)[:, 1] # (seq_length,)
    return notes_seq, prev_rest_seq, curr_durs_seq


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Save Predicted Notes Sequence to Midi')
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--load_checkpoint', type=str, default="./snapshots/lstm_50.pth")
    parser.add_argument('--midi_filename', type=str, default="./midi_outputs/lstm_50.mid")
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_notes', type=int, default=128)
    parser.add_argument('--seq_len', type=int, default=64)

    main_args = parser.parse_args()

    # sample midi file
    midi_iterator = fetch.midi_iterators()
    midi_file = next(midi_iterator)
    beats, notes = parse_midi_to_input_and_labels(midi_file)
    beats_list, notes_list = [beats], [notes]
    X, y = generate_sequences(beats_list, notes_list, main_args.seq_len)

    # load model
    model = DeepBeats(main_args.n_notes, main_args.embed_dim, main_args.hidden_dim).to(main_args.device)
    if main_args.load_checkpoint:
        model.load_state_dict(torch.load(main_args.load_checkpoint))
    print(model)

    # generate notes seq given durs seq
    notes, prev_rest, curr_durs = predict_notes_sequence(
        durs_seq = X[0][np.newaxis, :].copy(), # select the first durs seq for now, batch size = 1
        model=model,
        device=main_args.device
    )

    # convert stream to midi
    stream = convert_to_stream(notes, prev_rest, curr_durs)
    write_to_midi(stream, main_args.midi_filename)
