import argparse

import music21
import numpy as np
import preprocess.dataset
import torch
from models.lstm import DeepBeats
from utils.data_paths import DataPaths
from utils.beats_generator import create_beat


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
    parser.add_argument('--load_checkpoint', type=str, default=".project_data/snapshots/lstm_all_10.pth")
    parser.add_argument('--midi_filename', type=str, default="output.mid")
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_notes', type=int, default=128)
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--source', type=str, default="interactive")

    main_args = parser.parse_args()

    paths = DataPaths()

    # sample one midi file
    if main_args.source == 'interactive':
        X = create_beat()
        X[0][0] = 2.
    elif main_args.source == 'dataset':
        dataset = preprocess.dataset.BeatsRhythmsDataset(num_files=1)
        X, _ = next(iter(dataset))
    else:
        with open(main_args.source, 'rb') as f:
            X = np.load(f, allow_pickle=True)


    # load model
    model = DeepBeats(main_args.n_notes, main_args.embed_dim, main_args.hidden_dim).to(main_args.device)
    if main_args.load_checkpoint:
        model.load_state_dict(torch.load(main_args.load_checkpoint))
    print(model)

    # generate notes seq given durs seq
    notes, prev_rest, curr_durs = predict_notes_sequence(
        durs_seq = X[np.newaxis, :].copy(), # select the first durs seq for now, batch size = 1
        model=model,
        device=main_args.device
    )

    # convert stream to midi
    stream = convert_to_stream(notes, prev_rest, curr_durs)
    midi_paths = paths.midi_outputs_dir / main_args.midi_filename
    write_to_midi(stream, midi_paths)
