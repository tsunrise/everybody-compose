import argparse

import music21
import numpy as np
from models.lstm import DeepBeatsLSTM
import preprocess.dataset
import torch
import toml

from utils.data_paths import DataPaths
from utils.beats_generator import create_beat
from utils.model import CONFIG_PATH, get_model, load_checkpoint
from utils.sample import greedy_search


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
            # TODO: quarterLength is not necessarily the same as duration
            # quarterLength=1 is a whole note, quarterLength=0.25 is a quarter note
            # each note takes 4 seconds when tempo=60
            # we might also need to quantize this
            a.duration.quarterLength = float(r * 2) # TODO: hack
            s.append(a)
        a = music21.note.Note(n)
        a.duration.quarterLength = float(d * 2) # TODO: hack
        s.append(a)
    return s

def predict_notes_sequence(durs_seq, model, device, profile):
    """
    Predict notes sequence given durations sequence
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        if profile["strategy"] == "greedy":
            notes_seq = greedy_search(model, durs_seq, device, profile["top_p"], profile["repeat_decay"], profile["init_note"], profile["temperature"])
        else:
            raise NotImplementedError("Strategy {} not implemented".format(profile["strategy"]))
    prev_rest_seq = durs_seq[:, 0] # (seq_length,)
    curr_durs_seq = durs_seq[:, 1] # (seq_length,)

    return notes_seq, prev_rest_seq, curr_durs_seq

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Save Predicted Notes Sequence to Midi')
    parser.add_argument('-m','--model_name', type=str)
    parser.add_argument('-c','--checkpoint_path', type=str)
    parser.add_argument('-o','--midi_filename', type=str, default="output.mid")
    parser.add_argument('-d','--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('-s','--source', type=str, default="interactive", help="interactive, dataset, or path to recorded beats")
    parser.add_argument('--take', type=int, default=256, help="for dataset only: number of beats to take from dataset")
    parser.add_argument('-p','--profile', type=str, default="default", help="sampling profile name in config.toml")

    main_args = parser.parse_args()
    model_name = main_args.model_name
    checkpoint_path = main_args.checkpoint_path
    midi_filename = main_args.midi_filename
    device = main_args.device
    source = main_args.source
    beats_to_take = main_args.take
    profile = main_args.profile

    config = toml.load(CONFIG_PATH)
    global_config = config['global']
    model_config = config["model"][main_args.model_name]

    paths = DataPaths()

    # sample one midi file
    if main_args.source == 'interactive':
        X = create_beat()
        X[0][0] = 2.
        # convert to float32
        X = np.array(X, dtype=np.float32) 
    elif main_args.source == 'dataset':
        dataset = preprocess.dataset.BeatsRhythmsDataset(64) # not used
        dataset.load(global_config["dataset"])
        idx = np.random.randint(0, len(dataset))
        X = dataset.beats_list[idx][:beats_to_take]
    else:
        with open(main_args.source, 'rb') as f:
            X = np.load(f, allow_pickle=True)
            X[0][0] = 2.
    X = np.array(X, dtype=np.float32)

    # load model

    model = get_model(main_args.model_name, model_config, main_args.device)
    model.eval()
    load_checkpoint(checkpoint_path, model, device)
    print(model)

    # generate notes seq given durs seq
    notes, prev_rest, curr_durs = predict_notes_sequence(
        durs_seq = X.copy(), # select the first durs seq for now, batch size = 1
        model=model,
        device=device,
        profile=config["sampling"][profile]
    )

    # convert stream to midi
    stream = convert_to_stream(notes, prev_rest, curr_durs)
    midi_paths = paths.midi_outputs_dir / main_args.midi_filename
    write_to_midi(stream, midi_paths)
