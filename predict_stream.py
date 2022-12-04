import argparse

import numpy as np
import preprocess.dataset
import torch
import toml
from utils.constants import NOTE_MAP

from utils.data_paths import DataPaths
from utils.model import CONFIG_PATH, get_model, load_checkpoint
from utils.render import render_midi
from utils.sample import beam_search, stochastic_search

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Save Predicted Notes Sequence to Midi')
    parser.add_argument('-m','--model_name', type=str)
    parser.add_argument('-c','--checkpoint_path', type=str)
    parser.add_argument('-o','--midi_filename', type=str, default="output.mid")
    parser.add_argument('-d','--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('-s','--source', type=str, default="interactive")
    parser.add_argument('-t','--profile', type=str, default="default")

    main_args = parser.parse_args()
    model_name = main_args.model_name
    checkpoint_path = main_args.checkpoint_path
    midi_filename = main_args.midi_filename
    device = main_args.device
    source = main_args.source
    profile = main_args.profile

    config = toml.load(CONFIG_PATH)
    global_config = config['global']
    model_config = config["model"][main_args.model_name]

    paths = DataPaths()

    # sample one midi file
    if main_args.source == 'interactive':
        from utils.beats_generator import create_beat
        X = create_beat()
        X[0][0] = 2.
        # convert to float32
        X = np.array(X, dtype=np.float32) 
    elif main_args.source == 'dataset':
        dataset = preprocess.dataset.BeatsRhythmsDataset(64) # not used
        dataset.load(global_config['dataset'])
        idx = np.random.randint(0, len(dataset))
        X = dataset.beats_list[idx][:64]
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
    profile = config["sampling"][profile]
    try:
        hint = [NOTE_MAP[h] for h in profile["hint"]]
    except KeyError:
        print(f"some note in {profile['hint']} not found in NOTE_MAP")
        exit(1)
    if profile["strategy"] == "stochastic":
        notes = stochastic_search(model, X, hint, device, profile["top_p"], profile["top_k"], profile["repeat_decay"], profile["temperature"])
    elif profile["strategy"] == "beam":
        notes = beam_search(model, X, hint, device, profile["repeat_decay"], profile["num_beams"], profile["beam_prob"], profile["temperature"])
    else:
        raise NotImplementedError(f"strategy {profile['strategy']} not implemented")
    print(notes)
    # convert stream to midi
    midi_paths = paths.midi_outputs_dir / main_args.midi_filename
    render_midi(X, notes, midi_paths)