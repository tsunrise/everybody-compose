import argparse

import numpy as np
import preprocess.dataset
import torch
import toml

from utils.data_paths import DataPaths
from utils.beats_generator import create_beat
from utils.model import CONFIG_PATH, get_model, load_checkpoint
from utils.render import convert_to_melody


def write_to_midi(music, filename):
    """
    Write music21 stream to midi file
    """
    music.write('mid', fp=filename)

def predict_notes_sequence(beats, model, init_note, device, temperature):
    """
    Predict notes sequence given durations sequence
    """
    model.to(device)
    model.eval()
    beats = torch.from_numpy(beats).to(device)
    init_note_t = torch.tensor(init_note, dtype=torch.long).to(device)
    with torch.no_grad():
        notes_seq = model.sample(beats, init_note_t, temperature)
    return notes_seq

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Save Predicted Notes Sequence to Midi')
    parser.add_argument('-m','--model_name', type=str)
    parser.add_argument('-c','--checkpoint_path', type=str)
    parser.add_argument('-o','--midi_filename', type=str, default="output.mid")
    parser.add_argument('-d','--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('-s','--source', type=str, default="interactive")
    parser.add_argument('-i','--init_note', type=int, default=60)
    parser.add_argument('-t','--temperature', type=float, default=0.5)

    main_args = parser.parse_args()
    model_name = main_args.model_name
    checkpoint_path = main_args.checkpoint_path
    midi_filename = main_args.midi_filename
    device = main_args.device
    source = main_args.source
    init_note = main_args.init_note
    temperature = main_args.temperature

    config = toml.load(CONFIG_PATH)
    global_config = config['global']
    model_config = config[main_args.model_name]

    paths = DataPaths()

    # sample one midi file
    if main_args.source == 'interactive':
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
    notes = predict_notes_sequence(
        beats = X.copy(), # select the first durs seq for now, batch size = 1
        model=model,
        init_note=main_args.init_note,
        device=device,
        temperature=main_args.temperature
    )

    # convert stream to midi
    stream = convert_to_melody(X, notes)
    midi_paths = paths.midi_outputs_dir / main_args.midi_filename
    write_to_midi(stream, midi_paths)
