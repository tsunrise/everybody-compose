# Everybody Compose: Deep Beats To Music 
Authors: Tom Shen, Violet Yao, Yixin Liu

This repository contains the code for our final project for CS 230: Deep Learning at Stanford University.

## Abstract

This project presents a deep learning approach to generate monophonic melodies based on input beats, allowing even amateurs to create their own music compositions. Three effective methods - LSTM with Full Attention, LSTM with Local Attention, and Transformer with Relative Position Representation - are proposed for this novel task, providing great variation, harmony, and structure in the generated music. This project allows anyone to compose their own music by tapping their keyboards or ``recoloring'' beat sequences from existing works.

## Getting Started

To get started, clone this repository and install the required packages:
```sh
git clone https://github.com/tsunrise/cs230-proj.git
cd cs230-proj
pip -r requirements.txt
```
You may encouter dependency issues during training on `protobuf`. If so, try reinstall `tensorboard` by running:
```sh
pip install --upgrade tensorboard
```
This issue is due to an conflicting requirements of `note_seq` and `tensorboard`.

## Training
The preprocessed dataset will automatically be downloaded before training. To train a model, run the `train.py` script with the `-m` or `--model_name` argument followed by a string specifying the name of the model to use. The available model names are:

- `lstm_attn`: LSTM with Local Attention
- `vanilla_rnn`: Decoder Only Vanilla RNN
- `attention_rnn`: LSTM with Full Attention
- `transformer`: Transformer RPR

You can also use the `-nf` or `--n_files` argument followed by an integer to specify the number of files to use for training (the default value of -1 means that all available files will be used).

To specify the number of epochs to train the model for, use the `-n` or `--n_epochs` argument followed by an integer. The default value is 100.

To specify the device to use for training, use the `-d` or `--device` argument followed by a string. The default value is cuda if a CUDA-enabled GPU is available, or cpu if not.

To specify the frequency at which to save snapshots of the trained model, use the `-s` or `--snapshots_freq` argument followed by an integer. This specifies the number of epochs between each saved snapshot. The default value is 200. The snapshots will be saved in the `.project_data/snapshots` directory. The default value is 200.

To specify a checkpoint to load the model from, use the `-c` or `--checkpoint` argument followed by a string specifying the path to the checkpoint file. The default value is None, which means that no checkpoint will be loaded.

Here are some examples of how to use these arguments:

```sh
# Train the LSTM with Local Attention model using all available files, for 100 epochs, on the default device, saving snapshots every 200 epochs, and not using a checkpoint
python train.py -m lstm_attn

# Train the LSTM with Local Attention model using 10 files, for 1000 epochs, on the CPU, saving snapshots every 100 epochs, and starting from the checkpoint
python train.py -m lstm_attn -nf 10 -n 1000 -d cpu -s 100 -c ./.project_data/snapshots/my_checkpoint.pth

# Train the Transformer RPR model using all available files, for 500 epochs, on the default device, saving snapshots every 50 epochs, and not using a checkpoint
python train.py -m transformer -n 500 -s 50 ./.project_data/snapshots/my_checkpoint.pth
```

## Generating Melodies from Beats

To generate a predicted notes sequence and save it as a MIDI file, run the `predict_stream.py` script with the `-m` or `--model_name` argument followed by a string specifying the name of the model to use. The available model names are:

- `lstm_attn`: LSTM with Local Attention
- `vanilla_rnn`: Decoder Only Vanilla RNN
- `attention_rnn`: LSTM with Full Attention
- `transformer`: Transformer RPR

Use the `-c` or `--checkpoint_path` argument followed by a string 
specifying the path to the checkpoint file to use for the model.

The generated MIDI file will be saved using the filename specified by the `-o` or `--midi_filename` argument (the default value is `output.mid`).

To specify the device to use for generating the predicted sequence, use the `-d` or `--device` argument followed by a string. The default value is `cuda` if a CUDA-enabled GPU is available, or `cpu` if not.

To specify the source of the input beats, use the `-s` or `--source` argument followed by a string. The default value is interactive, which means that the user will be prompted to input the beats using the keyboard. Other possible values are:

- A file path, e.g. `my_input_sequence.txt`, to load the starting sequence from a file.
- `dataset` to use a random sample from the dataset as the starting sequence.

To specify the profile to use for generating the predicted sequence, use the `-t` or `--profile` argument followed by a string. The available values are `beta`, which uses stochastic search, or `beam`, which uses hybrid beam search. The heuristic parameters for these profiles can be customized in the config.toml file by adjusting the corresponding sections in `[sampling.beta]` and `[sampling.beam]`. The default value is default, which uses the settings specified in the `config.toml` file.

Here are some examples of how to use these arguments:

```sh
# Generate a predicted sequence using the LSTM with Local Attention model, from beats by the user using the keyboard, using the checkpoint at ./.project_data/snapshots/my_checkpoint.pth, on the default device, and using the beta profile with default settings
python predict_stream.py -m lstm_attn -c ./.project_data/snapshots/my_checkpoint.pth -t beta
```
