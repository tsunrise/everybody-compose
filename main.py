import argparse
import os

import numpy as np
import torch
from preprocess.dataset import BeatsRhythmsDataset, collate_fn
from utils.data_paths import DataPaths

import utils.devices as devices
import torch.utils.data
from models.lstm import DeepBeats

import tqdm

def model_file_name(model_name, n_files, n_epochs):
    return "{}_{}_{}.pth".format(model_name, "all" if n_files == -1 else n_files, n_epochs)

def train(args):
    # check cuda status
    devices.status_check()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = DataPaths()
    print(f"Using {device} device")

    # initialize mdoel
    model = DeepBeats(args.n_notes, args.embed_dim, args.hidden_dim).to(device)
    print(model)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # prepare training data
    dataset = BeatsRhythmsDataset(mono=True, num_files=args.n_files, seq_len=args.seq_len, save_freq=128, max_files_to_parse=args.max_files_to_parse)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=0, collate_fn=collate_fn)
    # training loop
    bar = tqdm.tqdm(total=args.n_epochs)
    for epochs in range(args.n_epochs):
        batch_loss = 0
        num_batches = 0
        for X, y in dataloader:
            optimizer.zero_grad()
            input_seq = torch.from_numpy(X.astype(np.float32)).to(device)
            target_seq = torch.from_numpy(y.astype(np.float32)).to(device)
            output = model(input_seq)
            loss = model.loss_function(output, target_seq)
            batch_loss += loss.item()
            loss.backward()
            optimizer.step()
            num_batches += 1
        bar.update(1)
        bar.set_description("Loss: {:.4f}".format(batch_loss / num_batches))
        # save snapshots
        if (epochs + 1) % args.snapshots_freq == 0:
            model_file = model_file_name(args.model_name, args.n_files, epochs + 1)
            torch.save(model.state_dict(), paths.snapshots_dir / model_file)

    bar.close()
    # save model
    
    model_file = model_file_name(args.model_name, args.n_files, args.n_epochs)
    model_path = paths.snapshots_dir / model_file
    torch.save(model.state_dict(), model_path)
    print(f'Model Saved at {model_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train DeepBeats')
    parser.add_argument('--model_name', type=str, default="lstm")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--n_notes', type=int, default=128)
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--n_files', type=int, default=-1,
                        help='number of midi files to use for training')
    parser.add_argument('--snapshots_freq', type=int, default=10)
    parser.add_argument('--max_files_to_parse', type=int, default=-1, help='max number of midi files to parse')

    main_args = parser.parse_args()
    train(main_args)
