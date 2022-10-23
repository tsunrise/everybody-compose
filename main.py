import argparse
import os

import numpy as np
import torch
from preprocess.prepare import batch_one_hot, prepare_dataset

import utils.devices as devices
from models.lstm import DeepBeats


def train(args):
    # check cuda status
    devices.status_check()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # initialize mdoel
    model = DeepBeats(args.n_notes, args.embed_dim, args.hidden_dim).to(device)
    print(model)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # prepare training data
    X, y = prepare_dataset(args.seq_len)
    y = batch_one_hot(y, args.n_notes)

    # training loop
    for epoch in range(1, args.n_epochs + 1):
        num_batches = X.shape[0] // args.batch_size + 1
        batch_loss = 0
        for i in range(num_batches):
            optimizer.zero_grad()
            input_seq = torch.from_numpy(X[i: i + args.batch_size, :, :].astype(np.float32)).to(device)
            target_seq = torch.from_numpy(y[i: i + args.batch_size, :, :].astype(np.float32)).to(device)
            output = model(input_seq)
            loss = model.loss_function(output, target_seq)
            batch_loss += loss.item()
            loss.backward()
            optimizer.step()
        print('Epoch: {}/{}.............'.format(epoch, args.n_epochs), end=' ')
        print("Loss: {:.4f} ".format(batch_loss / num_batches))

    # save model
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    torch.save(model.state_dict(), f'./{args.model_dir}/{args.model_name}_{epoch}.pth') 
    print(f'Model Saved at ./{args.model_dir}/{args.model_name}_{epoch}.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train DeepBeats')
    parser.add_argument('--model_name', type=str, default="vanilla_rnn")
    parser.add_argument('--model_dir', type=str, default="snapshots")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--n_notes', type=int, default=128)
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--mini_scale', default=False,
                        help='run mini scale training (10 midi files) for sanity check')

    main_args = parser.parse_args()
    train(main_args)
