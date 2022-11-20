import argparse
import os
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard.writer import SummaryWriter
from models.lstm_tf import DeepBeatsLSTM
from models.cnn_discriminator import CNNDiscriminator

import utils.devices as devices
from preprocess.constants import ADL_PIANO_TOTAL_SIZE
from preprocess.dataset import BeatsRhythmsDataset, collate_fn
from utils.data_paths import DataPaths

'''
    Code adapted from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
'''

def model_file_name(model_name, n_files, n_epochs):
    return "{}_{}_{}.pth".format(model_name, "all" if n_files == -1 else n_files, n_epochs)

def save_model(model, paths, model_name, n_files, n_epochs):
    model_file = model_file_name(model_name, n_files, n_epochs)
    model_path = paths.snapshots_dir / model_file
    torch.save(model.state_dict(), model_path)
    print(f'Model Saved at {model_path}')

def train(args):
    # check cuda status
    devices.status_check()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = DataPaths()
    print(f"Using {device} device")

    # initialize model
    netG = DeepBeatsLSTM(args.n_notes, args.embed_dim, args.hidden_dim).to(device)
    print(netG)
    netD = CNNDiscriminator(args.n_notes, args.seq_len, args.embed_dim).to(device)
    print(netD)

    if args.load_checkpoint_G:
        filename = paths.snapshots_dir / args.load_checkpoint_G
        netG.load_state_dict(torch.load(filename))
        print(f'Generator Checkpoint loaded {filename}')
    if args.load_checkpoint_D:
        filename = paths.snapshots_dir / args.load_checkpoint_D
        netD.load_state_dict(torch.load(filename))
        print(f'Discriminator Checkpoint loaded {filename}')

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.
    criterion = nn.BCELoss()

    # define optimizer
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.learning_rate)
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.learning_rate)

    # prepare training/validation loader
    indices = np.arange(args.n_files if args.n_files != -1 else ADL_PIANO_TOTAL_SIZE)
    np.random.seed(0)
    np.random.shuffle(indices)
    train_size = len(indices)
    train_indices = indices[: train_size]
    train_dataset = BeatsRhythmsDataset(mono=True, num_files=args.n_files, seq_len=args.seq_len, save_freq=128,
                                        max_files_to_parse=args.max_files_to_parse, indices=train_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0,
                                               collate_fn=collate_fn)

    # define tensorboard writer
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = paths.tensorboard_dir / "{}_{}/{}".format("GAN", "all" if args.n_files == -1 else args.n_files,
                                                        current_time)
    writer = SummaryWriter(log_dir=log_dir, flush_secs=60)

    # training loop
    netD.train()
    netG.train()
    for epoch in range(args.n_epochs):
        num_train_batches = 0
        for X, y, y_prev in train_loader:
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            label = torch.full((X.shape[0], 1), real_label, dtype=torch.float, device=device)
            input_seq = torch.from_numpy(X.astype(np.float32)).to(device)
            target_seq = torch.from_numpy(y).long().to(device)
            target_prev_seq = torch.from_numpy(y_prev).long().to(device)

            # Forward pass real batch through D
            target_one_hot = torch.nn.functional.one_hot(target_seq, args.n_notes).float()
            output = netD(input_seq, target_one_hot)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate fake image batch with G
            fake, _ = netG(input_seq, target_prev_seq)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(input_seq, fake.detach())
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(input_seq, fake)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            netG.clip_gradients_(5)
            optimizerG.step()

            num_train_batches += 1

        # Output training stats
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
              % (epoch, args.n_epochs, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        writer.add_scalar("Generator loss", errG.item(), epoch)
        writer.add_scalar("Discriminator loss", errD.item(), epoch)

        if (epoch + 1) % args.snapshots_freq == 0:
            save_model(netG, paths, args.generator_name, args.n_files, epoch + 1)
            save_model(netD, paths, args.discriminator_name, args.n_files, epoch + 1)

    # save model
    save_model(netG, paths, args.generator_name, args.n_files, epoch + 1)
    save_model(netD, paths, args.discriminator_name, args.n_files, epoch + 1)
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train DeepBeats')
    parser.add_argument('--generator_name', type=str, default="lstm_tf")
    parser.add_argument('--discriminator_name', type=str, default="cnn")
    parser.add_argument('--load_checkpoint_G', type=str, default="", help="load checkpoint path for generator")
    parser.add_argument('--load_checkpoint_D', type=str, default="", help="load checkpoint path for discriminator")
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
