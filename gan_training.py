import argparse
import datetime
import toml

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard.writer import SummaryWriter
from models.cnn_discriminator import CNNDiscriminator

from preprocess.dataset import BeatsRhythmsDataset
from utils.data_paths import DataPaths
from utils.model import get_model, load_checkpoint, save_checkpoint, model_forward

'''
    Code adapted from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
'''
CONFIG_PATH = "./config.toml"

def train(generator_name: str, discriminator_name: str, n_epochs: int, device: str, n_files:int=-1, snapshots_freq:int=10, generator_checkpoint: Optional[str] = None, discriminator_checkpoint: Optional[str] = None):
    # check cuda status
    print(f"Using {device} device")

    # initialize model
    config = toml.load(CONFIG_PATH)
    global_config = config["global"]
    netG_config = config[generator_name]
    netG = get_model(generator_name, netG_config, device)
    print(netG)
    netD_config = config[discriminator_name]
    netD = CNNDiscriminator(netG_config["n_notes"], netG_config["seq_len"], netD_config["embed_dim"]).to(device)
    print(netD)

    if generator_checkpoint:
        load_checkpoint(generator_checkpoint, netG, device)
    if discriminator_checkpoint:
        load_checkpoint(discriminator_checkpoint, netD, device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.
    criterion = nn.BCELoss()

    # define optimizer
    optimizerD = torch.optim.Adam(netD.parameters(), lr=netG_config["lr"])
    optimizerG = torch.optim.Adam(netG.parameters(), lr=netD_config["lr"])

    # prepare training/validation loader
    dataset = BeatsRhythmsDataset(netG_config["seq_len"], global_config["random_slice_seed"], global_config["initial_note"])
    dataset.load(global_config["dataset"])
    dataset = dataset.subset_remove_short()
    if n_files > 0:
        dataset = dataset.subset(n_files)

    training_data, val_data = dataset.train_val_split(global_config["train_val_split_seed"], 0)
    print(f"Training data: {len(training_data)}")

    train_loader = torch.utils.data.DataLoader(training_data, netG_config["batch_size"], shuffle=True)

    # define tensorboard writer
    paths = DataPaths()
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = paths.tensorboard_dir / "{}_{}/{}".format("GAN", "all" if n_files == -1 else n_files,
                                                        current_time)
    writer = SummaryWriter(log_dir=log_dir, flush_secs=60)

    # training loop
    netD.train()
    netG.train()
    for epoch in range(n_epochs):
        num_train_batches = 0
        for batch in train_loader:
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            input_seq = batch["beats"].to(device)
            target_seq = batch["notes"].long().to(device)
            target_prev_seq = batch["notes_shifted"].long().to(device)
            label = torch.full((input_seq.shape[0], 1), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            target_one_hot = torch.nn.functional.one_hot(target_seq, netG_config['n_notes']).float()
            output = netD(input_seq, target_one_hot)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate fake image batch with G
            fake_logits = model_forward(generator_name, netG, input_seq, target_seq, target_prev_seq, device)
            fake = F.gumbel_softmax(fake_logits)
            if generator_name == "transformer":
                fake = fake.transpose(0,1)
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
              % (epoch, n_epochs, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        writer.add_scalar("Generator loss", errG.item(), epoch)
        writer.add_scalar("Discriminator loss", errD.item(), epoch)

        if (epoch + 1) % snapshots_freq == 0:
            save_checkpoint(netG, paths, generator_name, n_files, epoch + 1)
            save_checkpoint(netD, paths, discriminator_name, n_files, epoch + 1)

    # save model
    save_checkpoint(netG, paths, generator_name, n_files, epoch + 1)
    save_checkpoint(netD, paths, discriminator_name, n_files, epoch + 1)
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train DeepBeats GAN')
    parser.add_argument('-gm', '--generator_name', type=str, default = "lstm")
    parser.add_argument('-dm', '--discriminator_name', type=str, default = "cnn")
    parser.add_argument('-nf', '--n_files', type=int, default=-1)
    parser.add_argument('-n', '--n_epochs', type=int, default=100)
    parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('-s', '--snapshots_freq', type=int, default=50)
    parser.add_argument('-gc', '--generator_checkpoint', type=str, default=None)
    parser.add_argument('-dc', '--discriminator_checkpoint', type=str, default=None)

    main_args = parser.parse_args()
    train(**vars(main_args))
