import argparse
import os
import datetime

import numpy as np
import torch
import torch.utils.data
from torch.utils.tensorboard.writer import SummaryWriter
from models.lstm_tf import DeepBeatsLSTM
from models.transformer import DeepBeatsTransformer, create_mask

import utils.devices as devices
from models.lstm import DeepBeats
from preprocess.constants import ADL_PIANO_TOTAL_SIZE
from preprocess.dataset import BeatsRhythmsDataset, collate_fn
from utils.data_paths import DataPaths


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

    # initialize mdoel
    if args.model_name == "lstm":
        model = DeepBeats(args.n_notes, args.embed_dim, args.hidden_dim).to(device)
    elif args.model_name == "lstm_tf":
        model = DeepBeatsLSTM(args.n_notes, args.embed_dim, args.hidden_dim).to(device)
    elif args.model_name == "transformer":
        model = DeepBeatsTransformer(
            num_encoder_layers=args.num_encoder_layers, 
            num_decoder_layers=args.num_encoder_layers,
            emb_size=args.embed_dim,
            nhead=args.num_head,
            src_vocab_size=2,
            tgt_vocab_size=args.n_notes,
            dim_feedforward=args.hidden_dim
        ).to(device)
    else:
        raise NotImplementedError("Model {} is not implemented.".format(args.model))
    print(model)

    if args.load_checkpoint:
        filename = paths.snapshots_dir / args.load_checkpoint
        model.load_state_dict(torch.load(filename, map_location=device))
        print(f'Checkpoint loaded {filename}')

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # prepare training/validation loader
    indices = np.arange(args.n_files if args.n_files != -1 else ADL_PIANO_TOTAL_SIZE)
    np.random.seed(0)
    np.random.shuffle(indices)
    # train_size = int(0.8 * len(indices))
    train_size = len(indices)
    train_indices = indices[: train_size]
    val_indices = indices[train_size: ]
    train_dataset = BeatsRhythmsDataset(mono=True, num_files=args.n_files, seq_len=args.seq_len, save_freq=128, max_files_to_parse=args.max_files_to_parse, indices = train_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, collate_fn=collate_fn)
    # val_dataset = BeatsRhythmsDataset(mono=True, num_files=args.n_files, seq_len=args.seq_len, save_freq=128, max_files_to_parse=args.max_files_to_parse, indices = val_indices)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0, collate_fn=collate_fn)

    # define tensorboard writer
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = paths.tensorboard_dir / "{}_{}/{}".format(args.model_name, "all" if args.n_files == -1 else args.n_files, current_time)
    writer = SummaryWriter(log_dir = log_dir, flush_secs= 60)

    # training loop
    # best_val_loss = np.inf
    for epoch in range(args.n_epochs):
        train_batch_loss = 0
        # val_batch_loss = 0
        num_train_batches = 0
        # num_val_batches = 0

        model.train()
        for X, y, y_prev in train_loader:
            optimizer.zero_grad()
            input_seq = torch.from_numpy(X.astype(np.float32)).to(device)
            target_seq = torch.from_numpy(y).long().to(device)
            target_prev_seq = torch.from_numpy(y_prev).long().to(device)
            if args.model_name == "transformer":
                # nn.Transformer takes seq_len * batch_size
                input_seq, target_seq, target_prev_seq = input_seq.permute(1, 0, 2), target_seq.permute(1, 0), target_prev_seq.permute(1, 0)
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = model.create_mask(input_seq, target_prev_seq)
                src_padding_mask, tgt_padding_mask = src_padding_mask.to(device), tgt_padding_mask.to(device)
                output = model(input_seq, target_prev_seq, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
            else:
                output, _ = model(input_seq, target_prev_seq)
            loss = model.loss_function(output, target_seq)
            train_batch_loss += loss.item()
            loss.backward()
            model.clip_gradients_(args.clip_gradients) # TODO: magic number
            optimizer.step()
            num_train_batches += 1
        
        # model.eval()
        # for X, y in val_loader:
        #     input_seq = torch.from_numpy(X.astype(np.float32)).to(device)
        #     target_seq = torch.from_numpy(y.astype(np.float32)).to(device)
        #     with torch.no_grad():
        #         output = model(input_seq)
        #     loss = model.loss_function(output, target_seq)
        #     val_batch_loss += loss.item()
        #     num_val_batches += 1

        avg_train_loss = train_batch_loss / num_train_batches
        # avg_val_loss = val_batch_loss / num_val_batches
        
        print('Epoch: {}/{}.............'.format(epoch, args.n_epochs), end=' ')
        print("Train Loss: {:.4f}".format(avg_train_loss))
        writer.add_scalar("train loss", avg_train_loss, epoch)
        # writer.add_scalar("validation loss", avg_val_loss, epoch)

        # save checkpoint with lowest validation loss
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     save_model(model, paths, args.model_name, args.n_files, "best")
        #     print("Minimum Validation Loss of {:.4f} at epoch {}/{}".format(best_val_loss, epoch, args.n_epochs))
            
        # save snapshots
        if (epoch + 1) % args.snapshots_freq == 0:
            save_model(model, paths, args.model_name, args.n_files, epoch + 1)

    # save model
    save_model(model, paths, args.model_name, args.n_files, args.n_epochs)
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train DeepBeats')
    parser.add_argument('--model_name', type=str, default="lstm_tf")
    parser.add_argument('--load_checkpoint', type=str, default="", help="load checkpoint path to continue training")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--n_notes', type=int, default=128)
    parser.add_argument('--num_encoder_layers', type=int, default=3)
    parser.add_argument('--num_decoder_layers', type=int, default=3)
    parser.add_argument('--num_head', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--clip_gradients', type=float, default=5.0)
    parser.add_argument('--n_files', type=int, default=-1,
                        help='number of midi files to use for training')
    parser.add_argument('--snapshots_freq', type=int, default=10)
    parser.add_argument('--max_files_to_parse', type=int, default=-1, help='max number of midi files to parse')

    main_args = parser.parse_args()
    train(main_args)
