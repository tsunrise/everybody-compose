from typing import Optional
import numpy as np
import torch
from models import lstm, bi_lstm, transformer, vanilla_rnn, attention_rnn
import toml
from preprocess.dataset import BeatsRhythmsDataset
import torch.utils.data
import datetime
from torch.utils.tensorboard.writer import SummaryWriter

from utils.data_paths import DataPaths
import models.lstm_local_attn as lstm_local_attn
from utils.metrics import Metrics
CONFIG_PATH = "./config.toml"

def get_model(name, config, device):
    if name == "lstm":
        return lstm.DeepBeatsLSTM(config["n_notes"], config["embed_dim"], config["hidden_dim"]).to(device)
    elif name == "lstm_attn":
        return lstm_local_attn.DeepBeatsLSTMLocalAttn(num_notes=config["n_notes"], hidden_dim=config["hidden_dim"],
         dropout_p=config["dropout_p"]).to(device)
    elif name == "vanilla_rnn":
        return vanilla_rnn.DeepBeatsVanillaRNN(config["n_notes"], config["embed_dim"], config["hidden_dim"]).to(device)
    elif name == "attention_rnn":
        return attention_rnn.DeepBeatsAttentionRNN(config["n_notes"], config["embed_dim"], config["hidden_dim"], config["num_head"]).to(device)
    elif name == "bi_lstm":
        return bi_lstm.DeepBeatsBiLSTM(config["n_notes"], config["embed_dim"], config["hidden_dim"]).to(device)
    elif name == "transformer":
        return transformer.DeepBeatsTransformer(
            num_encoder_layers=config["num_encoder_layers"], 
            num_decoder_layers=config["num_encoder_layers"],
            emb_size=config["embed_dim"],
            nhead= config["num_heads"],
            src_vocab_size=config["src_vocab_size"],
            tgt_vocab_size=config["n_notes"],
            dim_feedforward=config["hidden_dim"]
        ).to(device)
    else:
        raise ValueError("Invalid model name")

def model_forward(model_name, model, input_seq: torch.Tensor, target_seq: torch.Tensor, target_prev_seq: torch.Tensor, device):
    if model_name == "transformer":
        # nn.Transformer takes seq_len * batch_size
        input_seq, target_seq, target_prev_seq = input_seq.permute(1, 0, 2), target_seq.permute(1, 0), target_prev_seq.permute(1, 0)
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = model.create_mask(input_seq, target_prev_seq)
        src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)
        src_padding_mask, tgt_padding_mask = src_padding_mask.to(device), tgt_padding_mask.to(device)
        output = model(input_seq, target_prev_seq, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        output = output.permute(1, 0, 2) # permute back to batch first
    elif model_name == "lstm_attn":
        output = model(input_seq, target_prev_seq)
    else:
        output, _ = model(input_seq, target_prev_seq)
    return output

def model_file_name(model_name, n_files, n_epochs):
    return "{}_{}_{}.pth".format(model_name, "all" if n_files == -1 else n_files, n_epochs)

def save_checkpoint(model, paths, model_name, n_files, n_epochs):
    model_file = model_file_name(model_name, n_files, n_epochs)
    model_path = paths.snapshots_dir / model_file
    torch.save({
        'model': model.state_dict(),
        'n_epochs': n_epochs,
    }, model_path)
    print(f'Checkpoint Saved at {model_path}')

def load_checkpoint(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    n_epochs = checkpoint['n_epochs']
    print(f'Checkpoint Loaded from {checkpoint_path}')
    return n_epochs


def train(model_name: str, n_epochs: int, device: str, n_files:int=-1, snapshots_freq:int=10, checkpoint: Optional[str] = None):
    config = toml.load(CONFIG_PATH)

    global_config = config["global"]
    model_config = config["model"][model_name]

    model = get_model(model_name, model_config, device)
    print(model)

    dataset = BeatsRhythmsDataset(model_config["seq_len"], global_config["random_slice_seed"])
    dataset.load(global_config["dataset"])
    dataset = dataset.subset_remove_short()
    if n_files > 0:
        dataset = dataset.subset(n_files)

    training_data, val_data = dataset.train_val_split(global_config["train_val_split_seed"], global_config["val_ratio"])
    print(f"Training data: {len(training_data)}")
    print(f"Validation data: {len(val_data)}")

    train_loader = torch.utils.data.DataLoader(training_data, batch_size=model_config["batch_size"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=model_config["batch_size"], shuffle=False)

    # checkpoint
    if checkpoint is not None:
        epochs_start = load_checkpoint(checkpoint, model, device)
    else:
        epochs_start = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=model_config["lr"])
    # TODO: we can use a learning rate scheduler here
    paths = DataPaths()
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = paths.tensorboard_dir / "{}_{}/{}".format(model_name, "all" if n_files == -1 else n_files, current_time)
    writer = SummaryWriter(log_dir = log_dir, flush_secs= 60)
    
    best_val_loss = float("inf")
    metrics_train = Metrics("train")
    metrics_val = Metrics("validation")

    for epoch in range(epochs_start, n_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_seq = batch["beats"].to(device)
            target_seq = batch["notes"].long().to(device)
            target_prev_seq = batch["notes_shifted"].long().to(device)
            output = model_forward(model_name, model, input_seq, target_seq, target_prev_seq, device)
            loss = model.loss_function(output, target_seq)
            loss.backward()
            if "clip_grad" in model_config:
                model.clip_gradients_(model_config["clip_grad"])  # type: ignore
            optimizer.step()
            metrics_train.update(len(batch), loss.item(), output, target_seq)
        
        model.eval()
        for batch in val_loader:
            input_seq = batch["beats"].to(device)
            target_seq = batch["notes"].long().to(device)
            target_prev_seq = batch["notes_shifted"].long().to(device)
            with torch.no_grad():
                output = model_forward(model_name, model, input_seq, target_seq, target_prev_seq, device)
                loss = model.loss_function(output, target_seq)
            metrics_val.update(len(batch), loss.item(), output, target_seq)

        training_metrics = metrics_train.flush_and_reset(writer, epoch)
        validation_metrics = metrics_val.flush_and_reset(writer, epoch)
        
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Train Loss: {:.4f}, Val Loss: {:.4f}, Train Acc: {:.4f}, Val Acc: {:.4f}".format(training_metrics["loss"], validation_metrics["loss"], training_metrics["accuracy"], validation_metrics["accuracy"]))

        # save checkpoint with lowest validation loss
        if validation_metrics["loss"] < best_val_loss:
            best_val_loss = validation_metrics["loss"]
            save_checkpoint(model, paths, model_name, n_files, "best")
            print("Minimum Validation Loss of {:.4f} at epoch {}/{}".format(best_val_loss, epoch, n_epochs))
            
        # save snapshots
        if (epoch + 1) % snapshots_freq == 0:
            save_checkpoint(model, paths, model_name, n_files, epoch + 1)
    writer.close()
    save_checkpoint(model, paths, model_name, n_files, n_epochs)
    return model
