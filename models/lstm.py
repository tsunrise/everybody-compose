import torch
import torch.nn as nn

from preprocess.dataset import BeatsRhythmsDataset
from torch.utils.data import DataLoader

class DeepBeats(nn.Module):
    def __init__(self, num_notes, embed_size, hidden_dim):
        super(DeepBeats, self).__init__()
        self.durs_embed = nn.Linear(2, embed_size)
        self.layer1 = nn.LSTM(embed_size, hidden_dim, batch_first=True)
        self.layer2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.notes_output = nn.Linear(hidden_dim, num_notes)
        self.num_notes = num_notes

    def forward(self, x, y_prev):
        _ = y_prev # unused
        x = self.durs_embed(x)
        x = self.layer1(x)[0]
        x = self.layer2(x)[0]
        predicted_notes = self.notes_output(x)
        return predicted_notes

    def sample(self, x):
        return self.forward(x, None)

    def loss_function(self, pred, target):
        """
        Pred: (batch_size, seq_len, num_notes), logits
        Target: (batch_size, seq_len), range from 0 to num_notes-1
        """
        criterion = nn.CrossEntropyLoss()
        target_one_hot = torch.nn.functional.one_hot(target, self.num_notes).float()
        loss = criterion(pred, target_one_hot)
        return loss