import torch
import torch.nn as nn


class DeepBeats(nn.Module):
    def __init__(self, num_notes, embed_size, hidden_dim):
        super(DeepBeats, self).__init__()
        self.durs_embed = nn.Linear(2, embed_size)
        self.layer1 = nn.LSTM(embed_size, hidden_dim, batch_first=True)
        self.layer2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.notes_output = nn.Linear(hidden_dim, num_notes)

    def forward(self, x):
        x = self.durs_embed(x)
        x = self.layer1(x)[0]
        x = self.layer2(x)[0]
        predicted_notes = self.notes_output(x)
        return predicted_notes

    def loss_function(self, pred, target):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, target)
        return loss
