import torch
import torch.nn as nn


class DeepBeatsBiLSTM(nn.Module):
    def __init__(self, num_notes, embed_size, hidden_dim):
        super(DeepBeatsBiLSTM, self).__init__()
        self.durs_embed = nn.Linear(2, embed_size)
        self.layer = nn.LSTM(embed_size, hidden_dim, batch_first=True, bidirectional = True, num_layers = 2)
        self.notes_output = nn.Linear(hidden_dim * 2, num_notes)

    def forward(self, x, y_prev):
        x = self.durs_embed(x)
        x, hidden = self.layer(x)
        predicted_notes = self.notes_output(x)
        return predicted_notes, hidden

    def loss_function(self, pred, target):
        criterion = nn.CrossEntropyLoss()
        target = target.flatten() # (batch_size * seq_len)
        pred = pred.reshape(-1, pred.shape[-1]) # (batch_size * seq_len, num_notes)
        loss = criterion(pred, target)
        return loss

    def clip_gradients_(self, max_value):
        torch.nn.utils.clip_grad.clip_grad_value_(self.parameters(), max_value)
