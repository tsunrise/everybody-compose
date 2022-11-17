import torch
import torch.nn as nn
import numpy as np


class DeepBeats_VanillaRNN(nn.Module):
    def __init__(self, num_notes, embed_size, hidden_dim):
        super(DeepBeats_VanillaRNN, self).__init__()
        self.num_notes = num_notes
        self.note_embedding = nn.Embedding(num_notes, embed_size)
        self.layer1 = nn.RNN(embed_size + 2, hidden_dim, batch_first=True)
        self.layer2 = nn.RNN(hidden_dim, hidden_dim, batch_first=True)
        self.notes_output = nn.Linear(hidden_dim, num_notes)

    def _default_init_hidden(self, batch_size):
        device = next(self.parameters()).device
        h1_0 = torch.zeros(1, batch_size, self.layer1.hidden_size).to(device)
        h2_0 = torch.zeros(1, batch_size, self.layer2.hidden_size).to(device)
        return h1_0, h2_0

    def forward(self, x, y_prev, init_hidden = None):
        h1_0, h2_0 = self._default_init_hidden(x.shape[0]) if init_hidden is None else init_hidden
        y_prev_embed = self.note_embedding(y_prev)
        x = torch.cat((x, y_prev_embed), dim=2)
        x, h1 = self.layer1(x, h1_0)
        x, h2 = self.layer2(x, h2_0)
        predicted_notes = self.notes_output(x)
        return predicted_notes, (h1, h2)

    def sample(self, x, y_init=0, temperature=1.0):
        ys = [y_init]
        hidden = self._default_init_hidden(1)
        for i in range(x.shape[0]):
            x_curr = x[i].reshape(1, 1, 2)
            y_prev = ys[-1].reshape(1, 1)
            scores, hidden = self.forward(x_curr, y_prev, hidden)
            scores = scores.squeeze(0)
            scores = scores / temperature
            scores = torch.nn.functional.softmax(scores, dim=1)
            y = torch.multinomial(scores, 1)
            ys.append(y)
        out = [y.item() for y in ys]
        return np.array(out)

    def loss_function(self, pred, target):
        criterion = nn.CrossEntropyLoss()
        target_one_hot = torch.nn.functional.one_hot(target, self.num_notes).float()
        loss = criterion(pred, target_one_hot)
        return loss

    def clip_gradients_(self, max_value):
        torch.nn.utils.clip_grad.clip_grad_value_(self.parameters(), max_value)
