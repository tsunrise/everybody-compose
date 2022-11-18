import torch
import torch.nn as nn

from models.model_utils import ConcatPrev
import numpy as np

class DeepBeatsLSTM(nn.Module):
    """
    DeepBeats with Teacher Forcing. This is the same as DeepBeats, the label in previous step is used as input in the next step.
    """
    def __init__(self, num_notes, embed_size, hidden_dim):
        super(DeepBeatsLSTM, self).__init__()
        self.note_embedding = nn.Embedding(num_notes, embed_size)
        self.concat_prev = ConcatPrev()
        self.concat_input_fc = nn.Linear(embed_size + 2, embed_size + 2)
        self.concat_input_activation = nn.LeakyReLU()
        self.layer1 = nn.LSTM(embed_size + 2, hidden_dim, batch_first=True)
        self.layer2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.notes_logits_output = nn.Linear(hidden_dim, num_notes)
        self.num_notes = num_notes
        self.hidden_dim = hidden_dim

        self._initializer_weights()

    def _default_init_hidden(self, batch_size):
        device = next(self.parameters()).device
        h1_0 = torch.zeros(1, batch_size, self.layer1.hidden_size).to(device)
        c1_0 = torch.zeros(1, batch_size, self.layer1.hidden_size).to(device)
        h2_0 = torch.zeros(1, batch_size, self.layer2.hidden_size).to(device)
        c2_0 = torch.zeros(1, batch_size, self.layer2.hidden_size).to(device)
        return h1_0, c1_0, h2_0, c2_0

    def _initializer_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y_prev, init_hidden=None):
        """
        x: input, shape: (batch_size, seq_len, 2)
        y_prev: label, shape: (batch_size, seq_len), range from 0 to num_notes-1
           y_prev[i] should be the note label for x[i-1], and y[0] is 0.
        """
        h1_0, c1_0, h2_0, c2_0 = self._default_init_hidden(x.shape[0]) if init_hidden is None else init_hidden
        # Embedding
        y_prev_embed = self.note_embedding(y_prev)
        X = self.concat_prev(x, y_prev_embed)
        # Concat input
        X_fc = self.concat_input_fc(X)
        X_fc = self.concat_input_activation(X_fc)
        # residual connection
        X = X_fc + X

        X, (h1, c1) = self.layer1(X, (h1_0, c1_0))
        X, (h2, c2) = self.layer2(X, (h2_0, c2_0))
        predicted_notes = self.notes_logits_output(X)
        return predicted_notes, (h1, c1, h2, c2)

    def sample(self, x, y_init, temperature=1.0):
        """
        x: input, shape: (seq_len, 2)
        y_init: initial label, shape: (1), range from 0 to num_notes-1

        This function uses a for loop to generate the sequence using LSTMCell, one by one.
        """
        assert self.training == False, "This function should be used in eval mode."
        assert len(x.shape) == 2, "x should be 2D tensor"
        ys = [y_init]
        hidden = self._default_init_hidden(1)
        for i in range(x.shape[0]):
            x_curr = x[i].reshape(1, 1, 2)
            y_prev = ys[-1].reshape(1, 1)
            scores, hidden = self.forward(x_curr,y_prev, hidden)
            scores = scores.squeeze(0)
            scores = scores / temperature
            scores = torch.nn.functional.softmax(scores, dim=1)
            y = torch.multinomial(scores, 1)
            ys.append(y)
        out = [y.item() for y in ys]
        print(out)
        return np.array(out)
        
    def loss_function(self, pred, target):
        """
        Pred: (batch_size, seq_len, num_notes), logits
        Target: (batch_size, seq_len), range from 0 to num_notes-1
        """
        criterion = nn.CrossEntropyLoss()
        target_one_hot = torch.nn.functional.one_hot(target, self.num_notes).float()
        loss = criterion(pred, target_one_hot)
        return loss
    
    def clip_gradients_(self, max_value):
        torch.nn.utils.clip_grad.clip_grad_value_(self.parameters(), max_value)
