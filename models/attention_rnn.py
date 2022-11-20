import torch
import torch.nn as nn
import numpy as np
from models.model_utils import ConcatPrev


class DeepBeats_AttentionRNN(nn.Module):
    def __init__(self, num_notes, embed_size, hidden_dim, num_head = 8):
        super(DeepBeats_AttentionRNN, self).__init__()
        self.num_notes = num_notes
        self.embed_size = embed_size
        self.hidden_dim = hidden_dim

        self.encode_rnn = nn.RNN(2, hidden_dim, num_layers=2, batch_first=True, bidirectional = True)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_head, batch_first=True)
        self.encode_fc = nn.Linear(hidden_dim * 2, embed_size)

        self.note_embedding = nn.Embedding(num_notes, embed_size)
        self.concat_prev = ConcatPrev()
        self.concat_input_fc = nn.Linear(embed_size * 2, embed_size * 2)
        self.concat_input_activation = nn.LeakyReLU()
        self.decode_rnn = nn.RNN(embed_size * 2, hidden_dim, num_layers=2, batch_first=True)
        self.notes_output = nn.Linear(hidden_dim, num_notes)

        self._initializer_weights()

    def forward(self, x, y_prev):
        a, encode_hidden = self.encode(x)
        predicted_notes, decode_hidden = self.decode(encode_hidden, a, y_prev)
        return predicted_notes, decode_hidden

    def encode(self, x):
        x, encode_hidden = self.encode_rnn(x)
        a = self.attention(query = x, key = x, value = x)[0]
        a = self.encode_fc(a)
        h1 = torch.sum(encode_hidden[:2], 0, keepdim = True)
        h2 = torch.sum(encode_hidden[2:], 0, keepdim = True)
        encode_hidden = torch.cat((h1, h2), 0)
        return a, encode_hidden

    def decode(self, hidden, a, y_prev):
        y_prev_embed = self.note_embedding(y_prev)
        X = self.concat_prev(a, y_prev_embed)
        X_fc = self.concat_input_fc(X)
        X_fc = self.concat_input_activation(X_fc)
        # residual connection
        X = X_fc + X
        X, decode_hidden = self.decode_rnn(X, hidden)
        predicted_notes = self.notes_output(X)
        return predicted_notes, decode_hidden

    def _initializer_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def sample(self, x, y_init=0, temperature=1.0):
        a, hidden = self.encode(x)
        hidden = hidden.unsqueeze(1)
        ys = [y_init]
        for i in range(a.shape[0]):
            a_curr = a[i].reshape(1, 1, -1)
            y_prev = ys[-1].reshape(1, 1)
            scores, hidden = self.decode(hidden, a_curr, y_prev)
            scores = scores.squeeze(0)
            scores = scores / temperature
            scores = torch.nn.functional.softmax(scores, dim=1)
            y_next = None
            while y_next is None or y_next == ys[-1]:
                top10 = torch.topk(scores, 3, dim=1)
                indices, probs = top10.indices, top10.values
                probs = probs / torch.sum(probs)
                probs_idx = torch.multinomial(probs, 1)
                y_next = indices[0, probs_idx]
            ys.append(y_next)
        out = [y.item() for y in ys[1:]]
        return np.array(out)

    def loss_function(self, pred, target):
        criterion = nn.CrossEntropyLoss()
        target_one_hot = torch.nn.functional.one_hot(target, self.num_notes).float()
        loss = criterion(pred, target_one_hot)
        return loss

    def clip_gradients_(self, max_value):
        torch.nn.utils.clip_grad.clip_grad_value_(self.parameters(), max_value)
