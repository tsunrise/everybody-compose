import torch
import torch.nn as nn
import numpy as np


class DeepBeatsAttentionRNN(nn.Module):
    def __init__(self, num_notes, embed_dim, hidden_dim):
        super(DeepBeatsAttentionRNN, self).__init__()
        self.num_notes = num_notes
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.encode_lstm = nn.LSTM(2, embed_dim, batch_first=True, bidirectional = True)

        self.note_embedding = nn.Embedding(num_notes, embed_dim)
        self.attn1 = nn.Linear(embed_dim * 2 + hidden_dim, embed_dim)
        self.attn1_activation = nn.LeakyReLU()
        self.attn2 = nn.Linear(embed_dim, 1)
        self.attn2_activation = nn.Softmax(dim = 2)

        self.concat_fc = nn.Linear(embed_dim * 3, embed_dim * 3)
        self.concat_activation = nn.LeakyReLU()
        self.decode_lstm = nn.LSTM(embed_dim * 3, hidden_dim, batch_first=True)
        self.notes_output = nn.Linear(hidden_dim, num_notes)

        self._initializer_weights()

    def _default_init_hidden(self, batch_size):
        device = next(self.parameters()).device
        h1_0 = torch.zeros(2, batch_size, self.encode_lstm.hidden_size).to(device)
        c1_0 = torch.zeros(2, batch_size, self.encode_lstm.hidden_size).to(device)
        h2_0 = torch.zeros(1, batch_size, self.decode_lstm.hidden_size).to(device)
        c2_0 = torch.zeros(1, batch_size, self.decode_lstm.hidden_size).to(device)
        return h1_0, c1_0, h2_0, c2_0

    def _initializer_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y_prev, init_hidden = None):
        """
        x: input, shape: (batch_size, seq_len, 2)
        y_prev: label, shape: (batch_size, seq_len), range from 0 to num_notes-1
           y_prev[i] should be the note label for x[i-1], and y[0] is 0.
        """
        batch_size, seq_len, _ = x.shape
        h1_0, c1_0, h2, c2 = self._default_init_hidden(batch_size) if init_hidden is None else init_hidden
        predicted_notes = torch.zeros(batch_size, seq_len, self.num_notes).to(next(self.parameters()).device)
        encode_output, encode_hidden = self.encode_lstm(x, (h1_0, c1_0))
        for t in range(seq_len):
            output, (h2, c2) = self.decode(encode_output, y_prev[:,t:t+1], (h2, c2))
            predicted_notes[:, t:t+1, :] = output
        return predicted_notes, (h2, c2)

    def compute_attention(self, encode_output, hidden):
        # encode_output shape: (batch_size, seq_len, 2 * embed_dim)
        # hidden: (1, batch_size, hidden_dim)
        batch_size, seq_len, _ = encode_output.shape
        hidden = hidden.permute(1, 0, 2)
        hidden = hidden.repeat(1, seq_len, 1) # shape (batch_size, seq_len, hidden_dim)
        concat = torch.concat((hidden, encode_output), dim = -1) # shape (batch_size, seq_len, hidden_dim + 2*embed_dim)
        attn = self.attn1(concat)
        attn = self.attn1_activation(attn)
        attn = self.attn2(attn)
        attn = self.attn2_activation(attn) # shape (batch_size, seq_len, 1)
        attn = attn.view(batch_size, 1, seq_len)
        context = torch.bmm(attn, encode_output) # shape (batch_size, 1, 2 * embed_dim)
        return context

    def decode(self, encode_output, y_prev, hidden):
        h, c = hidden
        context = self.compute_attention(encode_output, h)
        y_prev_embed = self.note_embedding(y_prev)
        X = torch.concat((context, y_prev_embed), dim = -1)
        X_fc = self.concat_fc(X)
        X_fc = self.concat_activation(X_fc)
        X = X_fc + X
        X, (h, c) = self.decode_lstm(X, (h, c))
        predicted_notes = self.notes_output(X)
        return predicted_notes, (h, c)

    def sample(self, x, y_init=0, temperature=1.0):
        seq_len, _ = x.shape
        x = x.unsqueeze(0)
        h1_0, c1_0, h2, c2 = self._default_init_hidden(1)
        encode_output, encode_hidden = self.encode_lstm(x, (h1_0, c1_0))
        ys = [y_init]
        for t in range(seq_len):
            y_prev = ys[-1].reshape(1,1)
            scores, (h2, c2) = self.decode(encode_output, y_prev, (h2, c2))
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
