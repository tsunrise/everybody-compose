import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

class RNNEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super(RNNEncoder, self).__init__()
        self.hidden_dim = hidden_dim

        self.fc = nn.Linear(2, hidden_dim)
        self.encoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional = True)

    def forward(self, x):
        x = self.fc(x)
        encode_output, encode_hidden_dim = self.encoder_lstm(x)
        return encode_output, encode_hidden_dim

class Attention(nn.Module):
    def __init__(self, encode_hidden_dim, decode_hidden_dim):
        super(Attention, self).__init__()
        self.encode_hidden_dim = encode_hidden_dim
        self.decode_hidden_dim = decode_hidden_dim

        self.attn1 = nn.Linear(encode_hidden_dim * 2 + decode_hidden_dim, encode_hidden_dim)
        self.attn1_activation = nn.LeakyReLU()
        self.attn2 = nn.Linear(encode_hidden_dim, 1)
        self.attn2_activation = nn.Softmax(dim=2)

    def forward(self, encode_output, hidden_state):
        batch_size, seq_len, _ = encode_output.shape
        hidden_state = hidden_state.permute(1, 0, 2)
        hidden_state = hidden_state.repeat(1, seq_len, 1)
        concat = torch.concat((hidden_state, encode_output), dim=-1)
        attn = self.attn1(concat)
        attn = self.attn1_activation(attn)
        attn = self.attn2(attn)
        attn = self.attn2_activation(attn)
        attn = attn.view(batch_size, 1, seq_len)
        context = torch.bmm(attn, encode_output)
        return context

class RNNDecoder(nn.Module):
    def __init__(self, num_notes, embed_dim, encode_hidden_dim, decode_hidden_dim, dropout_p = 0.1):
        super(RNNDecoder, self).__init__()
        self.decode_hidden_dim = decode_hidden_dim

        self.attention = Attention(encode_hidden_dim, decode_hidden_dim)
        self.note_embedding = nn.Embedding(num_notes, embed_dim)
        self.combine_fc = nn.Linear(encode_hidden_dim * 2 + embed_dim, decode_hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.post_attention_lstm = nn.LSTM(decode_hidden_dim, decode_hidden_dim, batch_first=True)
        self.notes_output = nn.Linear(decode_hidden_dim, num_notes)

    def forward(self, tgt, encode_output, memory=None):
        memory = self._default_init_hidden(tgt.shape[0]) if memory is None else memory
        context = self.attention(encode_output, memory[0])
        tgt = self.note_embedding(tgt)
        tgt = torch.cat((tgt, context), dim=2)
        tgt = self.combine_fc(tgt)
        tgt = F.relu(tgt)
        tgt = self.dropout(tgt)
        tgt, memory = self.post_attention_lstm(tgt, memory)
        tgt = self.notes_output(tgt)
        return tgt, memory

    def _default_init_hidden(self, batch_size):
        device = next(self.parameters()).device
        h = torch.zeros(1, batch_size, self.decode_hidden_dim).to(device)
        c = torch.zeros(1, batch_size, self.decode_hidden_dim).to(device)
        return (h, c)

class DeepBeatsAttentionRNN(nn.Module):
    def __init__(self, num_notes, embed_dim, encode_hidden_dim, decode_hidden_dim, dropout_p = 0.1):
        super(DeepBeatsAttentionRNN, self).__init__()
        self.num_notes = num_notes

        self.encoder = RNNEncoder(encode_hidden_dim)
        self.decoder = RNNDecoder(num_notes, embed_dim, encode_hidden_dim, decode_hidden_dim, dropout_p)

    def forward(self, x, tgt):
        batch_size, seq_len, _ = x.shape
        predicted_notes = torch.zeros(batch_size, seq_len, self.num_notes).to(next(self.parameters()).device)
        encode_output, encode_hidden = self.encoder(x)
        memory = None
        for t in range(seq_len):
            output, memory = self.decoder(tgt[:, t:t + 1], encode_output, memory)
            predicted_notes[:, t:t+1, :] = output
        return predicted_notes

    def loss_function(self, pred, target):
        criterion = nn.CrossEntropyLoss()
        target = target.flatten() # (batch_size * seq_len)
        pred = pred.reshape(-1, pred.shape[-1]) # (batch_size * seq_len, num_notes)
        loss = criterion(pred, target)
        return loss

    def clip_gradients_(self, max_value):
        torch.nn.utils.clip_grad.clip_grad_value_(self.parameters(), max_value)