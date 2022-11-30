import torch
from torch import nn
from torch.nn import functional as F

class LocalAttnEncoder(nn.Module):
    """
    Local attention encoder.
    """
    def __init__(self, hidden_dim):
        """
        - `duration_fc_dim`: dimension of the fully connected layer for duration
        - `hidden_dim`: dimension of the hidden state of both encoder and decoder
        - `context_dim`: dimension of the context vector
        """
        super(LocalAttnEncoder, self).__init__()
        self.duration_fc = nn.Linear(2, hidden_dim)
        self.encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional = True)

    def forward(self, x):
        """
        - `x`: input sequence, shape: (seq_len, 2)
        Returns:
        - `context`: context vector, shape: (seq_len, context_dim)
        - `encoder_state`: encoder state, shape: (1, hidden_dim). The state only includes left-to-right direction.
        """
        x = self.duration_fc(x)
        x, encoder_state = self.encoder(x)
        return x, (encoder_state[0][:1], encoder_state[1][:1])

class LocalAttnDecoder(nn.Module):
    """
    Local Attention Decoder.
    """
    def __init__(self, hidden_dim, num_notes, dropout_p = 0.1):
        """
        - `note_embed_size`: embedding size of notes
        - `context_dim`: dimension of the context vector
        - `hidden_dim`: dimension of the hidden state of both encoder and decoder
        - `num_notes`: number of notes 
        """
        super(LocalAttnDecoder, self).__init__()
        self.note_embed = nn.Embedding(num_notes, hidden_dim)
        self.combine_fc = nn.Linear(hidden_dim * 3, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.notes_output = nn.Linear(hidden_dim, num_notes)

    def forward(self, tgt, context, memory = None):
        """
        - `tgt`: target sequence, shape: (seq_len, 1)
           tgt[i] is the (i-1)-th note in the sequence
        - `context`: context vector, shape: (seq_len, context_dim)
        - `memory`: encoder state or intermediate state, shape: pair of (1, hidden_dim)
        Returns:
        - `output`: output sequence, shape: (seq_len, num_notes)
                    output[i] is the probability distribution of notes at time step i
        """
        # print(f"{tgt.shape=}, {context.shape=}, {memory[0].shape=}")
        tgt = self.note_embed(tgt)
        tgt = torch.cat((tgt, context), dim=2)
        tgt = self.combine_fc(tgt)
        tgt = F.relu(tgt)
        tgt = self.dropout(tgt)
        tgt, memory = self.rnn(tgt, memory)
        tgt = self.notes_output(tgt)
        return tgt, memory
    
class DeepBeatsLSTMLocalAttn(nn.Module):
    """
    DeepBeats LSTM with encoder-decoder structure with local attention.
    Because the length of output sequence and input sequence is the same and has strong 1-to-1 relationship,
    i-th decoder block only uses i-th encoder block's output as context vector.
    This can improve the performance of the model, from quadratic to linear.
    """
    def __init__(self, num_notes, hidden_dim, dropout_p = 0.1):
        """
        - `num_notes`: number of notes 
        - `duration_fc_dim`: dimension of the fully connected layer for duration
        - `context_dim`: dimension of the context vector
        - `hidden_dim`: dimension of the hidden state of both encoder and decoder
        """
        super(DeepBeatsLSTMLocalAttn, self).__init__()
        self.encoder = LocalAttnEncoder(hidden_dim)
        self.decoder = LocalAttnDecoder(hidden_dim, num_notes, dropout_p)
        self.num_notes = num_notes
    
    def forward(self, x, tgt):
        """
        - `x`: input sequence, shape: (seq_len, 2)
        - `tgt`: target sequence, shape: (seq_len, 1)
           tgt[i] is the (i-1)-th note in the sequence
        Returns:
        - `output`: output sequence, shape: (seq_len, num_notes)
                    output[i] is the probability distribution of notes at time step i
        """
        context, encoder_state = self.encoder(x)
        output, _ = self.decoder(tgt, context, encoder_state)
        return output

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

    # TODO: next step
    # loss: https://github.com/gwinndr/MusicTransformer-Pytorch/blob/master/model/loss.py
    # get accuracy