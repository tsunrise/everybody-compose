import torch
import torch.nn as nn


class DeepBeatsTF(nn.Module):
    """
    DeepBeats with Teacher Forcing. This is the same as DeepBeats, the label in previous step is used as input in the next step.
    """
    def __init__(self, num_notes, embed_size, hidden_dim):
        super(DeepBeatsTF, self).__init__()
        self.note_embedding = nn.Embedding(num_notes, embed_size)
        self.layer1 = nn.LSTM(embed_size + 2, hidden_dim, batch_first=True)
        self.layer2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.notes_logits_output = nn.Linear(hidden_dim, num_notes)
        self.num_notes = num_notes

    def forward(self, x, y_prev):
        """
        x: input, shape: (batch_size, seq_len, 2)
        y_prev: label, shape: (batch_size, seq_len), range from 0 to num_notes-1
           y_prev[i] should be the note label for x[i-1], and y[0] is 0.
        """
        
        # Embedding
        y_prev_embed = self.note_embedding(y_prev)
        # concat x and y_prev_embed to be X
        X = torch.cat((x, y_prev_embed), dim=2)
        X = self.layer1(X)[0]
        X = self.layer2(X)[0]
        predicted_notes = self.notes_logits_output(X)
        return predicted_notes

    def sample(self, x):
        raise NotImplementedError("Not implemented yet")

    def loss_function(self, pred, target):
        """
        Pred: (batch_size, seq_len, num_notes), logits
        Target: (batch_size, seq_len), range from 0 to num_notes-1
        """
        criterion = nn.CrossEntropyLoss()
        target_one_hot = torch.nn.functional.one_hot(target, self.num_notes).float()
        loss = criterion(pred, target_one_hot)
        return loss
