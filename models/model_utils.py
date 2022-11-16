import torch
from torch import nn

class ConcatPrev(nn.Module):
    def forward(self, x, y_prev):
        """
        x: input, shape: (batch_size, seq_len, 2)
        y_prev: label, shape: (batch_size, seq_len, embedding_dim), 
           y_prev[i] should be the embedding of note label for x[i-1], and y[0] is 0.
        """
        
        # concat x and y_prev_embed to be X
        X = torch.cat((x, y_prev), dim=2)
        return X
