import torch
import torch.nn as nn
import torch.nn.functional as F

'''
 Code adapted from https://github.com/amazon-science/transformer-gan/blob/main/model/discriminator.py
'''

class CNNDiscriminator(nn.Module):
    def __init__(self, num_notes, seq_len, embed_dim, filter_sizes = [2, 3, 4, 5], num_filters = [300, 300, 300, 300], dropout = 0.2):
        super(CNNDiscriminator, self).__init__()
        self.num_notes = num_notes
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.feature_dim = sum(num_filters)

        self.embeddings = nn.Linear(num_notes + 2, embed_dim, bias = False)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, embed_dim)) for (n, f) in zip(num_filters, filter_sizes)
        ])

        self.highway = nn.Linear(self.feature_dim, self.feature_dim)
        self.feature2out = nn.Linear(self.feature_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, beats, notes):
        X = torch.cat((beats, notes), dim = 2)
        X = self.embeddings(X).unsqueeze(1)
        convs = [F.relu(conv(X)).squeeze(3) for conv in self.convs]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]
        pred = torch.cat(pools, 1)
        highway = self.highway(pred)
        highway = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * pred
        pred = self.feature2out(self.dropout(highway))
        pred = torch.sigmoid(pred)
        return pred



