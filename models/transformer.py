from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Transformer
import math

"""
Code adapted from: https://pytorch.org/tutorials/beginner/translation_transformer.html
"""

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class LayerNormalization(nn.Module):
    def __init__(self, eps=1e-6, d_model=128):
        super().__init__()
        self.eps = eps
        self.layer = nn.LayerNorm([d_model], elementwise_affine=True, eps=self.eps)

    def forward(self, x):
        x = self.layer(x)
        return 

# Seq2Seq Network
class DeepBeatsTransformer(nn.Transformer):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(DeepBeatsTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.num_notes = tgt_vocab_size
        self.emb_size = emb_size
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = nn.Linear(src_vocab_size, emb_size)
        self.relu = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)
        self.device = 'cpu'
        self.layer_norm = LayerNormalization(d_model=emb_size)
        self.combine_fc = nn.Linear(emb_size * 2, emb_size)
        self._initialize()
        
    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                # nn.init.normal_(p)
    
    def to(self, device):
        super(DeepBeatsTransformer, self).to(device)
        self.device = device   
        return self

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        # src_emb = self.positional_encoding(self.relu(self.src_tok_emb(src)) * math.sqrt(self.emb_size))
        src_emb = self.positional_encoding(self.src_tok_emb(src)) # seq_len, batch, emb_size
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))

        context, memory = self.encode(src, src_mask)
        # print("tgt_emb.shape before concat: ", tgt_emb.shape)
        tgt_emb = torch.cat((tgt_emb, context), dim=2) # seq_len, batch, emb_size * 2
        tgt_emb = F.relu(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)
        # print("tgt_emb.shape after concat: ", tgt_emb.shape)
        tgt_emb = self.combine_fc(tgt_emb)
        # print("tgt_emb.shape after combine fc: ", tgt_emb.shape)
        outs = self.decode(tgt_emb, memory, tgt_mask)
        # outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
        #                         src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        # return self.transformer.encoder(self.relu(self.positional_encoding(
        #                     self.src_tok_emb(src)) * math.sqrt(self.emb_size)), src_mask)
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        memory = self.transformer.encoder(src_emb, src_mask)
        
        return src_emb, memory


    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        # return self.transformer.decoder(self.positional_encoding(
        #                   self.tgt_tok_emb(tgt)), memory,
        #                   tgt_mask)
        return self.transformer.decoder(tgt, memory, tgt_mask)
    
    def loss_function(self, pred, target):
        """
        Pred: (batch_size, seq_len, num_notes), logits
        Target: (batch_size, seq_len), range from 0 to num_notes-1
        """
        criterion = nn.CrossEntropyLoss()
        target = target.flatten() # (batch_size * seq_len)
        pred = pred.reshape(-1, pred.shape[-1]) # (batch_size * seq_len, num_notes)
        loss = criterion(pred, target)
        return loss

    def clip_gradients_(self, max_value):
        torch.nn.utils.clip_grad.clip_grad_value_(self.parameters(), max_value)

    def create_mask(self, src, tgt):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]
        batch_size = src.shape[1]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

        src_padding_mask = torch.zeros((batch_size, src_seq_len)).type(torch.bool)# we don't have padding in our src/tgt
        tgt_padding_mask = torch.zeros((batch_size, tgt_seq_len)).type(torch.bool)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
