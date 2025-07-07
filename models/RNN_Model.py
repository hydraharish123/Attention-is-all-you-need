import torch
import torch.nn as nn
from EmbeddingLayer import *

class RNN_Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, embedding_matrix, attention=None):
        super(RNN_Model, self).__init__()
        self.attention = attention
        self.hidden_dim = hidden_dim
        self.embedding = EmbeddingLayer(embedding_matrix)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        rnn_out, hidden = self.rnn(x)  # hidden shape: (1, batch_size, hidden_dim)
        final_hidden = hidden[-1] # shape: (batch_size, hidden_dim)
        if self.attention is not None:
            context, attn_weights = self.attention(rnn_out, final_hidden)
            out = self.fc(context)
            return out, attn_weights
        else: 
            out = self.fc(final_hidden)
            return out, None
