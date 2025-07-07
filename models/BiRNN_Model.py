import torch
import torch.nn as nn
from EmbeddingLayer import *

class BiRNN_Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, embedding_matrix, attention=None):
        super(BiRNN_Model, self).__init__()
        self.attention = attention
        self.hidden_dim = hidden_dim
        self.embedding = EmbeddingLayer(embedding_matrix)

        # Bidirectional RNN
        self.birnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        rnn_out, hidden = self.birnn(x)
        
        final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)

        if self.attention is not None:
            context, attn_weights = self.attention(rnn_out, final_hidden)
            out = self.fc(context)
            return out, attn_weights
        else:
            out = self.fc(final_hidden)
            return out, None
        