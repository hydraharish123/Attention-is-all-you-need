import torch
import torch.nn as nn
from EmbeddingLayer import *

class BiLSTM_Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, embedding_matrix, attention=None):
        super(BiLSTM_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = attention
        self.embedding = EmbeddingLayer(embedding_matrix)

        #BiLSTM
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (h_n, c_n) = self.bilstm(x)
        final_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)

        if self.attention is not None:
            context, attn_weights = self.attention(lstm_out, final_hidden)
            out = self.fc(context)
            return out, attn_weights
        else:
            out = self.fc(final_hidden)
            return out, None
    