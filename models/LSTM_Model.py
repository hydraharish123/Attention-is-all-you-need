import torch
import torch.nn as nn
from EmbeddingLayer import *

class LSTM_Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, embedding_matrix, attention=None):
        super(LSTM_Model, self).__init__()
        self.attention = attention
        self.hidden_dim = hidden_dim
        self.embedding = EmbeddingLayer(embedding_matrix)
        ## LSTM
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        final_hidden = h_n[-1] ## capture the last hidden state values 
        ## now pass these values to attention if its mentioned else pass it to FC layer

        if self.attention is not None:
            context, attn_weights = self.attention(lstm_out, final_hidden)
            out = self.fc(context)
            return out, attn_weights
        else: 
            out = self.fc(final_hidden)
            return out, None
        