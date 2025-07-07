import torch 
import torch.nn as nn 
import torch.nn.functional as F

class EmbeddingLayer(nn.Module):
    def __init__(self, embedding_matrix):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
    
    def forward(self, x):
        return self.embedding(x)