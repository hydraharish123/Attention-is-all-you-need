import torch 
import torch.nn as nn 
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, attention_dim):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(in_features=encoder_hidden_dim, out_features=attention_dim)
        self.W2 = nn.Linear(in_features=decoder_hidden_dim, out_features=attention_dim)
        self.V = nn.Linear(attention_dim, 1)

    def forward(self, encoder_outputs, decoder_hidden):
        '''
            encoder_outputs ----> shape (batch, input_len, encoder_hidden_dim)
            decoder_hidden  ----> shape (batch, decoder_hidden_dim)
        '''

        decoder_hidden = decoder_hidden.unsqueeze(1) # decoder_hidden_shape ---> (batch, 1, decoder_hidden_dim)

        score = F.tanh(self.W1(encoder_outputs) + self.W2(decoder_hidden))
        # score shape ---> (batch, input_len, attention_dim)

        energy = self.V(score).squeeze(-1)
        # energy shape ----> (batch, input_len)

        attention_weights = F.softmax(energy, dim=1)
        # attention_weights ----> (batch, input_len)

        # attention_weights.unsqueeze(1) ---> (batch, 1, input_len)
        # encoder_inputs                 ---> (batch, input_len, encoder_hidden)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs) # shape --> (batch, 1, encoder_hidden_dim)

        context_vector = context.squeeze(1)

        return context_vector, attention_weights