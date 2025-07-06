import torch 
import torch.nn as nn 
import torch.nn.functional as F

class LuongConcatAttention(nn.Module):
    def __init__(self, hidden_dim, attention_dim):
        super(LuongConcatAttention, self).__init__()
        self.Wa = nn.Linear(hidden_dim*2, attention_dim) ## for concatentation --> [hi ; st]
        self.Va = nn.Linear(attention_dim,1)

    def forward(self, encoder_outputs, decoder_hidden):
        '''
            encoder_outputs ----> shape (batch, input_len, hidden_dim)
            decoder_hidden  ----> shape (batch, hidden_dim)
        '''

        _, input_len, _ = encoder_outputs.size()

        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, input_len, 1)
        # shape ---> (batch, hidden_dim) to (batch, input_len, hidden_dim)

        concatenate = torch.cat((encoder_outputs, decoder_hidden), dim=2)
        # shape --> (batch, input_len, hidden_dim * 2)

        energy = F.tanh(self.Wa(concatenate)) # shape --> (batch, input_len, attention_dim)

        scores = self.Va(energy).squeeze(2) # shape --> (batch, input_len)

        attention_weights = F.softmax(scores, dim=1) 
        # shape --> (batch, input_len)

        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        # shape --> (batch, 1, hidden_dim)

        context_vector = context.squeeze(1) # shape --> (batch, hidden_dim)
        return context_vector, attention_weights 