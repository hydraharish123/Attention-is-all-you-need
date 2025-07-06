import torch 
import torch.nn as nn 
import torch.nn.functional as F

class LuongDotAttention(nn.Module):
    def __init__(self):
        super(LuongDotAttention, self).__init__()

    def forward(self, encoder_outputs, decoder_hidden):
        '''
            encoder_outputs ----> shape (batch, input_len, encoder_hidden_dim)
            decoder_hidden  ----> shape (batch, decoder_hidden_dim)
        '''
        decoder_hidden = decoder_hidden.unsqueeze(1) # shape ---> (batch, 1, decoder_hidden_dim)

        scores = torch.bmm(decoder_hidden, encoder_outputs.transpose(1, 2))
        # scores shape ----> (batch, 1, input_len)

        attention_weighs = F.softmax(scores, dim=-1)
        # scores shape ----> (batch, 1, input_len)

        context = torch.bmm(attention_weighs, encoder_outputs) # (batch, 1, input_len) * (batch, input_len, encoder_hidden_dim)
        # context shape ----> (batch, 1, encoder_hidden_dim)

        context_vector = context.squeeze(1) # (batch, encoder_hidden_dim)
        attention_weighs = attention_weighs.squeeze(1) # shape ---> (batch, input_len)

        return context_vector, attention_weighs
