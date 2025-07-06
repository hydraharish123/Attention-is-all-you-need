import torch 
import torch.nn as nn 
import torch.nn.functional as F

class LuongGeneralAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(LuongGeneralAttention, self).__init__()
        self.Wa = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, encoder_outputs, decoder_hidden):
        '''
            encoder_outputs ----> shape (batch, input_len, hidden_dim)
            decoder_hidden  ----> shape (batch, hidden_dim)
        '''

        encoder_outputs = self.Wa(encoder_outputs)
        # shape --> (batch, input_len, hidden_dim)

        decoder_hidden = decoder_hidden.unsqueeze(1) # shape --> (batch, 1, hidden_dim)

        scores = torch.bmm(decoder_hidden, encoder_outputs.transpose(1,2))
        # shape --> (batch, 1, input_len)

        attention_weights = F.softmax(scores.squeeze(1), dim=1)
        # shape --> (batch, input_len)

        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        # shape --> (batch, 1, hidden_dim)
        
        return context.squeeze(1), attention_weights