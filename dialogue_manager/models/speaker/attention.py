import torch.nn as nn

class Attn(nn.Module):
    def __init__(self, hidden_dim, attention_dim):
        super().__init__()
        self.lin2att_enc = nn.Linear(hidden_dim*2, attention_dim)
        self.lin2att_dec = nn.Linear(hidden_dim, attention_dim)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        self.attention = nn.Linear(attention_dim, 1)

    #Intialise weights for faster convergence
    def init_weights(self):

        for ll in [self.lin2att_enc, self.lin2att_dec, self.attention]:
            ll.bias.data.fill_(0)
            ll.weight.data.uniform_(-0.1, 0.1)

    def forward(self, enc_output, decoder_hidden, masks):
        #Encoder output to attention dim
        encoder_att = self.lin2att_enc(enc_output)
        #Decoder hidden state to attention dim
        decoder_att = self.lin2att_dec(decoder_hidden)

        attention_out = self.attention(self.tanh(encoder_att + decoder_att.unsqueeze(1)))
        attention_out = attention_out.masked_fill_(masks, float('-inf'))
        att_weights = self.softmax(attention_out)

        att_context_vector = (encoder_att * att_weights).sum(dim=1)

        return att_context_vector