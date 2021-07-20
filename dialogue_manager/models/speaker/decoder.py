import torch.nn as nn
import torch

from attention import Attn

class DecoderGru(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, attention_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.lin2voc = nn.Linear(attention_dim + hidden_dim, vocab_size)

        self.decoder_gru = nn.GRUCell(embedding_dim, hidden_dim, bias=True)
        self.attn = Attn(hidden_dim, attention_dim)

        self.init_weights()
        # word prediction scores

    def init_weights(self):
        #Intialise weights for faster convergence
        self.lin2voc.bias.data.fill_(0)
        self.lin2voc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, dec_input, dec_hidden, enc_output, masks):

            #Forward pass through decoder GRU
            dec_hidden_state = self.decoder_gru(dec_input, dec_hidden)

            #Finding attention weights
            att_context_vector = self.attn(enc_output, dec_hidden_state, masks)

            #Predictions are concatinations of context vector with dec_hidden_state
            word_pred = self.lin2voc(torch.cat((dec_hidden_state, att_context_vector), dim=1))

            return word_pred, dec_hidden_state

