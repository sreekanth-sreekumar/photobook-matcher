import torch.nn as nn
import torch

class EncoderGru(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, img_dim, dropout_prob):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0, scale_grad_by_freq=True)
        self.encoder_gru = nn.GRU(embedding_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

        self.lin_visual_context = nn.Linear(img_dim*6, hidden_dim)
        self.lin_separate_img = nn.Linear(img_dim, hidden_dim)
        self.lin_hidden = nn.Linear(hidden_dim*2, hidden_dim)

        self.init_weights()

    def init_weights(self):
        #Intialise weights for faster convergence
        self.embedding.weight.data.uniform_(-0.1, 0.1)

        for ll in [self.lin_hidden, self.lin_separate_img, self.lin_visual_context]:
            ll.bias.data.fill_(0)
            ll.weight.data.uniform_(-0.1, 0.1)

    def forward(self, prev_utterance, prev_utt_lengths, visual_context, target_image_feat):
        visual_context_hid = self.relu(self.lin_visual_context(self.dropout(visual_context)))
        target_img_hid = self.relu(self.lin_separate_img(self.dropout(target_image_feat)))
        visual_context_concat = self.relu(self.lin_hidden(torch.cat((visual_context_hid, target_img_hid), dim=1)))

        #Hout = 512
        prev_utt_embedding = self.dropout(self.embedding(prev_utterance))
        packed_input = nn.utils.rnn.pack_padded_sequence(prev_utt_embedding, prev_utt_lengths, batch_first=True, enforce_sorted=False)

        #Intialise encoder hidden state
        visual_context_concat = torch.stack((visual_context_concat,visual_context_concat), dim=0)
        
        packed_output, hidden_state = self.encoder_gru(packed_input, visual_context_concat)

        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        return output, hidden_state
