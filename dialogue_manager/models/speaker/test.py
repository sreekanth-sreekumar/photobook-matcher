import torch
import numpy as np
import datetime
import torch.nn as nn
import argparse

from encoder import EncoderGru
from decoder import DecoderGru
from evaluate import beam_search_eval
from utils.SpeakerDataset import SpeakerDataset

from utils.Vocab import Vocab

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
speaker_file = 'saved_models/speaker_42_6.pkl'  #Substitute with appropriate epoch number

seed = 42

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

checkpoint = torch.load(speaker_file, map_location=device)

print('Loading the vocab')
vocab = Vocab('./data/vocab.csv')
vocab.index2word[len(vocab)] = '<nohs>'  # special token placeholder for no prev utt
vocab.word2index['<nohs>'] = len(vocab)

testset = SpeakerDataset("test")
print('vocab len', len(vocab))
print('test len', len(testset), 'longest sentence', testset.max_len)

max_len = 30    #For beam search
img_dim = 2048
embedding_dim = 1024
attention_dim = 512
hidden_dim = 512
beam_size = 3
dropout_prob = 0.3

encoder = EncoderGru(len(vocab), embedding_dim, hidden_dim, img_dim, dropout_prob).to(device)
#len(vocab) - 1 to not include nohs
decoder = DecoderGru(len(vocab)-1, embedding_dim, hidden_dim, attention_dim).to(device)

linear_dec = nn.Linear(hidden_dim*2, hidden_dim)
embedding = nn.Embedding(len(vocab)-1, embedding_dim, padding_idx=0, scale_grad_by_freq=True)

#Easier convergence
embedding.weight.data.uniform_(-0.1, 0.1)
linear_dec.bias.data.fill_(0)
linear_dec.weight.data.uniform_(-0.1, 0.1)

load_params_test = {
    'batch_size': 1,
    'shuffle': False,
    'collate_fn': SpeakerDataset.get_collate_fn(device, vocab['<nohs>'])
}

test_loader = torch.utils.data.DataLoader(testset, **load_params_test)

encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])

parser = argparse.ArgumentParser()
parser.add_argument("-print_please", action="store_true")
args = parser.parse_args()

def mask_attn(actual_num_tokens, max_num_tokens, device):
    masks = []

    for m in range(len(actual_num_tokens)):
        mask = [False] * actual_num_tokens[m] + [True] * (max_num_tokens - actual_num_tokens[m])
        masks.append(mask)
    masks = torch.tensor(masks).unsqueeze(2).to(device)
    return masks

with torch.no_grad():
    encoder.eval()
    decoder.eval()
    isTest = True

    t = datetime.datetime.now()
    timestamp = str(t.date()) + ' ' + str(t.hour) + ' hours ' + str(t.minute) + ' minutes ' + str(t.second) + ' seconds'

    beam_search_eval(test_loader, 1, encoder, decoder, 0, device, 
            beam_size, max_len, vocab, mask_attn, timestamp, isTest, linear_dec, embedding, args.print_please)
