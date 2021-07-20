import torch
import torch.nn as nn
import numpy as np
import datetime
from evaluate import evaluate

from utils.ListenerDataset import ListenerDataset
from model import ListenerModel

def mask_attn(actual_num_tokens, max_length_tokens, device):
    masks = []
    for m in range(len(actual_num_tokens)):
        mask = [False] * actual_num_tokens[m] + [True] * (max_length_tokens - actual_num_tokens[m])
        masks.append(mask)
    masks = torch.tensor(masks).unsqueeze(2).to(device)
    return masks

seed = 3
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

img_dim = 2048
attention_dim = 512
hidden_dim = 512
batch_size = 32
dropout_prob = 0.5

utterance_file = '_utterances.pickle'
rep_file = '_rep.pickle'
chain_file = '_chains.json'
embedding_dim = 768

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

test_set = ListenerDataset('test', utterance_file, rep_file, chain_file)

load_params_test = {
    'batch_size': batch_size, 'shuffle': False,
    'collate_fn': ListenerDataset.get_collate_fn(device)
}

test_loader = nn.utils.DataLoader(test_set, **load_params_test)

model = ListenerModel(embedding_dim, hidden_dim, img_dim, attention_dim, dropout_prob)

load_file_name = './saved_models/listener_3_6.pkl'    #Replace with suitable epoch number
checkpoint = torch.load(load_file_name, map_location=device)
model.load_state_dict(checkpoint['model']) 

with torch.no_grad():
    model.eval()
    evaluate(test_loader, model, False, device, mask_attn, len(test_set))