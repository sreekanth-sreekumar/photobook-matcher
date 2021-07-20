import os
import numpy as np
import datetime
import argparse

import torch
import torch.nn as nn
from utils.ListenerDataset import ListenerDataset
from evaluate import evaluate

from model import ListenerModel
from torch import optim

def mask_attn(actual_num_tokens, max_length_tokens, device):
    masks = []
    for m in range(len(actual_num_tokens)):
        mask = [False] * actual_num_tokens[m] + [True] * (max_length_tokens - actual_num_tokens[m])
        masks.append(mask)
    masks = torch.tensor(masks).unsqueeze(2).to(device)
    return masks

def save_model(epoch, best_acc, model, model_optim, seed):
    file_name = './saved_models/listener_' + str(seed) + '_' + str(epoch) + '.pkl'
    duration = datetime.datetime.now() - t
    print("Model running for: ", duration)
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'model_optimiser': model_optim.state_dict(),
        'best_accuracy': best_acc
    }, file_name)

if not os.path.isdir('./saved_models'):
    os.mkdir('./saved_models')

parser = argparse.ArgumentParser()
parser.add_argument("-load_file", action='store_true')
args = parser.parse_args()

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
learning_rate = 0.0001
gradient_clip = 5.0

utterance_file = '_utterances.pickle'
rep_file = '_rep.pickle'
chain_file = '_chains.json'
embedding_dim = 768

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_set = ListenerDataset('train', utterance_file, rep_file, chain_file)
val_set = ListenerDataset('val', utterance_file, rep_file, chain_file)

load_params = {
    'batch_size': 1, 'shuffle': True,
    'collate_fn': ListenerDataset.get_collate_fn(device)
}

load_params_val = {
    'batch_size': batch_size, 'shuffle': False,
    'collate_fn': ListenerDataset.get_collate_fn(device)
}

epochs = 100
epoch_sd = 0
patience = 50
patience_counter = 0

best_epoch = -1
best_accuracy = -1

model = ListenerModel(embedding_dim, hidden_dim, img_dim, attention_dim, dropout_prob)
model_optim = optim.Adam(model.parameters(), lr=learning_rate)
loss_criterion = nn.CrossEntropyLoss(reduction="sum")

if args.load_file:
    file_name = './saved_models/listener_' + str(seed) + '_' + '15.pkl' #Replace with appropriate epoch number later
    checkpoint = torch.load(file_name, map_location=device)
    best_accuracy = checkpoint['best_accuracy']
    model_sd = checkpoint['model']
    model_optim_sd = checkpoint['model_optimiser']
    epoch_sd = best_epoch = checkpoint['epoch']
    model.load_state_dict(model_sd)
    model_optim.load_state_dict(model_optim_sd)

t = datetime.datetime.now()
timestamp = str(t.date()) + ' ' + str(t.hour) + ' hours ' + str(t.minute) + ' minutes ' + str(t.second) + ' seconds'
print('Training starts: ', timestamp)

for epoch in range(epoch_sd, epochs):
    print('Epoch: ', epoch)
    training_loader = torch.utils.data.DataLoader(train_set, **load_params)
    val_loader = torch.utils.data.DataLoader(val_set, **load_params_val)

    losses = []
    model.train()
    torch.enable_grad()
    
    for i,data in enumerate(training_loader):
        model_optim.zero_grad()

        utterances_BERT = data['representations']
        context_separate = data['separate_images']
        context_concat = data['concat_context']
        lengths = data['length']
        targets = data['target']
        prev_hist = data['prev_histories']

        max_length_token = utterances_BERT.shape[1]
        masks = mask_attn(lengths, max_length_token, device)
        out = model(utterances_BERT, context_separate, context_concat, masks, prev_hist, device)
        loss = loss_criterion(out, targets)
        losses.append(loss.item())
        loss.backward()

        _ = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        model_optim.step()
    print("Total loss for epoch ", epoch, " is ", round(np.sum(losses), 5))

    with torch.no_grad():
        model.eval()
        current_accuracy = evaluate(val_loader, model, True, device, mask_attn, len(val_set))
        
        if best_accuracy >= current_accuracy:
            patience_counter +=1
            if (patience == patience_counter):
                duration = datetime.datetime.now() - t
                print("Model has been training for ", duration)
                break
        else:
            patience_counter = 0
            best_accuracy = current_accuracy
            best_epoch = epoch
            save_model(epoch, best_accuracy, model, model_optim, seed)
        
        print('Patience: ', patience_counter, '\n')
        print('\nBest epoch: ', best_epoch, "  Best Accuracy: ", round(best_accuracy, 5))  # validset
        print()





