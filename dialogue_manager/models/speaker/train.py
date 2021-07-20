import numpy as np
import torch
import datetime
import os
import argparse

from torch import optim
import torch.nn as nn

from utils.Vocab import Vocab
from utils.SpeakerDataset import SpeakerDataset
from evaluate import beam_search_eval

from encoder import EncoderGru
from decoder import DecoderGru

def save_model(encoder, decoder, epoch, score, encoder_optimiser, decoder_optimiser, t, vocab):
    file_name = './saved_models/speaker_' + str(seed) + '_' + str(epoch) + '.pkl'
    duration = datetime.datetime.now() - t
    print("Model running for: ", duration)
    torch.save({
        'score': score,
        'epoch': epoch,
        'encoder': encoder.state_dict(),
        'encoder_optimiser': encoder_optimiser.state_dict(),
        'decoder': decoder.state_dict(),
        'decoder_optimiser': decoder_optimiser.state_dict(),
    }, file_name)
    torch.save(vocab, './saved_models/vocab_obj.pth')



def print_predictions(predicted, expected, vocab):
    selected_tokens = torch.argmax(predicted, dim=2)
    for b in range(selected_tokens.shape[0]):
        # reference
        reference = expected[b].data
        reference_string = ''
        for r in range(len(reference)):
            reference_string += vocab.index2word[reference[r].item()]
            if r < len(reference) - 1:
                reference_string += ' '

        print('***REF***: ', reference_string)
        generation = selected_tokens[b].data
        generation_string = ''
        for g in range(len(generation)):
            generation_string += vocab.index2word[generation[g].item()]
            if g < len(generation) - 1:
                generation_string += ' '

        print('***GEN***: ', generation_string)

def mask_attn(actual_num_tokens, max_num_tokens, device):
    
    masks = []
    for n in range(len(actual_num_tokens)):
        # items to be masked are TRUE
        mask = [False] * actual_num_tokens[n] + [True] * (max_num_tokens - actual_num_tokens[n])
        masks.append(mask)
    masks = torch.tensor(masks).unsqueeze(2).to(device)
    return masks

if not os.path.isdir('saved_models'):
    os.mkdir('saved_models')

parser = argparse.ArgumentParser()
parser.add_argument("-load_file", action='store_true')
parser.add_argument("-print_please", action="store_true")

t = datetime.datetime.now()
timestamp = str(t.date()) + ' ' + str(t.hour) + ' hours ' + str(t.minute) + ' minutes ' + str(t.second) + ' seconds'

print('code starts at: ', timestamp)

args = parser.parse_args()


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print('Loading the vocab ...')
vocab = Vocab('./data/vocab.csv')
vocab.index2word[len(vocab)] = '<nohs>'  # special token placeholder for no prev utt
vocab.word2index['<nohs>'] = len(vocab) 

trainset = SpeakerDataset("train")
valset = SpeakerDataset("val")

print('vocab len', len(vocab))
print('train len', len(trainset), 'longest sentence', trainset.max_len)
print('val len', len(valset), 'longest sentence', valset.max_len)


max_len = 30    #For beam search
img_dim = 2048
embedding_dim = 1024
attention_dim = 512
hidden_dim = 512
batch_size = 16
beam_size = 3
dropout_prob = 0.3
learning_rate = 0.0001
decoder_learning_ratio = 5.0
gradient_clip = 5.0

#epoch = 5      #Set this to the last epoch at which a saved model file was created

loss_criterion = nn.CrossEntropyLoss(reduction="sum", ignore_index=0)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

encoder = EncoderGru(len(vocab), embedding_dim, hidden_dim, img_dim, dropout_prob).to(device)
encoder_optimiser = optim.Adam(encoder.parameters(), lr=learning_rate)

#len(vocab) - 1 to not include nohs
decoder = DecoderGru(len(vocab)-1, embedding_dim, hidden_dim, attention_dim).to(device)
decoder_optimiser = optim.Adam(decoder.parameters(), lr=learning_rate*decoder_learning_ratio)


load_params = {
    'batch_size': batch_size,
    'shuffle': True,
    'collate_fn': SpeakerDataset.get_collate_fn(device, vocab['<nohs>'])
}

load_params_val = {
    'batch_size': 1,
    'shuffle': False,
    'collate_fn': SpeakerDataset.get_collate_fn(device, vocab['<nohs>'])
}

training_loader = torch.utils.data.DataLoader(trainset, **load_params)
val_loader = torch.utils.data.DataLoader(valset, **load_params_val)

epochs = 100
patience = 50 # when to stop if there is no improvement
patience_counter = 0
epoch_sd = 0

best_score = -1
prev_score = -1
best_epoch = -1

linear_dec = nn.Linear(hidden_dim*2, hidden_dim)
embedding = nn.Embedding(len(vocab)-1, embedding_dim, padding_idx=0, scale_grad_by_freq=True)

#Easier convergence
embedding.weight.data.uniform_(-0.1, 0.1)
linear_dec.bias.data.fill_(0)
linear_dec.weight.data.uniform_(-0.1, 0.1)

if (args.load_file):
    file_name = './saved_models/speaker_' + str(seed) + '_' + '15.pkl'     #Change epoch number accordingly
    checkpoint = torch.load(file_name, map_location=device)
    best_score = prev_score = checkpoint['score']
    epoch_sd = checkpoint['epoch']
    encoder_sd = checkpoint['encoder']
    decoder_sd = checkpoint['decoder']
    encoder_optimiser_sd = checkpoint['encoder_optimiser']
    decoder_optimiser_sd = checkpoint['decoder_optimiser']
    encoder.load_state_dict(encoder_sd)
    encoder_optimiser.load_state_dict(encoder_optimiser_sd)
    decoder.load_state_dict(decoder_sd)
    decoder_optimiser.load_state_dict(decoder_optimiser_sd)

t = datetime.datetime.now()
timestamp_tr = str(t.date()) + ' ' + str(t.hour) + ' hours ' + str(t.minute) + ' minutes ' + str(t.second) + ' seconds'

print("Training starts: ", timestamp_tr)

for epoch in range(epoch_sd, epochs):
    print('Epoch: ', epoch)
    losses = []

    encoder.train()
    decoder.train()
    torch.enable_grad()

    for i, data in enumerate(training_loader):
        encoder_optimiser.zero_grad()
        decoder_optimiser.zero_grad()

        utterances_text_ids = data['utterance']
        original_reference = data['orig_utterance']
        prev_utterance_ids = data['prev_utterance']
        prev_lengths = data['prev_length']

        context_concat = data['concat_context']
        target_img_feats = data['target_img_feats']

        lengths = data['length']
        targets = data['target']  # image target

        max_length_tensor = prev_utterance_ids.shape[1]
        masks = mask_attn(prev_lengths, max_length_tensor, device)

        enc_output, enc_hidden = encoder(prev_utterance_ids, prev_lengths, context_concat, target_img_feats)

        #Teacher forcing, so giving current utterance as decoder input
        batch_size = utterances_text_ids.shape[0]
        decode_length = utterances_text_ids.shape[1] - 1    #Except EOS

        # word prediction scores
        predictions = torch.zeros(batch_size, decode_length, len(vocab)-1).to(device)
        dec_hidden = linear_dec(torch.cat((enc_hidden[0], enc_hidden[1]), dim=1))
        dec_input = embedding(utterances_text_ids)

        #Teacher forcing. Feeding utterance words as decoder input words
        for l in range(decode_length):
            word_pred, _ = decoder(dec_input[:,l], dec_hidden, enc_output, masks)
            predictions[:, l] = word_pred

        if args.print_please:
            print_predictions(predictions, utterances_text_ids, vocab)
        
        # predicitons is [batch_size, seq_length, number of classes]
        predictions = predictions.permute(0,2,1)
        # predictions is now [batch_size, number_of_classes, seq_length]

        target_utterance_ids = utterances_text_ids[:,1:] #utternace without SOS
        loss = loss_criterion(predictions, target_utterance_ids)

        losses.append(loss.item())
        loss.backward()

        _ = nn.utils.clip_grad_norm_(encoder.parameters(), gradient_clip)
        _ = nn.utils.clip_grad_norm_(decoder.parameters(), gradient_clip)

        encoder_optimiser.step()
        decoder_optimiser.step()
    
    print('Train loss: ', round(np.sum(losses), 3))

    isTest = False

    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        best_score, score, has_best_score = \
            beam_search_eval(val_loader, epoch, encoder, decoder, best_score, device, 
            beam_size, max_len, vocab, mask_attn, timestamp, isTest, linear_dec, embedding, args.print_please)

        if has_best_score:
            best_epoch = epoch
            patience_counter = 0

            save_model(encoder, decoder, epoch, score, encoder_optimiser, decoder_optimiser, t, vocab)

        else:
            patience_counter += 1
            if patience_counter == patience:
                duration = datetime.datetime.now() - t
                print('model ending duration', duration)
                break
        
        prev_score = score # not using, stopping based on best score

        print('\nBest score: ', round(best_score,5), ', epoch: ', best_epoch)  # , best_loss)  #validset
        print()
    


       


