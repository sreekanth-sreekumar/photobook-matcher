import os
import json
import pickle
from collections import defaultdict

import torch
from transformers import BertTokenizer, BertModel


if not os.path.exists('data'):
    os.makedirs('data')

#Loading a pre trained tokeniser (Vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#Loading a pre trained model
model = BertModel.from_pretrained('bert-base-uncased')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.eval()
model.to(device)

def get_bert_outputs(text, model, tokenizer):
    # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    input_tensors = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])

    tokenised_text = tokenizer.tokenize('[CLS]' + text + '[SEP]')

    segments_ids = [0] * input_tensors.shape[1]
    segments_tensors = torch.tensor([segments_ids])

    input_tensors = input_tensors.to(device)
    segments_tensors = segments_tensors.to(device)

    with torch.no_grad():
        outputs = model(input_tensors, token_type_ids=segments_tensors)
        encoded_layers = outputs[0]
    assert tuple(encoded_layers.shape) == (1,  input_tensors.shape[1], model.config.hidden_size)
    assert len(tokenised_text) == input_tensors.shape[1]

    return encoded_layers, tokenised_text


def process4bert(data, split, model, tokenizer):
    chain_dataset = []
    chain_count = 0

    utterance_dataset = defaultdict()
    utterance_count = 0

    rep_dataset = defaultdict()

    chains_path = './data/' + split + '_chains.json'
    utterances_path = './data/' + split + '_utterances.pickle'
    reps_path = './data/' + split + '_rep.pickle'

    for img_file in sorted(data):
        img_id = str(int(img_file.split('/')[1].split('.')[0].split('_')[2]))
        chains4img = data[img_file]

        for game_id in sorted(chains4img):
            chain_data = chains4img[game_id]
            utt_lengths = []
            utt_ids = []
            for m in range(len(chain_data)):
                utterance_data = chain_data[m]
                message = utterance_data['Message_Text']
                message_nr = utterance_data['Message_Nr']
                round_nr = utterance_data['Round_Nr']

                encoded_utt, tokenized_message = get_bert_outputs(message, model, tokenizer)

                rep_dataset[(game_id, round_nr, message_nr)] = encoded_utt
                
                # including CLS and SEP and wordpiece count
                utt_length = len(tokenized_message)

                if utterance_data['Message_Speaker'] == 'A':
                    visual_context = utterance_data['Round_Images_A']
                else:
                    visual_context = utterance_data['Round_Images_B']

                visual_context_ids = []
                for visual in visual_context:
                    v_id = str(int(visual.split('/')[1].split('.')[0].split('_')[2]))
                    visual_context_ids.append(v_id)
                visual_context_ids = sorted(visual_context_ids)

                # utterance information
                utterance = {'utterance': tokenized_message, 'image_set': visual_context_ids,
                             'target': [visual_context_ids.index(img_id)], 'length': utt_length, 'game_id': game_id,
                             'round_nr': round_nr, 'message_nr': message_nr}

                utterance_dataset[(game_id, round_nr, message_nr, img_id)] = utterance # add to the full dataset
                utt_lengths.append(utterance['length'])
                utt_ids.append((game_id, round_nr, message_nr, img_id))
                utterance_count += 1

                if utterance_count % 500 == 0:
                    print(utterance_count)
            
            chain = {'game_id': game_id, 'chainid': chain_count, 'utterances': utt_ids, 'target': img_id,
            'lengths': utt_lengths}  # utterance lengths

            chain_dataset.append(chain)
            chain_count += 1
    
    with open(chains_path, 'w+') as f:
        json.dump(chain_dataset, f)

    # save the bert texts of words in utterances
    with open(utterances_path, 'wb+') as f:
        pickle.dump(utterance_dataset, f)

    # save the bert representations of words in utterances
    with open(reps_path, 'wb+') as f:
        pickle.dump(rep_dataset, f)


with open('../../dataset/v2/train.json', 'r') as f:
    train = json.load(f)

with open('../../dataset/v2/test.json', 'r') as f:
    test = json.load(f)

with open('../../dataset/v2/val.json', 'r') as f:
    val = json.load(f)

print('processing train...')
process4bert(train, 'train', model, tokenizer)

print('processing val...')
process4bert(val, 'val', model, tokenizer)

print('processing test...')
process4bert(test, 'test', model, tokenizer)