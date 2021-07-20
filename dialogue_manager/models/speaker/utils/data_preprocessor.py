import json
import pickle
import csv
import os

from collections import defaultdict, Counter
from nltk import TweetTokenizer

from Vocab import Vocab

if not os.path.exists('data'):
    os.makedirs('data')

min_freq = 2
tweet_tokenizer = TweetTokenizer(preserve_case=False)

def process_data(data, split, min_freq=2):
    if split == "train":
        train_full_vocab = []
        vocab_csv_path = './data/vocab.csv'
    chain_dataset = []
    utterance_dataset = defaultdict(int)

    chain_count = 0
    utterance_count = 0

    chain_path = './data/' + split + '_text_chain.json'
    utterance_path = './data/' + split + '_text_utterance.pickle'

    for img in sorted(data):
        img_id = str(int(img.split('/')[1].split('.')[0].split('_')[2]))
        chain4data = data[img]

        for game_id in chain4data:
            chain_data = chain4data[game_id]
            utt_lengths = []
            utt_ids = []

            for m in range(len(chain_data)):
                utterance_data = chain_data[m]
                message = utterance_data['Message_Text']
                message_nr = utterance_data['Message_Nr']
                round_nr = utterance_data['Round_Nr']
                tokenized_message = tweet_tokenizer.tokenize(message)
                if (split == "train"):
                    train_full_vocab.extend(tokenized_message)
                if utterance_data['Message_Speaker'] == 'A':
                    visual_context = utterance_data['Round_Images_A']
                else:
                    visual_context = utterance_data['Round_Images_B']

                visual_context_ids = []
                for visual in visual_context:
                    v_id = str(int(visual.split('/')[1].split('.')[0].split('_')[2]))
                    visual_context_ids.append(v_id)
                visual_context_ids = sorted(visual_context_ids)
                utt_length = len(tokenized_message) + 2  # SOS and EOS tokens added
                utterance = {'utterance': tokenized_message, 'image_set': visual_context_ids,
                             'target': [visual_context_ids.index(img_id)], 'length': utt_length, 'game_id': game_id,
                             'round_nr': round_nr, 'message_nr': message_nr}

                utterance_dataset[(game_id, round_nr, message_nr, img_id)] = utterance
                utt_lengths.append(utt_length)
                utt_ids.append((game_id, round_nr, message_nr, img_id))

                utterance_count += 1
                if utterance_count % 500 == 0:
                    print("Total number of utterances parsed: ", utterance_count)
            
            chain = {'game_id': game_id, 'chain_id': chain_count, 'utterances': utt_ids, 'target': img_id,
                     'lengths': utt_lengths}
            chain_dataset.append(chain)
            chain_count += 1

    with open(chain_path, "w+") as f:
        json.dump(chain_dataset, f)
    
    with open(utterance_path, "wb+") as f:
        pickle.dump(utterance_dataset, f)

    if split == "train":
        vocab_ordered = Counter(train_full_vocab).most_common()
        truncated_word_list = []

        for word, freq in vocab_ordered:
            if freq > min_freq:
                truncated_word_list.append(( word,freq ))
        
        with open(vocab_csv_path, "w+") as f:
            writer = csv.writer(f, delimiter=',', quotechar='|')
            writer.writerows(truncated_word_list)


with open('../../../dataset/v2/train.json', 'r') as f:
    train = json.load(f)

with open('../../../dataset/v2/test.json', 'r') as f:
    test = json.load(f)

with open('../../../dataset/v2/val.json', 'r') as f:
    val = json.load(f)

print('processing train...')
process_data(train, 'train', min_freq)

print('processing val...')
process_data(val, 'val')

print('processing test...')
process_data(test, 'test')


vocab = Vocab('data/vocab.csv')

# convert words to indexes

def convert2indices(dataset, vocab, split):
    for tup in dataset:
        utt = dataset[tup]
        text = utt['utterance']
        ids = [vocab['<sos>']] + [vocab[t] for t in text] + [vocab['<eos>']]
        utt['utterance'] = ids
    new_file_name = 'data/' + split + '_id_utterance.pickle'
    with open(new_file_name, 'wb') as f:
        pickle.dump(dataset, f)

with open('./data/train_text_utterance.pickle', 'rb') as f:
    train_utterances = pickle.load(f)

with open('./data/val_text_utterance.pickle', 'rb') as f:
    val_utterances = pickle.load(f)

with open('./data/test_text_utterance.pickle', 'rb') as f:
    test_utterances = pickle.load(f)

print('Converting datasets into indices')
convert2indices(train_utterances, vocab, 'train')
convert2indices(val_utterances, vocab, 'val')
convert2indices(test_utterances, vocab, 'test')
