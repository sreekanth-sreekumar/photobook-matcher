import torch
import pickle
import json
import numpy as np

from collections import defaultdict
from torch.utils.data import Dataset

class ListenerDataset(Dataset):
    def __init__(self, split, utterance_file, rep_file, chain_file):
        
        with open('./data/' + split + utterance_file, 'rb') as f:
            self.utterances = pickle.load(f)

        with open('./data/' + split + chain_file, 'r') as f:
            self.chains = json.load(f)
        
        with open('./data/' + split + rep_file, 'rb') as f:
            self.reps = pickle.load(f)
        
        with open('../../../dataset/v2/vectors.json', 'r') as f:
            self.image_features = json.load(f)
        
        self.img_dim = 2048
        self.img_count = 6

        self.data = dict()
        self.img2chain = defaultdict(dict)

        for chain in self.chains:
            self.img2chain[chain['target']][chain['game_id']] = chain['utterances']
        
        print('Processing ', split)
        for chain in self.chains:
            chain_utterances = chain['utterances']
            game_id = chain['game_id']

            for s in range(len(chain_utterances)):
                prev_chains = defaultdict(list)
                prev_lengths = defaultdict(int)

                utterance_id = tuple(chain_utterances[s])
                round_nr = utterance_id[1]
                message_nr = utterance_id[2]

                cur_utterance_obj = self.utterances[utterance_id]
                cur_utterance_text = cur_utterance_obj['utterance']
                cur_utterance_rep = self.reps[(game_id, round_nr, message_nr)].squeeze(dim=0)
                length = cur_utterance_obj['length']

                assert len(cur_utterance_text) != 2 #CLS and SEP tokens were added

                images = cur_utterance_obj['image_set']
                target = cur_utterance_obj['target']
                target_image = images[target[0]]
                images = list(np.random.permutation(images))
                target = [images.index(target_image)]
                
                context_separate = torch.zeros(self.img_count, self.img_dim)
                im_counter = 0

                for im in images:
                    context_separate[im_counter] = torch.tensor(self.image_features[im])
                    im_counter += 1
                    
                    if game_id in self.img2chain[im]:

                        temp_chain = self.img2chain[im][game_id] #temp_chain indexes are (game_id, round_nr, messsage_nr, img_id)

                        hist_utt = []

                        for t in range(len(temp_chain)):
                            _, t_round_nr, t_message_nr, _ = temp_chain[t]
                            if t_round_nr < round_nr:
                                hist_utt.append((game_id, t_round_nr, t_message_nr))
                            elif t_round_nr == round_nr:
                                if t_message_nr < message_nr:
                                    hist_utt.append((game_id, t_round_nr, t_message_nr))

                        if len(hist_utt) > 0:
                            for hu in [hist_utt[-1]]: #Only getting the most recent history
                                prev_chains[im].extend(self.reps[hu].squeeze(dim=0))
                        else:
                            prev_chains[im] = []
                    
                    else:
                        prev_chains[im] = []
                    prev_lengths[im] = len(prev_chains[im])

                context_concat = context_separate.reshape(self.img_count * self.img_dim)

                self.data[len(self.data)] = {
                    'utterance': cur_utterance_text,
                    'representations': cur_utterance_rep,
                    'image_set': images,
                    'concat_context': context_concat,
                    'separate_images': context_separate,
                    'target': target,
                    'length': length,
                    'prev_histories': prev_chains,
                    'prev_history_lengths': prev_lengths
                }

    def __len__(self):
            return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def get_collate_fn(device):

        def collate_fn(data):
            max_source_len = max(d['length'] for d in data)
            batch = defaultdict(list)

            for sample in data:
                for key in data[0].keys():
                    if key == 'representations':
                        pad_rep = torch.zeros(max_source_len - sample['length'], sample[key].shape[1])
                        padded = torch.cat((sample[key], pad_rep), dim=0)

                    elif key == 'image_set':
                        padded = [int(img) for img in sample[key]]

                    elif key == 'prev_histories':
                        histories_per_img = []
                        for k in range(len(sample['image_set'])):
                            #keep the order of imgs
                            img_id = sample['image_set'][k]
                            history = sample[key][img_id]
                            histories_per_img.append(history)
                        padded = histories_per_img

                    elif key == 'prev_history_lengths':
                        histlens_per_img = []
                        for k in range(len(sample['image_set'])):
                            #keep the order of imgs
                            img_id = sample['image_set'][k]
                            history_length = sample[key][img_id]
                            histlens_per_img.append(history_length)
                        padded = histlens_per_img

                    else:
                        padded = sample[key]
                    
                    batch[key].append(padded)
            
            for key in batch.keys():
                    if key not in ['prev_histories', 'utterance', 'representations',
                            'separate_images', 'concat_context']:
                        batch[key] = torch.Tensor(batch[key]).long().to(device)

                    # for instance targets can be long and sent to device immediately
                    elif key in ['representations', 'separate_images', 'concat_context']:
                        batch[key] = torch.stack(batch[key]).to(device)  # float
            return batch

        return collate_fn




