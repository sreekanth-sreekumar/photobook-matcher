import json
import pickle
import torch
import numpy as np

from torch.utils.data import Dataset
from collections import defaultdict

class SpeakerDataset(Dataset):
    def __init__(self, split): 
        self.split = split
        self.max_len = 0
        
        #Loading utterance chains
        with open('./data/' + split + '_text_chain.json', 'r') as f:
            self.chains = json.load(f)
        
        #Loading utterance file 
        with open('./data/' + split + '_id_utterance.pickle', 'rb') as f:
            self.utterances = pickle.load(f)

        #Loading original utterance file without unks
        with open('./data/' + split + '_text_utterance.pickle', 'rb') as f:
            self.text_refs = pickle.load(f)
        
        #Loading pre one-hot encoded image feature file
        with open('../../../dataset/v2/vectors.json', 'r') as f:
            self.image_features = json.load(f)

        self.img_dim = 2048
        self.img_count = 6

        self.data = dict()
        self.img2chain = defaultdict(dict)

        for chain in self.chains:
            self.img2chain[chain['target']][chain['game_id']] = chain['utterances']

        print("Processing ", split)

        for chain in self.chains[:len(self.chains)]:
            chain_utterances = chain['utterances']
            game_id = chain['game_id']

            for s in range(len(chain['utterances'])):
                
                utterance_id = tuple(chain_utterances[s])  # utterance_id = (game_id, round_nr, messsage_nr, img_id)

                #Get the previous utterance
                for pi in range(len(chain['utterances'])):
                    if chain["utterances"][pi] == list(utterance_id):
                        if pi == 0:
                            previous_utterance = []
                        else:
                            prev_id = chain["utterances"][pi - 1]
                            previous_utterance = self.utterances[tuple(prev_id)]['utterance']
                            break
                
                cur_utterance_obj = self.utterances[utterance_id]
                cur_utterance_text_ids = cur_utterance_obj['utterance']

                orig_target = self.text_refs[utterance_id]['utterance']
                orig_target = ' '.join(orig_target)

                length = cur_utterance_obj['length']

                if length > self.max_len:
                    self.max_len = length

                assert len(cur_utterance_text_ids) != 2
                # already had added sos eos into length and IDS version

                images = cur_utterance_obj['image_set']
                target = cur_utterance_obj['target']  # index of correct img

                target_image = images[target[0]]

                images = list(np.random.permutation(images))
                target = [images.index(target_image)]

                context_separate = torch.zeros(self.img_count, self.img_dim)
                im_counter = 0

                reference_chain = []

                for im in images:
                    context_separate[im_counter] = torch.tensor(self.image_features[im])
                    if im == images[target[0]]:
                        target_img_feats = context_separate[im_counter]
                        ref_chain = self.img2chain[im][game_id]

                        for rc in ref_chain:
                            rc_tuple = (rc[0], rc[1], rc[2], im)
                            reference_chain.append(' '.join(self.text_refs[rc_tuple]['utterance']))
                    im_counter += 1

                context_concat = context_separate.reshape(self.img_count * self.img_dim)

                self.data[len(self.data)] = {
                    'utterance': cur_utterance_text_ids,
                    'orig_utterance': orig_target,
                    'image_set': images,
                    'concat_context': context_concat,
                    'separate_images': context_separate,
                    'prev_utterance': previous_utterance,
                    'prev_length': len(previous_utterance),
                    'target': target,
                    'target_img_feats': target_img_feats,
                    'length': length,
                    'reference_chain': reference_chain
                }

    def __len__(self):
            return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def get_collate_fn(device, NOHS):
        
        def collate_fn(data):

            max_utt_length = max(d['length'] for d in data)
            max_prev_utt_length = max(d['prev_length'] for d in data)
            
            batch = defaultdict(list)

            for sample in data:
                for key in sample.keys():
                    if key == 'utterance':
                        padded = sample[key] + [0] * (max_utt_length - sample['length'])
                    elif key == 'prev_utterance':
                        if len(sample[key]) == 0:
                            # OTHERWISE pack_padded wouldn't work
                            padded = [NOHS] + [0] * (max_prev_utt_length - 1) # SPECIAL TOKEN FOR NO HIST

                        else:
                            padded = sample[key] + [0] * (max_prev_utt_length - len(sample[key]))
                    elif key == 'prev_length':
    
                        if sample[key] == 0:
                            # wouldn't work in pack_padded
                            padded = 1

                        else:
                            padded = sample[key]
                    elif key == 'image_set':
                        padded = [int(img) for img in sample['image_set']]
                    else:
                        padded = sample[key]
                    batch[key].append(padded)
            
            for key in batch.keys():
                # print(key)

                if key in ['separate_images', 'concat_context', 'target_img_feats']:
                    batch[key] = torch.stack(batch[key]).to(device)

                elif key in ['utterance', 'prev_utterance', 'target', 'length', 'prev_length']:
                    batch[key] = torch.Tensor(batch[key]).long().to(device)
            
            return batch
        
        return collate_fn