import torch
import torch.nn.functional as F

import os
import json

from bert_score import score

#This method returns Official MSCOCO evaluator scores - bleu4, cider, rouge, meteor
def beam_search_eval(dataset, epoch, encoder, decoder, best_score, device, 
            beam_size, max_len, vocab, mask_attn, timestamp, isTest, linear_dec, embedding, print_please):

    if not os.path.exists('speaker_outputs'):
        os.makedirs('speaker_outputs')
    #True captions. For n images, references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...]
    references = []

    #Predictions. For n images, hypothesis = [hypo1, hypo2, ...]
    hypotheses = []
    empty_count = 0

    if isTest:
        split = 'test'
    else:
        split = 'val'

    file_name = split + '_' + str(epoch) + '_' + timestamp
    sos_token = torch.tensor(vocab['<sos>']).to(device)
    eos_token = torch.tensor(vocab['<eos>']).to(device)

    for i, data in enumerate(dataset):
        completed_sentences = []
        completed_scores = []
        beam_k = beam_size

        #original utternace without sos, eos, pad, nohs for calculation of metric scores
        original_reference = data['orig_utterance']
        #Complete reference of an image
        reference_chain = data['reference_chain'][0]

        prev_utterance = data['prev_utterance']
        prev_utt_lengths = data['prev_length']

        visual_context = data['concat_context']
        target_img_feats = data['target_img_feats']

        max_length_tensor = prev_utterance.shape[1]
        masks = mask_attn(prev_utt_lengths, max_length_tensor, device)

        enc_output, enc_hidden = encoder(prev_utterance, prev_utt_lengths, visual_context, target_img_feats)
        dec_hidden = linear_dec(torch.cat((enc_hidden[0], enc_hidden[1]), dim=1))
        dec_hidden = dec_hidden.expand(beam_k, -1)


        gen_len = 0
        dec_input = sos_token.expand(beam_k, 1)  # beam_k sos copies
        gen_sentences_k = dec_input  # all start off with sos now
        top_scores = torch.zeros(beam_k, 1).to(device)  # top-k generation scores

        while True:
            if gen_len > max_len:
                break

            dec_embed = embedding(dec_input).squeeze(1)
            word_pred, dec_hidden_state = decoder(dec_embed, dec_hidden, enc_output, masks)
            word_pred = F.log_softmax(word_pred, dim=1)
            word_pred = top_scores.expand_as(word_pred) + word_pred

            if gen_len == 0:
                top_scores, top_words = word_pred[0].topk(beam_k, 0, True, True)

            else:
                # unrolled
                top_scores, top_words = word_pred.view(-1).topk(beam_k, 0, True, True)

            # vocab - 1 to exclude <NOHS>
            sentence_index = top_words // (len(vocab)-1)  # which sentence it will be added to
            word_index = top_words % (len(vocab)-1)  # predicted word

            gen_len += 1

            # add the newly generated word to the sentences
            gen_sentences_k = torch.cat((gen_sentences_k[sentence_index], word_index.unsqueeze(1)), dim=1)

            
            #there could be incomplete sentences
            incomplete_sents_inds = [inc for inc in range(len(gen_sentences_k)) if
                                    eos_token not in gen_sentences_k[inc]]
                                
            complete_sents_inds = list(set(range(len(gen_sentences_k))) - set(incomplete_sents_inds))

            # save the completed sentences
            if len(complete_sents_inds) > 0:
                completed_sentences.extend(gen_sentences_k[complete_sents_inds].tolist())
                completed_scores.extend(top_scores[complete_sents_inds])

                beam_k -= len(complete_sents_inds)  # fewer, because we closed at least 1 beam

            if beam_k == 0:
                break

            # continue generation for the incomplete sentences
            gen_sentences_k = gen_sentences_k[incomplete_sents_inds]
            # use the ongoing hidden states of the incomplete sentences
            dec_hidden = dec_hidden_state[sentence_index[incomplete_sents_inds]]
            top_scores = top_scores[incomplete_sents_inds].unsqueeze(1)
            dec_input = word_index[incomplete_sents_inds]

        if len(completed_scores) == 0:

            empty_count += 1
            # all incomplete here
            completed_sentences.extend((gen_sentences_k[incomplete_sents_inds].tolist()))
            completed_scores.extend(top_scores[incomplete_sents_inds])

        sorted_scores, sorted_indices = torch.sort(torch.tensor(completed_scores), descending=True)

        best_seq = completed_sentences[sorted_indices[0]]

        hypothesis = [vocab.index2word[w] for w in best_seq if w not in
                    [vocab.word2index['<sos>'], vocab.word2index['<eos>'], vocab.word2index['<pad>']]]
        # remove sos and pads # I want to check eos
        hypothesis_string = ' '.join(hypothesis)
        hypotheses.append(hypothesis_string)

        if not os.path.isfile('./speaker_outputs/refs_' + file_name + '.json'):
            # Reference
            references.append(reference_chain)

        if print_please:
        # Reference
            print('REF:', original_reference) # single one
            print('HYP:', hypothesis_string)
            
    if os.path.isfile('./speaker_outputs/refs_' + file_name + '.json'):
        with open('./speaker_outputs/refs_' + file_name + '.json', 'r') as f:
            references = json.load(f)
    else:
        with open('./speaker_outputs/refs_' + file_name + '.json', 'w+') as f:
            json.dump(references, f)
        
    (P, R, Fs), hashname = score(hypotheses, references, lang='en', return_hash=True, model_type="bert-base-uncased")
    print(f'{hashname}: P={P.mean().item():.6f} R={R.mean().item():.6f} F={Fs.mean().item():.6f}')

    selected_metric_score = Fs.mean().item()
    print("Bert_Metric_Score: ", round(selected_metric_score, 5))

    if isTest:
        with open('./speaker_outputs/hyps_' + file_name + '_test.json', 'w+') as f:
            json.dump(hypotheses, f)
    else:
        has_best_score = False

        if selected_metric_score > best_score:
            best_score = selected_metric_score
            has_best_score = True

            with open('./speaker_outputs/hyps_' + file_name + '_val.json', 'w+') as f:
                json.dump(hypotheses, f)

        return best_score, selected_metric_score, has_best_score



