import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import EncoderGru
from .decoder import DecoderGru

from .utils.Vocab import Vocab
from nltk import TweetTokenizer

tweet_tokenizer = TweetTokenizer(preserve_case=False)

def getPreviousUttEmbedding(message, vocab):
    tokenized_message = tweet_tokenizer.tokenize(message)
    utt_ids = [vocab['<sos>']] + [vocab[t] for t in tokenized_message] + [vocab['<eos>']]
    utt_length = len(tokenized_message) + 2
    return utt_ids, utt_length


def getMaskAttention(tokens_length, device):
    masks = [[False] * tokens_length]
    masks = torch.tensor(masks).unsqueeze(2).to(device)
    return masks

def evaluate_speaker_game(visual_context, target_img_feats, message, beam_size):

    speaker_model_file = './saved_models/speaker_42_6.pkl' #Substitute with appropriate epoch number
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    checkpoint = torch.load(speaker_model_file, map_location=device)

    max_len = 30    #For beam search
    img_dim = 2048
    embedding_dim = 1024
    attention_dim = 512
    hidden_dim = 512
    batch_size = 16
    beam_size = 3
    gradient_clip = 5.0
    dropout_prob = 0.3

    # vocab = Vocab('./models/speaker/data/vocab.csv')
    # vocab.index2word[len(vocab)] = '<nohs>'  # special token placeholder for no prev utt
    # vocab.word2index['<nohs>'] = len(vocab)  # len(vocab) updated (depends on w2i)

    vocab = torch.load('./saved_models/vocab_obj.pth')

    prev_utterance, prev_utt_length = getPreviousUttEmbedding(message, vocab)
    prev_utterance = torch.tensor(prev_utterance).long().to(device).unsqueeze(0)
    prev_utt_length = torch.tensor(prev_utt_length).long().to(device).unsqueeze(0)

    encoder = EncoderGru(len(vocab), embedding_dim, hidden_dim, img_dim, dropout_prob).to(device)
    #len(vocab) - 1 to not include nohs
    decoder = DecoderGru(len(vocab)-1, embedding_dim, hidden_dim, attention_dim).to(device)

    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

    linear_dec = nn.Linear(hidden_dim*2, hidden_dim)
    embedding = nn.Embedding(len(vocab)-1, embedding_dim, padding_idx=0, scale_grad_by_freq=True)

    #Easier convergence
    embedding.weight.data.uniform_(-0.1, 0.1)
    linear_dec.bias.data.fill_(0)
    linear_dec.weight.data.uniform_(-0.1, 0.1)

    sos_token = torch.tensor(vocab['<sos>']).to(device)
    eos_token = torch.tensor(vocab['<eos>']).to(device)

    with torch.no_grad():
        completed_sentences = []
        completed_scores = []
        encoder.eval()
        decoder.eval()

        enc_output, enc_hidden = encoder(prev_utterance, prev_utt_length, visual_context, target_img_feats)
        dec_hidden = linear_dec(torch.cat((enc_hidden[0], enc_hidden[1]), dim=1))
        dec_hidden = dec_hidden.expand(beam_size, -1)

        empty_count = 0
        gen_len = 0
        dec_input = sos_token.expand(beam_size, 1)
        gen_sentences_beam = dec_input
        top_scores = torch.zeros(beam_size, 1).to(device)

        while True:
            if gen_len > max_len:
                break

            mask = getMaskAttention(prev_utt_length, device)
        
            dec_embed = embedding(dec_input).squeeze(1)
            word_pred, decoder_hidden_state = decoder(dec_embed, dec_hidden, enc_output, mask)
            word_pred = F.log_softmax(word_pred, dim=1)
            word_pred = top_scores.expand_as(word_pred) + word_pred

            if gen_len == 0:
                top_scores, top_words = word_pred[0].topk(beam_size, 0, True, True)
            else:
                top_scores, top_words = word_pred[-1].topk(beam_size, 0, True, True)

            sentence_index = top_words // (len(vocab) - 1)
            word_index = top_words % (len(vocab) - 1)

            gen_len += 1

            gen_sentences_beam = torch.cat((gen_sentences_beam[sentence_index], word_index.unsqueeze(1)), dim=1)

            inc_sentence_ids = [inc for inc in range(len(gen_sentences_beam))
                if eos_token not in gen_sentences_beam[inc]]
            complete_sent_ids = list(set(range(len(gen_sentences_beam))) - set(inc_sentence_ids))

            if len(complete_sent_ids)  > 0:
                completed_sentences.extend(gen_sentences_beam[sentence_index[complete_sent_ids]]).toList()
                completed_scores.extend(top_scores[complete_sent_ids])
                beam_size -= len(complete_sent_ids)

            if beam_size == 0:
                break

            gen_sentences_beam = gen_sentences_beam[inc_sentence_ids]

            dec_hidden = decoder_hidden_state[sentence_index[inc_sentence_ids]]
            top_scores = top_scores[inc_sentence_ids].unsqueeze(1)
            dec_input = word_index[inc_sentence_ids]

        if len(complete_sent_ids) == 0:
            empty_count += 1
            # all incomplete here
            completed_sentences.extend((gen_sentences_beam[inc_sentence_ids].tolist()))
            completed_scores.extend(top_scores[inc_sentence_ids])

        _, sorted_indices = torch.sort(torch.tensor(completed_scores), descending=True)
        best_seq = completed_sentences[sorted_indices[0]]
        
        hypothesis = [vocab.index2word[w] for w in best_seq if w not in 
            [vocab.word2index['<sos>'], vocab.word2index['<eos>'], vocab.word2index['<pad>']]]
        
        hypothesis_string = ' '.join(hypothesis)
        return hypothesis_string






