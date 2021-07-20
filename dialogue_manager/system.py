from transformers import BertTokenizer, BertModel
import torch
import json
from models.listener.run_model import evaluate_listener_game
from models.speaker.run_model import evaluate_speaker_game

#Loading a bert model and tokeniser to convert utterances to bert reps
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()
model.to(device)

with open('../dataset/v2/vectors.json', 'r') as f:
            image_features = json.load(f)

def get_bert_rep(text, model, tokenizer):
    # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    input_tensors = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])

    segments_ids = [0] * input_tensors.shape[1]
    segments_tensors = torch.tensor([segments_ids])

    input_tensors = input_tensors.to(device)
    segments_tensors = segments_tensors.to(device)

    with torch.no_grad():
        outputs = model(input_tensors, token_type_ids=segments_tensors)
        encoded_layers = outputs[0]
    assert tuple(encoded_layers.shape) == (1,  input_tensors.shape[1], model.config.hidden_size)
    return encoded_layers

def getListenerPredictions(image_set, message):
    img_dim = 2048
    img_count = 6
    encoded_utt = get_bert_rep(message, model, tokenizer)
    separate_images = torch.zeros(img_count, img_dim)

    im_counter = 0
    for im in image_set:
        img_id = str(int(im.split('.')[0].split('_')[2]))
        separate_images[im_counter] = torch.tensor(image_features[img_id])
        im_counter += 1
    
    visual_context = separate_images.reshape(img_dim * img_count)
    visual_context = visual_context.unsqueeze(0).to(device)
    separate_images = separate_images.unsqueeze(0).to(device)
    preds = evaluate_listener_game(encoded_utt, separate_images, visual_context, [[]])
    top_scores, top_words = preds[0].topk(3, 0, True, True)
    return top_scores.squeeze(1).tolist(), top_words.squeeze(1).tolist()

def getSpeakerMessage(image_set, message, target_image_index):
    img_dim = 2048
    img_count = 6
    separate_images = torch.zeros(img_count, img_dim)

    im_counter = 0
    for im in image_set:
        img_id = str(int(im.split('.')[0].split('_')[2]))
        separate_images[im_counter] = torch.tensor(image_features[img_id])
        im_counter += 1
    visual_context = separate_images.reshape(img_dim * img_count)
    visual_context = visual_context.unsqueeze(0).to(device)

    target_image = image_set[target_image_index]
    target_image_id = str(int(target_image.split('.')[0].split('_')[2]))
    target_image_feats = torch.tensor(image_features[target_image_id])
    target_image_feats = target_image_feats.unsqueeze(0).to(device)
    return evaluate_speaker_game(visual_context, target_image_feats, message, 3)

    


image_set = ["COCO_train2014_000000220226.jpg", "COCO_train2014_000000449919.jpg", "COCO_train2014_000000039360.jpg", "COCO_train2014_000000326499.jpg", "COCO_train2014_000000580257.jpg", "COCO_train2014_000000239417.jpg"]
print(getSpeakerMessage(image_set, "Old lady sitting on a bench", 4))


