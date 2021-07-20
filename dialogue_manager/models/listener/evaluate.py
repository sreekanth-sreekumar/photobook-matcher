import torch
import numpy as np

def evaluate(split_loader, model, isValidate, device, mask_attn, data_len):
    accuracies = []
    for i, data in enumerate(split_loader):
        utterances_BERT = data['representations']
        context_separate = data['separate_images']
        context_concat = data['concat_context']
        lengths = data['length']
        targets = data['target']
        prev_hist = data['prev_histories']

        max_length_tensor = utterances_BERT.shape[1]
        masks = mask_attn(lengths, max_length_tensor, device)

        out = model(utterances_BERT, context_separate, context_concat, masks, prev_hist, device)
        preds = torch.argmax(out, dim=1) #Maximum of the 6 images

        correct = torch.eq(preds, targets).sum()
        accuracies.append(float(correct))
    
    sum_accuracy = np.sum(accuracies)
    cur_acc = sum_accuracy/data_len

    print('Accuracy: ', round(cur_acc, 5))

    if isValidate:
        return cur_acc