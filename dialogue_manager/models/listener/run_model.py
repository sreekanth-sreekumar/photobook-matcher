import torch
from .model import ListenerModel

def getMaskAttention(tokens_length, device):
    masks = [[False] * tokens_length]
    masks = torch.tensor(masks).unsqueeze(2).to(device)
    return masks

def evaluate_listener_game(utterance_reps, separate_images, visual_context, prev_hist):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    listener_model_file = './saved_models/listener_3_6.pkl'     #Substitute with equivalent file name
    
    img_dim = 2048
    attention_dim = 512
    hidden_dim = 512
    dropout_prob = 0.5
    embedding_dim = 768
    model = ListenerModel(embedding_dim, hidden_dim, img_dim, attention_dim, dropout_prob)
    softmax = torch.nn.Softmax(dim=1)

    checkpoint = torch.load(listener_model_file, map_location=device)
    model.load_state_dict(checkpoint['model'])
    
    with torch.no_grad():
        model.eval()
        print(utterance_reps.shape[1])
        masks = getMaskAttention(utterance_reps.shape[1], device)

        out = model(utterance_reps, separate_images, visual_context, masks, prev_hist, device)
        return softmax(out)