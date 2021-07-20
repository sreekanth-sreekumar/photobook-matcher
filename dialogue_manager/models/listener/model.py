import torch.nn as nn
import torch

class ListenerModel(nn.Module):
    def __init__(self, embed_dim, hid_dim, img_dim, att_dim, dropout_prob):
        super().__init__()

        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.hidden_dim = hid_dim

        self.lin_viscontext = nn.Linear(6 * img_dim, hid_dim)
        self.lin_emb2hid = nn.Linear(embed_dim, hid_dim)
        self.lin_mm = nn.Linear(hid_dim * 2, hid_dim)
        self.lin_separate = nn.Linear(img_dim, hid_dim)

        self.lin_att1 = nn.Linear(hid_dim, att_dim)
        self.lin_att2 = nn.Linear(att_dim, 1)

    def init_weights(self):
        for ll in [self.lin_viscontext, self.lin_emb2hid, self.lin_mm, self.lin_separate, self.lin_att1, self.lin_att2]:
            ll.bias.data.fill_(0)
            ll.weight.data.uniform_(-0.1, 0.1)

    def forward(self, reps, separate_imgs, visual_context, masks, prev_hist, device):

        batch_size = reps.shape[0]

        #Images to hidden dims
        visual_context = self.dropout(visual_context)
        projected_context = self.relu(self.lin_viscontext(visual_context))

        #Input representations to hidden dims 
        reps = self.dropout(reps)
        input_reps = self.relu(self.lin_emb2hid(reps))

        #Creating multimodal inputs by concatination visual context with each bert reps
        repeated_context = projected_context.unsqueeze(1).repeat(1, input_reps.shape[1], 1)
        mm_context = self.relu(self.lin_mm(torch.cat((input_reps, repeated_context), dim=2)))

        #Applying attention over multimodel context and masking output
        att_output = self.lin_att2(self.tanh(self.lin_att1(mm_context)))
        att_output = att_output.masked_fill_(masks, float('-inf'))

        #Final encoder context representation
        att_weights = self.softmax(att_output)
        attended_hid = (mm_context * att_weights).sum(dim=1)

        # image features per image in context are processed
        separate_imgs = self.dropout(separate_imgs)
        separate_imgs = self.lin_separate(separate_imgs)

        #Apply history to candidate images
        for b in range(batch_size):

            cur_history = prev_hist[b]
            for s in range(len(cur_history)):
                if (len(cur_history[s]) > 0):
                    #If there is history for a candidate image
                    hist_rep = torch.stack(cur_history[s]).to(device)

                    #Take the average history vector
                    hist_avg = self.dropout(hist_rep.sum(dim=0)/hist_rep.shape[0])

                    #Process the history representation and add to image representations
                    separate_imgs[b][s] += self.relu(self.lin_emb2hid(hist_avg))
                
        #Taking dot product between multimodal attended context and separate imgs with prev history
        dot = torch.bmm(separate_imgs, attended_hid.view(batch_size, self.hidden_dim, 1))
        return dot






