from torch import nn
import torch
import torch.nn.functional as F
import difflib
import time

class SM_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.SM = difflib.SequenceMatcher()
    
    def forward(self, tensor_1, raw_texts):
        bs = tensor_1.size(0)
        tensor_1 = tensor_1.view(bs, -1)
        tensor_2 = torch.ones(tensor_1.size()).cuda()
        tensor_2[:bs-1, :] = tensor_1.detach()[1:, :]
        tensor_2[bs-1, :] = tensor_1.detach()[0, :]
        
        seq_sim_score = torch.ones((1,bs)).cuda()
        
        for i in range(bs-1):
            self.SM.set_seq1(raw_texts[i])
            self.SM.set_seq2(raw_texts[i+1])
            seq_sim_score[0,i] = self.SM.ratio()

        self.SM.set_seq1(raw_texts[bs-1])
        self.SM.set_seq2(raw_texts[0])
        seq_sim_score[0, bs-1] = self.SM.ratio()

        dist_score = F.pairwise_distance(tensor_1, tensor_2)
        dist_sim_score = 1. / (1.+dist_score)
        dist_sim_score = dist_sim_score.unsqueeze(0)
        loss = F.pairwise_distance(dist_sim_score, seq_sim_score, p=1)
        loss = loss / bs

        return loss
    

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        #anchor = F.normalize(anchor, p=2, dim=1)
        #positive = F.normalize(positive, p=2, dim=1)
        #negative = F.normalize(negative, p=2, dim=1)
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses
    
class TripletLossVarMargin(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self):
        super(TripletLossVarMargin, self).__init__()

    def forward(self, anchor, positive, negative, margin):
        #anchor = F.normalize(anchor, p=2, dim=1)
        #positive = F.normalize(positive, p=2, dim=1)
        #negative = F.normalize(negative, p=2, dim=1)
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        #print(distance_negative)
        #print(distance_positive)
        losses = F.relu(distance_positive - distance_negative + margin)
        return losses