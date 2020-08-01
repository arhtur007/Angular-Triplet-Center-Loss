import torch
import torch.nn as nn


class AngularTripletCenterLoss(nn.Module):

    def __init__(self, margin=0.5, eps=1e-7, spkr_num=20, utt_num=10):
        super(AngularTripletCenterLoss, self).__init__()
        self.margin = margin
        self.eps = eps
        self.spkr_num = spkr_num
        self.utt_num = utt_num

    def bdot(self, a, b):
        B = a.shape[0]
        S = a.shape[1]
        return torch.bmm(a.view(B, 1, S), b.view(B, S, 1)).reshape(-1)

    def forward(self, x):
        centroids = torch.mean(x, 1)

        centroids = torch.nn.functional.normalize(centroids, p=2, dim=1)

        intra_centroids = torch.cat(self.utt_num * [centroids], -1)
        intra_centroids = intra_centroids.view(self.spkr_num * self.utt_num, 64)

        x = x.view(self.spkr_num * self.utt_num, 64)
        intra_d = torch.acos(torch.clamp(self.bdot(x, intra_centroids), -1.+self.eps, 1-self.eps))

        intra_d = intra_d.view(self.spkr_num, self.utt_num)
        intra_d, intra_idx = torch.max(intra_d, 1)

        idx = []
        for i in range(self.spkr_num):
            idx.append(intra_idx[i] + i*self.utt_num)
        idx = torch.stack(idx)
        maxd_x = x[idx]
        maxd_x = torch.cat((self.spkr_num-1) * [maxd_x], -1)
        maxd_x = maxd_x.view(self.spkr_num * (self.spkr_num-1), 64)
        temp_centroids = torch.cat(self.spkr_num * [centroids])

        inter_centroids = torch.tensor([])
        n = self.spkr_num+1
        for i in range(self.spkr_num):
            inter_centroids = torch.cat([inter_centroids, temp_centroids[i*n+1:i*n+n]])

        inter_d = torch.acos(torch.clamp(self.bdot(maxd_x, inter_centroids), -1.+self.eps, 1-self.eps))
        inter_d = inter_d.view(self.spkr_num, self.spkr_num - 1)
        inter_d, inter_idx = torch.min(inter_d, 1)

        dist_hinge = torch.clamp(self.margin + intra_d - inter_d, min=0.0)

        loss = torch.sum(dist_hinge)

        return loss

