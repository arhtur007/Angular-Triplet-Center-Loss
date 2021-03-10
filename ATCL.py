import torch
import torch.nn as nn
import math
import numpy as np
from torch.autograd import Function


class AngularTripletCenterMarginLoss(Function):

    def __init__(self, device, margin=0.5, eps=1e-7, class_num=20, embedding_num=10, embedding_dim=64):
        super(AngularTripletCenterMarginLoss, self).__init__()
        self.margin = margin
        self.eps = eps
        self.class_num = class_num
        self.embedding_num = embedding_num
        self.embedding_dim = 64
        self.device = device
        torch.autograd.set_detect_anomaly(True)

    def bdot(self, a, b):
        B = a.shape[0]
        S = a.shape[1]
        return torch.bmm(a.view(B, 1, S), b.view(B, S, 1)).reshape(-1)

    def forward(self, x):
        centroids = torch.mean(x, 1)

        centroids = torch.nn.functional.normalize(centroids, p=2, dim=1)

        intra_centroids = torch.cat(self.embedding_num * [centroids], -1)
        intra_centroids = intra_centroids.view(self.class_num * self.embedding_num, embedding_dim)

        x = x.view(self.class_num * self.embedding_num, embedding_dim)
        intra_d = torch.acos(torch.clamp(self.bdot(x, intra_centroids), -1.+self.eps, 1-self.eps))

        # dist_hinge = torch.clamp(intra_d - self.margin, min=0.0)

        # loss = torch.sum(dist_hinge)
        # return loss
        intra_d = intra_d.view(self.class_num, self.embedding_num)
        intra_d, intra_idx = torch.max(intra_d, 1)

        idx = []
        for i in range(self.class_num):
            idx.append(intra_idx[i] + i*self.embedding_num)
        idx = torch.stack(idx)
        maxd_x = x[idx]
        maxd_x = torch.cat((self.class_num-1) * [maxd_x], -1)
        maxd_x = maxd_x.view(self.class_num * (self.class_num-1), embedding_dim)
        temp_centroids = torch.cat(self.class_num * [centroids])

        inter_centroids = torch.tensor([]).to(self.device)
        n = self.class_num+1
        for i in range(self.class_num):
            inter_centroids = torch.cat([inter_centroids, temp_centroids[i*n+1:i*n+n]])

        inter_d = torch.acos(torch.clamp(self.bdot(maxd_x, inter_centroids), -1.+self.eps, 1-self.eps))
        inter_d = inter_d.view(self.class_num, self.class_num - 1)
        inter_d, inter_idx = torch.min(inter_d, 1)

        dist_hinge = torch.clamp(self.margin + intra_d - inter_d, min=0.0)

        loss = torch.sum(dist_hinge)

        return loss


if __name__ == "__main__":
    # criterion = AngularTripletMarginLoss(margin=0.4)
    t = torch.tensor([[2, 1], [4, 2]])
    print(t)
    m = torch.max(t, 1)
    print(m)
    print(t[m[1]])
    # print(t.view(3, 3))
    # t = t.view(4, 3)
    # print(t)
    # t = t.view(2, 2, 3)
    # print(t)
    # print(t)
    # centroids = torch.mean(t, 1)
    # print(centroids)
    # print(torch.nn.functional.normalize(centroids, p=2, dim=1))
    # a = torch.randn(60, 10, 64)
    # b = torch.randn(3, 4)
    # c = torch.randn(3, 4)
    # print(torch.mean(a, 1).shape)
    # print(a)
    # print(b)
    # print(torch.dot(a[0], b[0]))
    # print(bdot(a, b))
    # print(criterion.forward(a, b, c))
    # print(torch.acos(a))
