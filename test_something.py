from re import I
import torch
import torch.nn as nn
from mmdet.core.bbox.iou_calculators import bbox_overlaps,wasserstein_dist
import torch.nn.functional as F


bboxes1 = torch.rand(8,256,80,80)
bboxes2 = torch.rand(8,256,80,80)
bboxes1 = bboxes1.view(8,256,-1)
bboxes2 = bboxes2.view(8,256,-1)
# overlaps = bbox_overlaps(bboxes1, bboxes2)
# x = bboxes1
# y = bboxes2
# x=F.normalize(x,dim=-1)
# y=F.normalize(y,dim=-1)
# dxy=torch.clamp(torch.sum(x*y,dim=-1),-1,1)
# d=torch.acos(dxy)
# print(d)
loss = F.cosine_similarity(bboxes1,bboxes2,dim = 2)
print(torch.mean(loss))
