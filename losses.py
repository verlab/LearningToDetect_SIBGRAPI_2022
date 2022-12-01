import torch
import torch.nn as nn
import torch.nn.functional as F

def cossim2(score_map, ground_truth, score_map_base, ground_truth_base):
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    out = 1 - cos(torch.flatten(score_map), torch.flatten(ground_truth))
    outB = 1 - cos(torch.flatten(score_map_base), torch.flatten(ground_truth_base))
    return (out + outB)/2

def usq(img):
    return torch.unsqueeze(torch.unsqueeze(img, 0), 0 )

def peak_loss2(score_map, score_map_base, ground_truth, ground_truth_base, N):
    maxpool = nn.MaxPool2d(N, stride=N, padding=N//2)
    avgpool = nn.AvgPool2d(N, stride=N, padding=N//2)
    
    gt_avg = avgpool(usq(ground_truth))
    gt_avg_base = avgpool(usq(ground_truth_base))
    filter = (gt_avg != 0)
    filter_base = (gt_avg_base != 0)
    peak = 1 - ( maxpool(usq(score_map)) - avgpool(usq(score_map))).masked_select(filter).mean()
    peak_base = 1 - (maxpool(usq(score_map_base)) - avgpool(usq(score_map_base))).masked_select(filter_base).mean()
    
    return (peak.mean() + peak_base.mean())/2

def L2_loss2(score_map, ground_truth, score_map_base, ground_truth_base):
    loss = nn.MSELoss()
    return (loss(score_map, ground_truth) + loss(score_map_base, ground_truth_base))/2
