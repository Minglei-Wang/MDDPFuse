import torch
import torch.nn as nn
import torch.nn.functional as F

class layer_contrast(nn.Module):
    def __init__(self):
        super(layer_contrast, self).__init__()

    def forward(self, feature_shuffle):
        mean_vector = torch.mean(feature_shuffle, [1, 2], keepdim=True)
        feature_contrast = torch.mean((feature_shuffle - mean_vector) ** 2, [1, 2], keepdim=True).sqrt()
        contrast_vector = torch.mean(feature_contrast, [1, 2], keepdim=True)
        feature_fusion_enhancement = contrast_vector * feature_shuffle
        return feature_fusion_enhancement



def CMDAF(vi_feature, ir_feature):
    sigmoid = nn.Sigmoid()
    gap = nn.AdaptiveAvgPool2d(1)
    batch_size, channels, _, _ = vi_feature.size()

    vi_att = torch.matmul(vi_feature.view(batch_size, channels, -1), vi_feature.view(batch_size, channels, -1).permute(0, 2, 1))
    ir_att = torch.matmul(ir_feature.view(batch_size, channels, -1), ir_feature.view(batch_size, channels, -1).permute(0, 2, 1))
    cross_att = torch.matmul(vi_feature.view(batch_size, channels, -1), ir_feature.view(batch_size, channels, -1).permute(0, 2, 1))
    vi_att = F.softmax(vi_att, dim=-1)
    ir_att = F.softmax(ir_att, dim=-1)
    cross_att = F.softmax(cross_att, dim=-1)
    vi_feature = torch.matmul(vi_att, vi_feature.view(batch_size, channels, -1)).view(batch_size, channels, _, _) + torch.matmul(cross_att, ir_feature.view(batch_size, channels, -1)).view(batch_size, channels, _, _)
    ir_feature = torch.matmul(ir_att, ir_feature.view(batch_size, channels, -1)).view(batch_size, channels, _, _) + torch.matmul(cross_att.permute(0, 2, 1), vi_feature.view(batch_size, channels, -1)).view(batch_size, channels, _, _)

    vi_weight = gap(vi_feature)
    ir_weight = gap(ir_feature)
    vi_weight = F.softmax(vi_weight, dim=1)
    ir_weight = F.softmax(ir_weight, dim=1)
    vi_feature = vi_feature * vi_weight
    ir_feature = ir_feature * ir_weight

    feature = torch.cat([vi_feature, ir_feature], 1)

    return feature

