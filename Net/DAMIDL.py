import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import torch


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.cs = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.cat([avg_out, max_out], dim=1)
        a = self.cs(a)
        return x*a


class AttentionBlock(nn.Module):
    def __init__(self, patch_num):
        super(AttentionBlock, self).__init__()
        self.patch_num = patch_num
        self.GlobalAveragePool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.GlobalMaxPool = nn.AdaptiveMaxPool3d(output_size=(1, 1, 1))
        self.Attn = nn.Sequential(
            nn.Conv3d(self.patch_num, self.patch_num // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.patch_num // 2, self.patch_num, kernel_size=1)
        )
        self.pearson_attn = nn.Linear(self.patch_num - 1, 1)

    def forward(self, input, patch_pred):
        mean_input = input.mean(2)
        attn1 = self.Attn(self.GlobalAveragePool(mean_input))
        attn2 = self.Attn(self.GlobalMaxPool(mean_input))
        patch_pred = patch_pred.unsqueeze(-1)
        patch_pred = patch_pred.unsqueeze(-1)
        patch_pred = patch_pred.unsqueeze(-1)
        a = attn1 + attn2 + patch_pred
        a = torch.sigmoid(a)
        return mean_input*a, a.flatten(1)


class BaseNet(nn.Module):
    def __init__(self, feature_depth):
        super(BaseNet, self).__init__()
        self.feature_depth = feature_depth
        self.spatial_attention = SpatialAttention()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(1, self.feature_depth[0], kernel_size=4)),
            ('norm1', nn.BatchNorm3d(self.feature_depth[0])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(self.feature_depth[0], self.feature_depth[1], kernel_size=3)),
            ('norm2', nn.BatchNorm3d(self.feature_depth[1])),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool3d(kernel_size=2)),
            ('conv3', nn.Conv3d(self.feature_depth[1], self.feature_depth[2], kernel_size=3)),
            ('norm3', nn.BatchNorm3d(self.feature_depth[2])),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv3d(self.feature_depth[2], self.feature_depth[3], kernel_size=3)),
            ('norm4', nn.BatchNorm3d(self.feature_depth[3])),
            ('relu4', nn.ReLU(inplace=True)),
        ]))
        self.classify = nn.Sequential(
            nn.Linear(self.feature_depth[3], 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        local_feature = self.features(input)
        attended_feature = self.spatial_attention(local_feature)
        feature_ = F.adaptive_avg_pool3d(local_feature, (1, 1, 1))
        score = self.classify(feature_.flatten(1, -1))
        return [attended_feature, score]


class DAMIDL(nn.Module):
    def __init__(self, patch_num=60, feature_depth=None):
        super(DAMIDL, self).__init__()
        self.patch_num = patch_num
        if feature_depth is None:
            feature_depth = [32, 64, 128, 128]
        self.patch_net = BaseNet(feature_depth)
        self.attention_net = AttentionBlock(self.patch_num)
        self.reduce_channels = nn.Sequential(
            nn.Conv3d(self.patch_num, 128, kernel_size=2),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            nn.Conv3d(128, 64, kernel_size=2),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, input):
        patch_feature, patch_score = [], []
        for i in range(self.patch_num):
            feature, score = self.patch_net(input[i])
            feature = feature.unsqueeze(1)
            patch_feature.append(feature)
            patch_score.append(score)
        feature_maps = torch.cat(patch_feature, 1)
        patch_scores = torch.cat(patch_score, 1)
        attn_feat, ca = self.attention_net(feature_maps, patch_scores)
        features = self.reduce_channels(attn_feat).flatten(1)
        subject_pred = self.fc(features)
        return subject_pred
