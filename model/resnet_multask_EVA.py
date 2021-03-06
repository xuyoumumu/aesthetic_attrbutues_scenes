'''
    date:2020/10/14
    多任务网络用于预测属性分布以及场景分布
    使用EVA数据集
'''


import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
from torch.autograd import Variable




class PoolFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_maps = None

    def forward(self, inp):
        kernel_size = (inp.size()[2], inp.size()[3])
        self.feature_maps = F.avg_pool2d(inp, kernel_size)
        return inp


class FeaturesMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_maps = None

    def forward(self, inp):
        self.feature_maps = inp
        return inp


class ResNetGAPFeatures(nn.Module):
    def __init__(self, resnet, n_features=9, num_class = 6):
        super().__init__()
        self.model = nn.Sequential(*list(resnet.children())[:4])
        self.all_features = []
        self.all_pooled_features = []
        self.attribute_weights = nn.Linear(15104, n_features)
        self.content_weights = nn.Linear(15104, num_class)
        self.weights = torch.nn.Parameter(torch.ones(6,9).requires_grad_())
        self.Score_1 = nn.Linear(9,4)
        self.Score_2 = nn.Linear(4,2)
        self.Score_3 = nn.Linear(2,1)
        count = 0
        for i, mod in enumerate(list(resnet.children())):
            # Extract the bottleneck layers
            if isinstance(mod, nn.Sequential):
                for bn in mod:
                    self.model.add_module(f"bn_{count}", bn)

                    # Use "Transparent layers and save references to their objects for later use"
                    pooled_feature_map = PoolFeatures()
                    feature_map = FeaturesMap()
                    self.model.add_module(f"pooled_feature_{count}", pooled_feature_map)
                    self.model.add_module(f"feature_map_{count}", feature_map)
                    self.all_pooled_features.append(pooled_feature_map)
                    self.all_features.append(feature_map)
                    count += 1

    def forward(self, inp):
        _ = self.model(inp)
        features = torch.cat([pool_fp.feature_maps for pool_fp in self.all_pooled_features], dim=1).squeeze()
        features = F.dropout(features, p=0.5)
        if len(features.size()) == 1:
            features = features.unsqueeze(0)

        # Use features to predict scores
        attribute_scores = self.attribute_weights(features)

        # The first 9 scores reflect:
        # 'BalancingElements', 'ColorHarmony', 'Content', 'DoF','Light',
        #  'MotionBlur', 'Object', 'RuleOfThirds', 'VividColor'
        # which are between values -1 and 1, hence the tanh non-linearity
        attr = F.tanh(attribute_scores[:, :8])

        # The last 3 scores reflect
        # 'Repetition', 'Symmetry', 'score' which are between values 0 and 1
        # hence the sigmoid non-linearity
        non_neg_attr = F.sigmoid(attribute_scores[:, 8:])
        predictions = torch.cat([attr, non_neg_attr], dim=1)

        content_scores = self.content_weights(features)
        #content_pre = F.softmax(content_scores)

        #print(predictions.shape)
        
        content_score = F.softmax(content_scores,dim =1 )
        #print(content_score.shape)
        content_score  =torch.transpose(content_score,1,0)
        #print(content_score.shape)
        fusion = torch.matmul(content_score,predictions)
        #print(fusion.shape)
        fusion = torch.mul(self.weights,fusion)
        #print(fusion.shape)
        fusion = torch.sum(fusion,dim=0)
        score = self.Score_1(fusion)
        score = self.Score_2(score)
        score = self.Score_3(score)
        score = torch.sigmoid(score)

        return predictions,content_scores,score

def resnet_gap_features():
    resnet = models.resnet50(pretrained=True)
    model = ResNetGAPFeatures(resnet)
    return model
