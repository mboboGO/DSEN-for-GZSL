import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import deepdish as dd
from  MPNCOV import MPNCOV
import torch.nn.functional as F
import numpy as np

__all__ = ['dsen']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://drive.google.com/file/d/132PzY3eVDuGg8ROz5wON5FTC2E2o12Ck/view',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
        
        
class Model(nn.Module):
    def __init__(self, block, layers, params=None):
    
        ''' params '''
        self.inplanes = 64
        self.num_C = params['num_classes']
        is_fix = params['is_fix']
        self.s_C = params['sf_size']
        
        super(Model, self).__init__()
        
        ''' backbone net'''
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        
        if(is_fix):
            for p in self.parameters():
                p.requires_grad=False
        
        # pro
        self.proj = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(28, stride=1)
        )
                
        # s2v_pro
        self.s2v_c = nn.Sequential(
            nn.Linear(self.s_C,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,2048),
            nn.LeakyReLU()
        )
        self.s2v_s = nn.Sequential(
            nn.Linear(self.s_C,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,2048),
            nn.LeakyReLU()
        )
        self.s2v_u = nn.Sequential(
            nn.Linear(self.s_C,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,2048),
            nn.LeakyReLU()
        )
        
        # recon
        self.rec = nn.Sequential(
            nn.Linear(2048,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,self.s_C),
            nn.LeakyReLU()
        )
        
        ''' fc '''
        self.cls_w = torch.nn.Parameter(torch.Tensor(2048, self.num_C).cuda(), requires_grad=True)
        self.cls_b = torch.nn.Parameter(torch.Tensor(1, self.num_C).cuda(), requires_grad=True)
        nn.init.kaiming_uniform_(self.cls_w, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.cls_w)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.cls_b, -bound, bound)
        
        self.classifier = nn.Linear(2048, self.num_C)
		
        
        ''' params ini '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, sf):
        # backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        '''v2s'''
        vf_emb = self.proj(x).view(x.size(0),-1)
        
        ''' rec '''
        sf_emb_s = self.s2v_c(sf)+self.s2v_s(sf)
        sf_emb_u = self.s2v_c(sf)+self.s2v_u(sf)
        
        sf_rec_s = self.rec(sf_emb_s)
        sf_rec_u = self.rec(sf_emb_u)
        
        ''' cls '''
        logit_s = self.classifier(vf_emb)
        logit_u = self.classifier(sf_emb_u)
        
        return (vf_emb,sf_emb_s,sf_emb_u,sf_rec_s,sf_rec_u),(logit_s,logit_u)
		
class LOSS(nn.Module):
    def __init__(self, params):
        super(LOSS, self).__init__()
		
        self.mse_loss = nn.MSELoss()
        self.cls_loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

        # loss weights
        self.w_rec = params[0]
        self.alpha = params[1]

    def forward(self, label,sf,feat,logits,seen_c,unseen_c):
        vf_emb = feat[0]
        sf_emb_s = feat[1]
        sf_emb_u = feat[2]
        sf_rec_s = feat[3]
        sf_rec_u = feat[4]
        
        logit_s = logits[0]
        logit_u = logits[1] 

        ''' rank loss '''
        vf_norm = F.normalize(vf_emb, p=2, dim=1)
        sf_norm = F.normalize(sf_emb_s, p=2, dim=1)
        sf_norm_pos = sf_norm[label,:]
        L_rank = (1-torch.sum(vf_norm*sf_norm_pos,dim=1)).mean()
        
        vf_norm = F.normalize(vf_emb, p=2, dim=1)
        sf_norm = F.normalize(sf_emb_u, p=2, dim=1)
        sf_norm_pos = sf_norm[label,:]
        L_rank += (1-torch.sum(vf_norm*sf_norm_pos,dim=1)).mean()
        
        ''' cls '''
        logit_u = self.softmax(logit_u)
        logit_u_sum = torch.sum(logit_u[:,unseen_c],dim=1)
        logit_u = -torch.log10(logit_u_sum)
        
        L_cls =  self.cls_loss(logit_s,label) + self.alpha*torch.mean(logit_u)
        
        ''' rec '''
        L_rec = self.w_rec*(self.mse_loss(sf_rec_s[seen_c,...],sf[seen_c,:]) + self.mse_loss(sf_rec_u[unseen_c,...],sf[unseen_c,:]))
        

        # loss sum
        L_all = L_rank + L_cls + L_rec
        
        return (L_all,L_rank,L_cls,L_rec)
		
def dsen(pretrained=False, lws=None, params=None):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Model(Bottleneck, [3, 4, 23, 3], params=params)
    loss_model = LOSS(lws)
    if pretrained:
        model_dict = model.state_dict()
        #pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        pretrained_dict = torch.load('./pretrained/resnet101-5d3b4d8f.pth')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model,loss_model
	
	
	
