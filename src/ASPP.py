import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_channel, depth):
        super(ASPP,self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Sequential(
            nn.Conv2d(in_channel, depth, 1, 1), 
            nn.BatchNorm2d(depth), 
            nn.ReLU())
        self.atrous_block6 = nn.Sequential( nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6), 
                                            nn.BatchNorm2d(depth),
                                            nn.ReLU())
        self.atrous_block12 = nn.Sequential(nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12),
                                            nn.BatchNorm2d(depth),
                                            nn.ReLU())
        
        self.atrous_block18 = nn.Sequential(nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18), 
                                            nn.BatchNorm2d(depth),
                                            nn.ReLU())
        
        self.conv_1x1_output = nn.Sequential(nn.Conv2d(depth * 5, depth, 1, 1),
                                nn.BatchNorm2d(depth),
                                nn.ReLU())
 
    def forward(self, x):
        size = x.shape[2:]
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')
        
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net