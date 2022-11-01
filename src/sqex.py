import torch
from torch import nn

class squeeze_excite(torch.nn.Module):
    def __init__(self, in_channels, size=1, reduction="/16", activation=torch.nn.GELU):
        super(squeeze_excite, self).__init__()
        # squeeze part
        self.in_channels = in_channels
        # average adaaptive pooling to change the data to torch.size([1,1,C])
        self.avg = torch.nn.AdaptiveAvgPool2d(1)
        if type(reduction) == str:
            self.reductionsize = self.in_channels // int(reduction[1:])
        else:
            self.reductionsize = reduction  # they use this part
        # excitation part
        self.net = nn.Sequential(
            nn.Linear(self.in_channels , self.reductionsize, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.reductionsize, self.in_channels , bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        b, c, *l = inputs.shape
        x_ = self.avg(inputs).view(b, c)
        x_ = self.net(x_).view(b, c, 1, 1)
        x_ = inputs * x_
        return x_
    
class ResBlockSqEx(nn.Module):

    def __init__(self, n_features):
        super(ResBlockSqEx, self).__init__()

        # convolutions

        self.norm1 = nn.BatchNorm2d(n_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1, bias=False)

        self.norm2 = nn.BatchNorm2d(n_features)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1, bias=False)

        # squeeze and excitation

        self.sqex  = squeeze_excite(n_features)

    def forward(self, x):
        
        # convolutions

        y = self.conv1(self.relu1(self.norm1(x)))
        y = self.conv2(self.relu2(self.norm2(y)))

        # squeeze and excitation

        y = self.sqex(y)

        # add residuals
        
        y = torch.add(x, y)

        return y