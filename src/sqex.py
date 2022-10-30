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