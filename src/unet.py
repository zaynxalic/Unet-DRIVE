from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .CBAM import *
from .dropblock import DropBlock2D
from .ASPP import ASPP
from .sqex import squeeze_excite as sqex

if torch.cuda.is_available():
    device = torch.device(f'cuda:{torch.cuda.device_count()-1}')
else:
    device = torch.device('cpu')
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout = 0.18, block_size = 7, residual = False):
        super(DoubleConv, self).__init__()
        self.residual = residual
        if mid_channels is None:
            mid_channels = out_channels
            
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            DropBlock2D(drop_prob = dropout, block_size = block_size) if dropout else DropBlock2D(0.,  None),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            DropBlock2D(drop_prob = dropout, block_size = block_size) if dropout else DropBlock2D(0., None),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        if self.residual:
            self.c2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_channels),
            )
            self.sqex = sqex(out_channels)
            
    def forward(self, inputs):
        if self.residual:
            x = inputs
            inputs = self.c1(inputs)
            x = self.sqex(self.c2(x) + inputs)
            return x
        else:
            return self.c1(inputs)
        
class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )
        
class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels, residual = False):
        super(Down, self).__init__(
            nn.MaxPool2d(2), # devide the channel in halves
            DoubleConv(in_channels, out_channels, residual = residual) # afterwards, perform two convolutions
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            # Double the size of image
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # This parts involve skip connections
        # where concat current and previous network channels
        x1 = self.up(x1)
        # if they don't have the same shape then do the padding.
        
        if x2.shape != x1.shape:
            diff_y = x2.size()[2] - x1.size()[2]
            diff_x = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64, 
                 is_cbam: bool = False,
                 is_aspp: bool = False,
                 is_sqex: bool = False):
        """
        Args:
            in_channels (int, optional): The number of input channels. Defaults to 1.
            num_classes (int, optional): The number of classes in this case we only have 2 classes for classification. Defaults to 2.
            bilinear (bool, optional): . Defaults to True.
            base_c (int, optional): _description_. Defaults to 64.
            is_cbam (bool, optional): Whether involves cbam settings. Defaults to False.
            cbam_layers (bool, optional): Whether involves  
        """
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.is_cbam = is_cbam
        # self.channel_lists = [for i in range()]
        # at first the network is in the double convolution
        self.in_conv = DoubleConv(in_channels, base_c, dropout = None, residual = is_sqex ) # in_channels = 3, base_c = 64
        if self.is_cbam:
            self.cbam1 = CBAM(base_c)
    
        self.down1 = Down(base_c, base_c * 2, residual = is_sqex)
        if self.is_cbam:
            self.cbam2 = CBAM(base_c)
            
        self.is_aspp = is_aspp
        if self.is_aspp:
            self.aspp = ASPP(base_c * 8, base_c * 8)
        self.down2 = Down(base_c * 2, base_c * 4, residual = is_sqex)
        self.down3 = Down(base_c * 4, base_c * 8, residual = is_sqex)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor, residual = is_sqex)
        # [1024, 512, 1]
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x) # the first layer for 
        if self.is_cbam: 
            x1 = self.cbam1(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.is_aspp:
            x5 = self.aspp(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)
        return {"out": logits}


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{torch.cuda.device_count()-1}')
    else:
        device = torch.device('cpu')
    model = UNet(in_channels=3, num_classes=2, base_c=32).to(device)
    
    x = torch.randn((32,3,480,480)).to(device)
    model(x)