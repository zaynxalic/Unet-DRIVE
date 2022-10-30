import torch
from torch import nn
import torch.nn.functional as F


from .unet import DoubleConv
from .CBAM import CBAM
from .ASPP import ASPP
# from unet import DoubleConv
# from CBAM import CBAM

class Up(nn.Module):
    """Upscaling and concat"""
    def __init__(self) -> None:
        super(Up, self).__init__()
        # Double the size of image
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

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
        return x
    

class Unetpp(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, is_cbam = False, is_aspp = False) -> None:
        super(Unetpp, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.up = Up()
        self.is_cbam = is_cbam
        self.is_aspp = is_aspp
        
        if self.is_cbam:
            self.CBAM_0 = CBAM(64)
            self.CBAM_1 = CBAM(128)
            self.CBAM_2 = CBAM(96)
            self.CBAM_3 = CBAM(256)
            self.CBAM_4 = CBAM(192)
            self.CBAM_5 = CBAM(128)
        
        if self.is_aspp:
            self.aspp = ASPP(512, 512)
            
        self.conv0_0 = DoubleConv(in_channels, 32, 32)
        self.conv1_0 = DoubleConv(32, 64, 64)
        self.conv2_0 = DoubleConv(64, 128, 128)
        self.conv3_0 = DoubleConv(128, 256, 256)
        self.conv4_0 = DoubleConv(256, 512, 512)

        self.conv0_1 = DoubleConv(32+64, 32, 32)
        self.conv1_1 = DoubleConv(64+128, 64, 64)
        self.conv2_1 = DoubleConv(128+256, 128, 128)
        self.conv3_1 = DoubleConv(256+512, 256, 256)

        self.conv0_2 = DoubleConv(32*2+64, 32, 32)
        self.conv1_2 = DoubleConv(64*2+128, 64, 64)
        self.conv2_2 = DoubleConv(128*2+256, 128, 128)

        self.conv0_3 = DoubleConv(32*3+64, 32, 32)
        self.conv1_3 = DoubleConv(64*3+128, 64, 64)

        self.conv0_4 = DoubleConv(32*4+64, 32, 32)
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, input):
        
        """_summary_
        Input:
            input: input picture I_n (N, C , W, H)
        Returns:
        """
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(self.up(x1_0, x0_0))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(self.up(x2_0, x1_0))
            
        x0_01 = torch.cat([x0_0, x0_1], 1)
        if self.is_cbam:
            x0_01 = self.CBAM_0(x0_01)
        x0_2 = self.conv0_2(self.up(x1_1,x0_01 ))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(self.up(x3_0, x2_0))   
        
        x1_01 =torch.cat([x1_0, x1_1], 1)
        if self.is_cbam:
            x1_01 = self.CBAM_1(x1_01)
        x1_2 = self.conv1_2(self.up(x2_1,x1_01))
        
        x0_012 = torch.cat([x0_0, x0_1, x0_2], 1)
        if self.is_cbam:
            x0_012 = self.CBAM_2(x0_012)
        x0_3 = self.conv0_3(self.up(x1_2, x0_012))
        
        x4_0 = self.conv4_0(self.pool(x3_0))
        if self.is_aspp:
            x4_0 = self.aspp(x4_0)
        x3_1 = self.conv3_1(self.up(x4_0, x3_0))
        
        x2_01 = torch.cat([x2_0, x2_1], 1)
        if self.is_cbam:
            x2_01 = self.CBAM_3(x2_01)
        x2_2 = self.conv2_2(self.up(x3_1, x2_01))
        
        x1_012 = torch.cat([x1_0, x1_1, x1_2], 1)
        if self.is_cbam:
            x1_012 = self.CBAM_4(x1_012)
        x1_3 = self.conv1_3(self.up(x2_2, x1_012))
        
        x0_0123 = torch.cat([x0_0, x0_1, x0_2, x0_3], 1)
        if self.is_cbam:
            x0_0123 = self.CBAM_5(x0_0123)
        x0_4 = self.conv0_4(self.up(x1_3,x0_0123 ))

        output = self.final(x0_4)
        return {"out": output}
    