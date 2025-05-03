import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from models_utils import weights_init, print_network
import cv2
import numpy as np


def heatmap(img):
    avg_img = np.mean(img, axis=1)  # 对通道维度求平均
    # 移除批量维度，得到 (768, 1024) 的图像
    avg_img = avg_img[0]
    # 归一化到 0-255 之间，并转换为 uint8 类型
    avg_img = ((avg_img - avg_img.min()) / (avg_img.max() - avg_img.min()) * 255).astype('uint8')
    # 使用 applyColorMap 生成热图
    heatmap = cv2.applyColorMap(avg_img, cv2.COLORMAP_JET)
    return heatmap




###### Layer 
def conv1x1(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels,out_channels,kernel_size = 1,
                    stride =stride, padding=0,bias=False)

def conv3x3(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels,out_channels,kernel_size = 3,
        stride =stride, padding=1,bias=False)

class Bottleneck(nn.Module):
    def __init__(self,in_channels,out_channels,):
        super(Bottleneck,self).__init__()
        m  = OrderedDict()
        m['conv1'] = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        m['relu1'] = nn.ReLU(True)
        m['conv2'] = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, bias=False,dilation=2)
        m['relu2'] = nn.ReLU(True)
        m['conv3'] = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.group1 = nn.Sequential(m)
        self.relu= nn.Sequential(nn.ReLU(True))

    def forward(self, x):
        out = self.group1(x) 
        return out

class irnn_layer(nn.Module):
    def __init__(self,in_channels):
        super(irnn_layer,self).__init__()
        self.left_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        self.right_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        self.up_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        self.down_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        
    def forward(self,x):
        _,_,H,W = x.shape
        top_left = x.clone()
        top_right = x.clone()
        top_up = x.clone()
        top_down = x.clone()
        top_left[:,:,:,1:] = F.relu(self.left_weight(x)[:,:,:,:W-1]+x[:,:,:,1:],inplace=False)
        top_right[:,:,:,:-1] = F.relu(self.right_weight(x)[:,:,:,1:]+x[:,:,:,:W-1],inplace=False)
        top_up[:,:,1:,:] = F.relu(self.up_weight(x)[:,:,:H-1,:]+x[:,:,1:,:],inplace=False)
        top_down[:,:,:-1,:] = F.relu(self.down_weight(x)[:,:,1:,:]+x[:,:,:H-1,:],inplace=False)
        return (top_up,top_right,top_down,top_left)


class Attention(nn.Module):
    def __init__(self,in_channels):
        super(Attention,self).__init__()
        self.out_channels = int(in_channels/2)
        self.conv1 = nn.Conv2d(in_channels,self.out_channels,kernel_size=3,padding=1,stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channels,self.out_channels,kernel_size=3,padding=2, stride=1, dilation=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(self.out_channels,4,kernel_size=1,padding=0,stride=1)
        self.sigmod = nn.Sigmoid()
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.sigmod(out)
        return out


class SAM(nn.Module):
    def __init__(self,in_channels,out_channels,attention=1):
        super(SAM,self).__init__()
        self.out_channels = out_channels
        self.irnn1 = irnn_layer(self.out_channels)
        self.irnn2 = irnn_layer(self.out_channels)
        self.conv_in = conv3x3(in_channels,self.out_channels)
        self.relu1 = nn.ReLU(True)
        
        self.conv1 = nn.Conv2d(self.out_channels,self.out_channels,kernel_size=1,stride=1,padding=0)
        # self.conv2 = nn.Conv2d(self.out_channels*4,self.out_channels,kernel_size=1,stride=1,padding=0)
        self.conv3 = nn.Conv2d(self.out_channels*4,self.out_channels,kernel_size=1,stride=1,padding=0)
        self.relu2 = nn.ReLU(True)
        self.attention = attention
        if self.attention:
            self.attention_layer = Attention(in_channels)
        self.conv_out = conv1x1(self.out_channels,1)
        self.sigmod = nn.Sigmoid()
    
    def forward(self,x):
        if self.attention:
            weight = self.attention_layer(x)
        out = self.conv1(x)
        top_up,top_right,top_down,top_left = self.irnn1(out)
        
        # direction attention
        if self.attention:
            top_up.mul(weight[:,0:1,:,:])
            top_right.mul(weight[:,1:2,:,:])
            top_down.mul(weight[:,2:3,:,:])
            top_left.mul(weight[:,3:4,:,:])
        out = torch.cat([top_up,top_right,top_down,top_left],dim=1)

        out = self.conv3(out)
        out = self.relu2(out)
        mask = self.sigmod(self.conv_out(out))
        return mask
###### Network
class SPANet(nn.Module):
    def __init__(self):
        super(SPANet, self).__init__()

        self.conv_in = nn.Sequential(
            conv3x3(3, 50),  # 特征提取后 通过padding 保证shape 不变
            nn.ReLU(True)
        )
        self.SAM1 = SAM(50, 50, 1)
        self.bn1 = nn.BatchNorm2d(50)
        self.bn2 = nn.BatchNorm2d(50)
        # self.bn3 = nn.BatchNorm2d(70)
        # self.bn4 = nn.BatchNorm2d(50)
        self.bn5 = nn.BatchNorm2d(50)
        self.res_block1 = Bottleneck(50, 50)
        # self.res_block2 = Bottleneck(100, 100)
        # self.res_block3 = Bottleneck(100, 100)
        self.res_block4 = Bottleneck(50, 50)
        self.res_block5 = Bottleneck(50, 50)
        self.res_block6 = Bottleneck(50, 50)
        self.res_block7 = Bottleneck(50, 50)
        self.res_block8 = Bottleneck(50, 50)
        self.res_block9 = Bottleneck(50, 50)

        self.conv_out = nn.Sequential(
            conv1x1(50, 50)
        )



    def forward(self, x):

        out = self.conv_in(x)  # CONV +RELU
        out = F.relu(self.res_block1(out) + out)  # RB

        Attention1, up_heatmap, right_heatmap,down_heatmap,left_heatmap = self.SAM1(out)  # SAB
        out = F.relu(self.res_block4(out) * Attention1 + out)
        out = F.relu(self.res_block5(out) * Attention1 + out)
        out = F.relu(self.res_block6(out) * Attention1 + out)

        out = self.bn1(out)
        Attention2,_,_,_,_  = self.SAM1(out)
        out = F.relu(self.res_block7(out) * Attention2 + out)
        out = F.relu(self.res_block8(out) * Attention2 + out)
        out = F.relu(self.res_block9(out) * Attention2 + out)

        out = self.bn2(out)

        out = self.conv_out(out)
        out = self.bn5(out)
        return Attention1, out

