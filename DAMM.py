import torch
import torch.nn as nn

class DAMM(nn.Module):
    def __init__(self, dim, use_spa=True, use_ca=True, use_pa=True, use_dir=True):
        super().__init__()
        self.norm2 = nn.BatchNorm2d(dim)
        self.use_spa = use_spa  # 简单像素注意力分支开关
        self.use_ca = use_ca    # 通道注意力分支开关
        self.use_pa = use_pa    # 像素注意力分支开关
        self.use_dir = use_dir  # 方向感知分支开关
        
        # 简单像素注意力
        if self.use_spa:
            self.Wv = nn.Sequential(
                nn.Conv2d(dim, dim, 1),
                nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim, padding_mode='reflect')
            )
            self.Wg = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, dim, 1),
                nn.Sigmoid()
            )
        
        # 通道注意力
        if self.use_ca:
            self.ca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, dim, 1, padding=0, bias=True),
                nn.GELU(),
                nn.Conv2d(dim, dim, 1, padding=0, bias=True),
                nn.Sigmoid()
            )
        
        # 像素注意力
        if self.use_pa:
            self.pa = nn.Sequential(
                nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),
                nn.GELU(),
                nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
            )
        
        # 方向感知分支
        if self.use_dir:
            # 水平卷积，用于捕获水平特征
            self.conv_h = nn.Conv2d(dim, dim, kernel_size=(1, 7), padding="same", bias=False)
            # 垂直卷积，用于捕获垂直特征
            self.conv_v = nn.Conv2d(dim, dim, kernel_size=(7, 1), padding="same", bias=False)
            # 融合卷积，用于合并水平和垂直特征
            self.conv_fusion = nn.Conv2d(dim*2, dim, kernel_size=1, padding="same", bias=False)
            self.bn_fusion = nn.BatchNorm2d(dim)
            self.activation = nn.GELU()
        
        # 根据启用的分支数量计算MLP输入通道数
        in_channels_mlp = 0
        if self.use_spa: in_channels_mlp += dim
        if self.use_ca: in_channels_mlp += dim
        if self.use_pa: in_channels_mlp += dim
        if self.use_dir: in_channels_mlp += dim
        
        # 确保至少有一个分支被启用
        if in_channels_mlp == 0:
            in_channels_mlp = dim  # 如果没有分支启用，直接使用输入
        
        # 修改MLP以适应不同数量的分支输入
        self.mlp2 = nn.Sequential(
            nn.Conv2d(in_channels_mlp, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1)
        )
    
    def forward(self, x):
        identity = x  # 保存输入以便残差连接
        x = self.norm2(x)  # 批归一化
        
        branches_outputs = []  # 存储各分支输出

        # 简单像素注意力分支
        if self.use_spa:
            spa = self.Wv(x) * self.Wg(x)
            branches_outputs.append(spa)
        
        # 通道注意力分支
        if self.use_ca:
            ca_out = self.ca(x) * x
            branches_outputs.append(ca_out)
        
        # 像素注意力分支
        if self.use_pa:
            pa_out = self.pa(x) * x
            branches_outputs.append(pa_out)
        
        # 方向感知分支
        if self.use_dir:
            h = self.conv_h(x)
            v = self.conv_v(x)
            dir_feat = torch.cat([h, v], dim=1)
            dir_feat = self.conv_fusion(dir_feat)
            dir_feat = self.bn_fusion(dir_feat)
            dir_feat = self.activation(dir_feat)
            branches_outputs.append(dir_feat)
        
        # 如果没有分支被启用，直接使用归一化后的输入
        if len(branches_outputs) == 0:
            combined = x
        else:
            # 连接所有分支的输出
            combined = torch.cat(branches_outputs, dim=1)
        
        # 通过MLP处理
        out = self.mlp2(combined)
        
        # 残差连接
        out = identity + out
        
        return out