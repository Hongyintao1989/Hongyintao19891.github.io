import torch
import torch.nn as nn

# 定义一个多尺度并行大核卷积模型的类，继承自nn.Module
class MFRPP(nn.Module):
    def __init__(self, dim):
        super().__init__()  # 调用父类的初始化方法

        # 定义一个2D批归一化层
        self.norm1 = nn.BatchNorm2d(dim)

        # 定义一系列卷积层
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)  # 1x1卷积
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, padding_mode='reflect')  # 5x5卷积，反射填充

        # 深度卷积，使用不同的卷积核大小和膨胀率
        # 大感受野卷积：7×7卷积核，膨胀率4，有效感受野达到27×27
        self.conv3_7 = nn.Conv2d(dim, dim, kernel_size=7, padding=12, groups=dim, dilation=4, padding_mode='reflect')
        # 中等感受野卷积：5×5卷积核，膨胀率2，有效感受野达到9×9
        self.conv3_5 = nn.Conv2d(dim, dim, kernel_size=5, padding=4, groups=dim, dilation=2, padding_mode='reflect')
        # 小感受野卷积：3×3卷积核，膨胀率1，有效感受野为3×3
        self.conv3_3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, dilation=1, padding_mode='reflect')

        # 多层感知机（MLP）模块，包含两个卷积层和一个GELU激活函数
        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),  # 卷积层，改变通道数
            nn.GELU(),  # GELU激活函数
            nn.Conv2d(dim * 4, dim, 1)  # 卷积层，恢复通道数
        )

    # 定义前向传播过程
    def forward(self, x):
        identity = x  # 保存输入用于残差连接

        x = self.norm1(x)  # 批归一化
        x = self.conv1(x)  # 1x1卷积
        x = self.conv2(x)  # 5x5卷积

        # 将三个不同卷积核的输出在通道维度上拼接
        x = torch.cat([self.conv3_7(x), self.conv3_5(x), self.conv3_3(x)], dim=1)

        x = self.mlp(x)  # 通过MLP模块
        x = identity + x  # 残差连接
        return x  # 返回输出

if __name__ == '__main__':
    # 实例化模型对象，指定输入通道数
    model = MFRPP(dim=32)
    input = torch.randn(8, 32, 64, 64)
    # 执行前向传播，获取输出
    output = model(input)


    print('input_size:', input.size())
    print('output_size:', output.size())
      # 打印输入和输出的张量形状
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params / 1e6:.2f}M')