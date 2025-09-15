import torch
import torch.nn as nn
import torch.nn.functional as F


class DropBlock1D(nn.Module):
    """DropBlock1D的PyTorch实现: https://arxiv.org/pdf/1810.12890.pdf"""

    def __init__(self,
                 block_size,
                 keep_prob,
                 sync_channels=False,
                 data_format="channels_last"):
        """初始化层
        :param block_size: 每个掩码块的大小
        :param keep_prob: 保留原特征的概率
        :param sync_channels: 是否对所有通道使用相同的dropout
        :param data_format: 'channels_first'或'channels_last'(默认)
        """
        super(DropBlock1D, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.sync_channels = sync_channels
        self.data_format = data_format
        self.kernel_size = block_size
        self.stride = 1
        self.padding = block_size // 2

    def _get_gamma(self, feature_dim):
        """获取要丢弃的激活单元数量"""
        return ((1.0 - self.keep_prob) / self.block_size) * \
               (feature_dim / (feature_dim - self.block_size + 1.0))

    def _compute_valid_seed_region(self, seq_length, device):
        """计算可以应用dropblock的有效种子区域"""
        seq_range = torch.arange(seq_length, dtype=torch.float, device=device)
        half_block_size = self.block_size // 2
        
        # 创建有效区域的掩码
        mask_left = seq_range >= half_block_size
        mask_right = seq_range < (seq_length - half_block_size)
        valid_seed_region = (mask_left & mask_right).float()
        
        return valid_seed_region.view(1, seq_length, 1)

    def _compute_drop_mask(self, shape, device):
        """计算输入张量的丢弃掩码"""
        seq_length = shape[1]
        gamma = self._get_gamma(seq_length)
        
        # 使用均匀分布生成二进制掩码
        mask = (torch.rand(shape, device=device) < gamma).float()
        mask = mask * self._compute_valid_seed_region(seq_length, device)
        
        # 应用最大池化来扩展mask
        # 注意: 在PyTorch中，池化层输入要求为[N, C, L]格式
        mask = mask.permute(0, 2, 1)  # [batch, 1, seq_len]
        mask = F.max_pool1d(
            mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )
        mask = mask.permute(0, 2, 1)  # [batch, seq_len, 1]
        
        return 1.0 - mask

    def forward(self, x):
        if not self.training or self.keep_prob == 1:
            return x
        
        # 处理不同的数据格式
        if self.data_format == 'channels_first':
            # [batch, channels, seq_len] -> [batch, seq_len, channels]
            x = x.permute(0, 2, 1)
        
        shape = x.shape
        device = x.device
        
        # 根据同步通道选项计算掩码
        if self.sync_channels:
            mask = self._compute_drop_mask([shape[0], shape[1], 1], device)
        else:
            mask = self._compute_drop_mask(shape, device)
        
        # 归一化输出
        scale = torch.prod(torch.tensor(shape, dtype=torch.float, device=device)) / (torch.sum(mask) + 1e-8)
        x = x * mask * scale
        
        # 恢复原始数据格式
        if self.data_format == 'channels_first':
            # [batch, seq_len, channels] -> [batch, channels, seq_len]
            x = x.permute(0, 2, 1)
        
        return x


class DropBlock2D(nn.Module):
    """DropBlock2D的PyTorch实现: https://arxiv.org/pdf/1810.12890.pdf"""

    def __init__(self,
                 block_size,
                 keep_prob,
                 sync_channels=False,
                 data_format="channels_last"):
        """初始化层
        :param block_size: 每个掩码块的大小
        :param keep_prob: 保留原特征的概率
        :param sync_channels: 是否对所有通道使用相同的dropout
        :param data_format: 'channels_first'或'channels_last'(默认)
        """
        super(DropBlock2D, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.sync_channels = sync_channels
        self.data_format = data_format
        self.kernel_size = (block_size, block_size)
        self.stride = (1, 1)
        self.padding = (block_size // 2, block_size // 2)

    def _get_gamma(self, height, width):
        """获取要丢弃的激活单元数量"""
        return ((1.0 - self.keep_prob) / (self.block_size ** 2)) * \
               (height * width / ((height - self.block_size + 1.0) * (width - self.block_size + 1.0)))

    def _compute_valid_seed_region(self, height, width, device):
        """计算可以应用dropblock的有效种子区域"""
        # 创建高度和宽度范围
        h_range = torch.arange(height, dtype=torch.float, device=device)
        w_range = torch.arange(width, dtype=torch.float, device=device)
        
        # 创建坐标网格
        h_grid, w_grid = torch.meshgrid(h_range, w_range, indexing='ij')
        
        half_block_size = self.block_size // 2
        
        # 创建有效区域的掩码
        mask_top = h_grid >= half_block_size
        mask_left = w_grid >= half_block_size
        mask_bottom = h_grid < (height - half_block_size)
        mask_right = w_grid < (width - half_block_size)
        
        valid_seed_region = ((mask_top & mask_left) & (mask_bottom & mask_right)).float()
        
        return valid_seed_region.unsqueeze(0).unsqueeze(-1)

    def _compute_drop_mask(self, shape, device):
        """计算输入张量的丢弃掩码"""
        height, width = shape[1], shape[2]
        gamma = self._get_gamma(height, width)
        
        # 使用均匀分布生成二进制掩码
        mask = (torch.rand(shape, device=device) < gamma).float()
        mask = mask * self._compute_valid_seed_region(height, width, device)
        
        # 应用最大池化来扩展mask
        # 注意: 在PyTorch中，池化层输入要求为[N, C, H, W]格式
        mask = mask.permute(0, 3, 1, 2)  # [batch, channels, height, width]
        mask = F.max_pool2d(
            mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )
        mask = mask.permute(0, 2, 3, 1)  # [batch, height, width, channels]
        
        return 1.0 - mask

    def forward(self, x):
        if not self.training or self.keep_prob == 1:
            return x
        
        # 处理不同的数据格式
        if self.data_format == 'channels_first':
            # [batch, channels, height, width] -> [batch, height, width, channels]
            x = x.permute(0, 2, 3, 1)
        
        shape = x.shape
        device = x.device
        
        # 根据同步通道选项计算掩码
        if self.sync_channels:
            mask = self._compute_drop_mask([shape[0], shape[1], shape[2], 1], device)
        else:
            mask = self._compute_drop_mask(shape, device)
        
        # 归一化输出
        scale = torch.prod(torch.tensor(shape, dtype=torch.float, device=device)) / (torch.sum(mask) + 1e-8)
        x = x * mask * scale
        
        # 恢复原始数据格式
        if self.data_format == 'channels_first':
            # [batch, height, width, channels] -> [batch, channels, height, width]
            x = x.permute(0, 3, 1, 2)
        
        return x