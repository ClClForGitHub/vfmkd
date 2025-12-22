import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BioRetinaLayer(nn.Module):
    """
    单层注入模块：负责将 DoG 图注入到某一层特征 (S3/S4/S5)
    [修复版] 采用 Zero-Initialization 策略，防止破坏预训练特征
    """
    def __init__(self, stride, feat_channels):
        super().__init__()
        self.stride = stride
        
        # 门控投影器
        # 修改为更轻量的结构，去掉 BN，防止初始化分布漂移
        # 结构: Conv(1->C) -> ReLU -> Conv(C->C) -> Sigmoid
        # 这样能更好学习非线性关系
        self.projector = nn.Sequential(
            nn.Conv2d(1, feat_channels // 4, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels // 4, feat_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        
        # [核心修复] 添加一个可学习的缩放系数 gamma，初始化为 0
        # 这保证了初始状态下: out = feat + 0 * enhancement = feat
        # 实现了 "Identity Mapping" (恒等映射)，完全保留预训练能力
        self.gamma = nn.Parameter(torch.zeros(1)) 

        # 权重初始化
        for m in self.projector.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, dog_map, feat):
        """
        dog_map: [B, 1, H, W]
        feat:    [B, C, H/s, W/s]
        """
        # 1. MaxPool 下采样
        dog_down = F.max_pool2d(dog_map, kernel_size=self.stride, stride=self.stride)
        
        # 2. 计算 Gate
        gate = self.projector(dog_down)
        
        # 3. 零初始化残差注入
        # 初始时 gamma=0，返回纯 feat，AP 不会掉
        # 训练中 gamma 变大，DoG 边缘信息逐渐融入
        return feat + (self.gamma * feat * gate)


class BioRetinaModule(nn.Module):
    """
    仿生视网膜主模块：管理多层注入
    """
    def __init__(self, 
                 strides=[8, 16, 32], 
                 in_channels=[512, 1024, 2048], 
                 sigma1=1.0, 
                 sigma2=2.0,
                 dog_threshold=0.0): 
        super().__init__()
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.dog_threshold = dog_threshold
        
        self.layers = nn.ModuleList([
            BioRetinaLayer(stride=s, feat_channels=c) 
            for s, c in zip(strides, in_channels)
        ])

    def get_gaussian_kernel(self, kernel_size=3, sigma=1.0, channels=1):
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma**2.

        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean)**2., dim=-1) / \
                              (2 * variance)
                          )
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        return gaussian_kernel

    def get_dog(self, img):
        # img: [B, 3, H, W] normalized
        # 注意：RT-DETR 输入通常已经归一化 (mean/std)
        # DoG 是线性差分，归一化不影响相对边缘，但会影响数值幅度
        # 我们直接在归一化图上算即可，Projector 会适应这个幅度
        
        if img.shape[1] == 3:
            gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        else:
            gray = img
            
        k_size1 = int(2 * 3 * self.sigma1 + 1) | 1
        k_size2 = int(2 * 3 * self.sigma2 + 1) | 1
        
        kernel1 = self.get_gaussian_kernel(k_size1, self.sigma1).to(img.device)
        kernel2 = self.get_gaussian_kernel(k_size2, self.sigma2).to(img.device)
        
        pad1 = k_size1 // 2
        pad2 = k_size2 // 2
        
        # 使用 groups=1 进行单通道卷积
        g1 = F.conv2d(gray, kernel1, padding=pad1, groups=1)
        g2 = F.conv2d(gray, kernel2, padding=pad2, groups=1)
        
        dog = g1 - g2
        
        if self.dog_threshold > 0:
            dog = torch.abs(dog)
            dog = F.relu(dog - self.dog_threshold)
            
        return dog

    def forward(self, images, features):
        with torch.no_grad():
            dog_map = self.get_dog(images)
        
        new_feats = []
        for i, feat in enumerate(features):
            new_feat = self.layers[i](dog_map, feat)
            new_feats.append(new_feat)
            
        return new_feats
