import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from abc import abstractmethod
from datetime import timedelta
from collections import OrderedDict
from einops import rearrange, repeat


# =============================================================================
# 基礎工具與模組
# =============================================================================


# ----------------------------
# TimestepBlock 與 TimestepEmbedSequential
# ----------------------------
class TimestepBlock(nn.Module):
    """
    任意 forward() 接受 timestep embedding 作為第二個輸入的模組。
    """
    @abstractmethod
    def forward(self, x, emb):
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    會將 timestep embedding 傳遞給支持此功能的 sequential 模組。
    """
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


# ----------------------------
# PositionalEmbedding：計算 timestep 位置編碼
# ----------------------------
class PositionalEmbedding(nn.Module):
    def __init__(self, dim, scale=1):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale


    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# ----------------------------
# Downsample 與 Upsample
# ----------------------------
class Downsample(nn.Module):
    def __init__(self, in_channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = in_channels
        out_channels = out_channels or in_channels
        if use_conv:
            self.downsample = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        else:
            assert in_channels == out_channels
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)


    def forward(self, x, time_embed=None):
        assert x.shape[1] == self.channels
        return self.downsample(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = in_channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)


    def forward(self, x, time_embed=None):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


# =============================================================================
# 注意力相關模組
# =============================================================================


# ----------------------------
# Spatial Attention
# ----------------------------
class SpatialAttention(nn.Module):
    """
    聚焦輸入空間關係的 Spatial Attention 模組。
    使用兩層 1x1 卷積和 sigmoid 激活計算注意力權重，並與殘差結合後經 GroupNorm 正規化。
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(channels // 8, channels, kernel_size=1)
        self.norm = nn.GroupNorm(32, channels)
       
    def forward(self, x):
        shortcut = x
        weights = self.conv1(x)
        weights = F.relu(weights)
        weights = self.conv2(weights)
        weights = torch.sigmoid(weights)
        out = x * weights
        return self.norm(out + shortcut)


# ----------------------------
# 旋轉感知注意力 (Rotation-Aware Attention) - 修正版
# ----------------------------
class RotationAwareAttention(nn.Module):
    """
    旋轉感知注意力機制：專門用於處理不同角度的輸入。
    使用離散化的旋轉（0°, 90°, 180°, 270°）來避免梯度計算問題。
    """
    def __init__(self, in_channels, num_rotations=8, heads=8):
        super().__init__()
        self.in_channels = in_channels
        self.num_rotations = 4  # 使用固定的4個旋轉角度: 0, 90, 180, 270
        self.norm = nn.GroupNorm(32, in_channels)
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, 1, bias=False)
        self.heads = heads
        self.scale = (in_channels // heads) ** -0.5
        self.proj = nn.Conv2d(in_channels, in_channels, 1)
       
    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        x = self.norm(x)
       
        # 使用4個固定旋轉（0, 90, 180, 270度）而不是任意角度
        responses = []
       
        # 0度（無旋轉）
        responses.append(self._process_attention(x))
       
        # 90度
        x_90 = torch.rot90(x, k=1, dims=[2, 3])
        out_90 = self._process_attention(x_90)
        responses.append(torch.rot90(out_90, k=3, dims=[2, 3]))  # 旋轉回原始位置
       
        # 180度
        x_180 = torch.rot90(x, k=2, dims=[2, 3])
        out_180 = self._process_attention(x_180)
        responses.append(torch.rot90(out_180, k=2, dims=[2, 3]))  # 旋轉回原始位置
       
        # 270度
        x_270 = torch.rot90(x, k=3, dims=[2, 3])
        out_270 = self._process_attention(x_270)
        responses.append(torch.rot90(out_270, k=1, dims=[2, 3]))  # 旋轉回原始位置
       
        # 在所有旋轉中進行最大池化
        stacked_responses = torch.stack(responses, dim=0)
        max_response, _ = torch.max(stacked_responses, dim=0)
        out = self.proj(max_response)
       
        return out + shortcut


    def _process_attention(self, x):
        # 計算 QKV 注意力
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
       
        q = rearrange(q, 'b (head d) h w -> b head (h w) d', head=self.heads)
        k = rearrange(k, 'b (head d) h w -> b head (h w) d', head=self.heads)
        v = rearrange(v, 'b (head d) h w -> b head (h w) d', head=self.heads)
       
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
       
        out = rearrange(out, 'b head (h w) d -> b (head d) h w', h=x.shape[2], w=x.shape[3])
        return out


# ----------------------------
# 角度感知自注意力 (Angle-Aware Self-Attention)
# ----------------------------
class AngleAwareSelfAttention(nn.Module):
    """
    角度感知自注意力：使用相對位置編碼來增強對角度變化的敏感性。
    此模組通過計算特徵點之間的相對方向和距離，增強模型對旋轉的不變性。
    """
    def __init__(self, in_channels, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(in_channels)
       
        self.to_qkv = nn.Linear(in_channels, inner_dim * 3, bias=False)
       
        # 相對位置編碼
        self.rel_pos_encoding = nn.Parameter(torch.randn(2 * 7 - 1, 2 * 7 - 1, heads))  # 7x7 感受野
       
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, in_channels),
            nn.Dropout(dropout)
        )


    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
       
        # 將特徵圖轉為序列
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
       
        # 計算 QKV
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
       
        # 注意力計算
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
       
        # 添加相對位置編碼
        rel_pos = self._get_rel_pos(H, W)
        dots = dots + rel_pos
       
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
       
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
       
        # 重塑回原始格式
        out = rearrange(out, 'b (h w) c -> b c h w', h=H)
       
        return out + shortcut
   
    def _get_rel_pos(self, H, W):
        """獲取相對位置編碼"""
        rel_pos = self.rel_pos_encoding.unsqueeze(0)  # [1, 2*H-1, 2*W-1, heads]
        center_h, center_w = self.rel_pos_encoding.shape[0] // 2, self.rel_pos_encoding.shape[1] // 2
        rel_pos = rel_pos[:, center_h-H+1:center_h+H, center_w-W+1:center_w+W]
        rel_pos = rearrange(rel_pos, '1 h w c -> c (h w)')
        return rel_pos


# ----------------------------
# 角度自適應注意力模塊 (Angle-Adaptive Attention Module)
# ----------------------------
class AngleAdaptiveAttention(nn.Module):
    """
    角度自適應注意力：專門設計用於檢測和適應不同角度的目標特徵。
    通過多個方向性過濾器和角度預測機制，提高模型對旋轉不變性的支持。
    """
    def __init__(self, in_channels, num_angles=16, kernel_size=5):
        super().__init__()
        self.in_channels = in_channels
        self.num_angles = num_angles
        self.kernel_size = kernel_size
        padding = kernel_size // 2
       
        # 創建多個方向性過濾器
        self.directional_filters = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, bias=False)
            for _ in range(num_angles)
        ])
       
        # 初始化方向性過濾器
        self._initialize_directional_filters()
       
        # 角度預測層
        self.angle_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_angles, 1),
            nn.Softmax(dim=1)
        )
       
        # 輸出投影
        self.proj = nn.Conv2d(in_channels, in_channels, 1)
       
    def _initialize_directional_filters(self):
        """初始化方向性過濾器權重"""
        for i, filt in enumerate(self.directional_filters):
            angle = i * (2 * math.pi / self.num_angles)
            kernel = torch.zeros(self.kernel_size, self.kernel_size)
            center = self.kernel_size // 2
           
            for r in range(1, center + 1):
                x = int(round(center + r * math.cos(angle)))
                y = int(round(center + r * math.sin(angle)))
               
                if 0 <= x < self.kernel_size and 0 <= y < self.kernel_size:
                    kernel[y, x] = 1.0
               
                x = int(round(center - r * math.cos(angle)))
                y = int(round(center - r * math.sin(angle)))
               
                if 0 <= x < self.kernel_size and 0 <= y < self.kernel_size:
                    kernel[y, x] = 1.0
           
            if kernel.sum() > 0:
                kernel = kernel / kernel.sum()
           
            filt.weight.data.fill_(0)
            for c_out in range(self.in_channels):
                for c_in in range(self.in_channels):
                    if c_out == c_in:
                        filt.weight.data[c_out, c_in] = kernel
       
    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
       
        directional_outputs = []
        for filt in self.directional_filters:
            directional_outputs.append(filt(x))
       
        angle_weights = self.angle_predictor(x)
       
        combined = torch.zeros_like(x)
        for i, dir_out in enumerate(directional_outputs):
            weight = angle_weights[:, i:i+1]
            combined += dir_out * weight
       
        out = self.proj(combined)
        return out + shortcut


# ----------------------------
# Halo Attention
# ----------------------------
class HaloAttention(nn.Module):
    """
    Halo Attention 機制：結合區域內局部注意力與全局上下文資訊。
    此版本對每個非重疊區塊採用擴展（halo）視野進行注意力計算，
    並僅更新中心區域的特徵，有助於捕捉局部與鄰近上下文間的關聯。
    """
    def __init__(self, dim, block_size=8, halo_size=3, num_heads=16, qkv_bias=False):
        super().__init__()
        self.block_size = block_size
        self.halo_size = halo_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
       
        self.patch_size = self.block_size + 2 * self.halo_size
        center = []
        for r in range(self.halo_size, self.halo_size + self.block_size):
            start = r * self.patch_size + self.halo_size
            center.extend(range(start, start + self.block_size))
        center_indices = torch.tensor(center)
        self.register_buffer('center_indices', center_indices)


    def forward(self, x):
        B, C, H, W = x.shape
        pad_h = (self.block_size - H % self.block_size) % self.block_size
        pad_w = (self.block_size - W % self.block_size) % self.block_size
        x = F.pad(x, (self.halo_size, pad_w + self.halo_size, self.halo_size, pad_h + self.halo_size))
        H_blocks = (H + pad_h) // self.block_size
        W_blocks = (W + pad_w) // self.block_size
       
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.block_size)
        patches = patches.view(B, C, self.patch_size * self.patch_size, H_blocks, W_blocks)
        patches = patches.permute(0, 3, 4, 2, 1).contiguous()
        patches = patches.view(-1, self.patch_size * self.patch_size, C)
       
        qkv = self.qkv(patches)
        dim = C
        qkv = qkv.view(-1, self.patch_size * self.patch_size, 3, self.num_heads, dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
       
        q = torch.index_select(q, dim=2, index=self.center_indices)
       
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(-1, self.block_size * self.block_size, dim)
        out = self.proj(out)
       
        out = out.view(B, H_blocks, W_blocks, self.block_size, self.block_size, dim)
        out = out.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, dim, H + pad_h, W + pad_w)
        out = out[:, :, :H, :W]
        return out


# ----------------------------
# QKV Attention（用於 EnhancedAttentionBlock）
# ----------------------------
class QKVAttention(nn.Module):
    """
    基於 QKV 計算的多頭注意力模組。
    將輸入按照頭數分割，並計算注意力後合併輸出。
    """
    def __init__(self, n_heads, embed_dim):
        super().__init__()
        self.n_heads = n_heads
        self.scale = (embed_dim // n_heads) ** -0.5


    def forward(self, qkv):
        B, N, three_dim = qkv.shape
        embed_dim = three_dim // 3
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, self.n_heads, embed_dim // self.n_heads).transpose(1, 2)
        k = k.view(B, N, self.n_heads, embed_dim // self.n_heads).transpose(1, 2)
        v = v.view(B, N, self.n_heads, embed_dim // self.n_heads).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, embed_dim)
        return out


# ----------------------------
# 綜合角度不變注意力模組 (Comprehensive Angle-Invariant Attention)
# ----------------------------
class ComprehensiveAngleInvariantAttention(nn.Module):
    """
    綜合角度不變注意力模組：結合多種注意力機制，
    採用門控機制動態調整各個分支的權重，以最大化特徵提取效率。
    """
    def __init__(self, in_channels, heads=8, num_rotations=8, num_angles=16):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels)
       
        self.spatial_attn = SpatialAttention(in_channels)
        self.rotation_attn = RotationAwareAttention(in_channels, num_rotations, heads)
        self.angle_attn = AngleAwareSelfAttention(in_channels, heads)
        self.angle_adaptive_attn = AngleAdaptiveAttention(in_channels, num_angles)
       
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, 4, 1),
            nn.Softmax(dim=1)
        )
       
        self.proj = nn.Conv2d(in_channels, in_channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
       
    def forward(self, x, time=None):
        shortcut = x
        normed = self.norm(x)
       
        spatial_out = self.spatial_attn(normed)
        rotation_out = self.rotation_attn(normed)
        angle_out = self.angle_attn(normed)
        adaptive_out = self.angle_adaptive_attn(normed)
       
        gates = self.gate(x)
       
        g1, g2, g3, g4 = gates.chunk(4, dim=1)
       
        g1 = g1.repeat(1, self.spatial_attn.conv1.weight.shape[0], 1, 1)
        g2 = g2.repeat(1, self.rotation_attn.in_channels, 1, 1)
        g3 = g3.repeat(1, self.angle_attn.to_qkv.weight.shape[1], 1, 1)
        g4 = g4.repeat(1, self.angle_adaptive_attn.in_channels, 1, 1)
       
        combined = (
            spatial_out * g1 +
            rotation_out * g2 +
            angle_out * g3 +
            adaptive_out * g4
        )
       
        out = self.proj(combined)
        return out + shortcut


# ----------------------------
# Enhanced Attention Block
# ----------------------------
class EnhancedAttentionBlock(nn.Module):
    """
    角度增強型注意力模塊：結合多種注意力機制，特別關注角度不變性。
    增加了旋轉感知和角度自適應注意力，以處理不同角度的對象。
    """
    def __init__(self, in_channels, n_heads=16, block_size=8, halo_size=3, num_rotations=8):
        super().__init__()
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(32, in_channels)
        self.spatial_attn = SpatialAttention(in_channels)
        self.halo_attn = HaloAttention(in_channels,
                                       block_size=block_size,
                                       halo_size=halo_size,
                                       num_heads=n_heads)
        self.to_qkv = nn.Linear(in_channels, in_channels * 3)
        self.qkv_attn = QKVAttention(n_heads, in_channels)
       
        self.rotation_attn = RotationAwareAttention(in_channels, num_rotations=num_rotations, heads=n_heads)
       
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)
        nn.init.zeros_(self.proj_out.weight)
        if self.proj_out.bias is not None:
            nn.init.zeros_(self.proj_out.bias)
        self.attn_weight = nn.Parameter(torch.ones(4))


    def forward(self, x, time=None):
        shortcut = x
        spatial_out = self.spatial_attn(x)
        halo_out = self.halo_attn(x)
        rotation_out = self.rotation_attn(x)
       
        b, c, h, w = x.shape
        x_flat = x.reshape(b, c, -1)
        normed = self.norm(x_flat)
        normed = normed.permute(0, 2, 1)
        qkv = self.to_qkv(normed)
        qkv_out = self.qkv_attn(qkv)
        qkv_out = qkv_out.permute(0, 2, 1).reshape(b, c, h, w)
       
        weights = F.softmax(self.attn_weight, dim=0)
        combined = (
            spatial_out * weights[0] +
            halo_out * weights[1] +
            qkv_out * weights[2] +
            rotation_out * weights[3]
        )
       
        out = self.proj_out(combined)
        return out + shortcut


# ----------------------------
# ResBlock
# ----------------------------
class ResBlock(nn.Module):  # 假設 TimestepBlock 為某個基礎類別，此處保持不變
    def __init__(self, in_channels, time_embed_dim, dropout, out_channels=None, use_conv=False, up=False, down=False):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.updown = up or down
        if up:
            self.h_upd = Upsample(in_channels, False)
            self.x_upd = Upsample(in_channels, False)
        elif down:
            self.h_upd = Downsample(in_channels, False)
            self.x_upd = Downsample(in_channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        self.embed_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels)
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)


    def forward(self, x, time_embed):
        if not isinstance(self.h_upd, nn.Identity):
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.embed_layers(time_embed).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


# =============================================================================
# TransUNet + UNet++ 架構相關模組
# =============================================================================


# ----------------------------
# 基本卷積塊：ConvBlock
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()


    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        return x


# ----------------------------
# Encoder：多層卷積與池化
# ----------------------------
class Encoder(nn.Module):
    def __init__(self, in_channels, base_channels, levels=4, dropout=0.0, channel_mults=None):
        super().__init__()
        self.levels = levels
        self.blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
       
        if channel_mults is None:
            channel_mults = [2**i for i in range(levels)]
        current_channels = in_channels
        for mult in channel_mults:
            out_channels = base_channels * mult
            block = ConvBlock(current_channels, out_channels, dropout)
            self.blocks.append(block)
            current_channels = out_channels
        self.out_channels = current_channels
       
    def forward(self, x):
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features, x


# ----------------------------
# ViT Transformer Encoder Layer
# ----------------------------
class ViTEncoderLayer(nn.Module):
    def __init__(self, emb_dim, num_heads, mlp_ratio=4.0, dropout=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, int(emb_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(emb_dim * mlp_ratio), emb_dim),
            nn.Dropout(dropout),
        )


    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ----------------------------
# ViTTransformer：堆疊多層 Encoder Layer
# ----------------------------
class ViTTransformer(nn.Module):
    def __init__(self, emb_dim, depth, num_heads, mlp_ratio=4.0, dropout=0):
        super().__init__()
        self.layers = nn.ModuleList(
            [ViTEncoderLayer(emb_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ----------------------------
# Transformer Bottleneck：處理 Encoder 最深層特徵
# ----------------------------
class TransformerBottleneck(nn.Module):
    def __init__(self, in_channels, emb_dim, num_heads, depth, patch_size=1):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=1)
        self.transformer = ViTTransformer(emb_dim, depth, num_heads)
        self.out_conv = nn.Conv2d(emb_dim, in_channels, kernel_size=1)


    def forward(self, x):
        B, C, H, W = x.shape
        x_proj = self.proj(x)
        x_flat = x_proj.flatten(2).transpose(1, 2)
        x_trans = self.transformer(x_flat)
        x_out = x_trans.transpose(1, 2).view(B, -1, H, W)
        x_out = self.out_conv(x_out)
        return x_out


# ----------------------------
# UNet++ Decoder（簡化版本）
# ----------------------------
class UNetPlusPlusDecoder(nn.Module):
    def __init__(self, encoder_channels, base_channels, dropout=0.1, out_channels=4):
        """
        encoder_channels：從 Encoder 各層獲得的通道數列表。
        out_channels：最終輸出通道數（通常與輸入相同）。
        """
        super().__init__()
        self.num_levels = len(encoder_channels)
        self.decoders = nn.ModuleDict()
       
        for i in range(self.num_levels):
            if i == self.num_levels - 1:
                in_ch = encoder_channels[-1] * 2
            else:
                in_ch = encoder_channels[i] + encoder_channels[i+1]
            out_ch = encoder_channels[i]
            self.decoders[f"layer_{i}"] = ConvBlock(in_ch, out_ch, dropout)
           
        self.final_conv = nn.Conv2d(encoder_channels[0], out_channels, kernel_size=1)


    def forward(self, features):
        bottleneck = features[-1]
        features = features[:-1]
        x = bottleneck
        for i in range(len(features)-1, -1, -1):
            x = F.interpolate(x, size=features[i].shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, features[i]], dim=1)
            x = self.decoders[f"layer_{i}"](x)
        return self.final_conv(x)


# ----------------------------
# UNetModel：結合 TransUNet 與 UNet++ 架構
# ----------------------------
class UNetModel(nn.Module):
    def __init__(self, img_size, base_channels, dropout=0.1, in_channels=4,
                 n_head_channels=None, n_heads=None, channel_mults=None, attention_resolutions=None):
        """
        img_size：影像尺寸。
        base_channels：初始卷積通道數。
        in_channels：輸入（及輸出）通道數，預設為 4。
        n_head_channels：每個 head 的通道數（保留參數，暫未直接使用）。
        n_heads：多頭注意力 head 數量。
        channel_mults：各層通道倍數，預設為 [1,2,4,8]。
        attention_resolutions：於哪些解析度下加入注意力模組（例如 [16, 8]）。
        """
        super().__init__()
        self.img_size = img_size
        self.n_head_channels = n_head_channels
       
        if channel_mults is None:
            channel_mults = [2**i for i in range(4)]
        levels = len(channel_mults)
        self.encoder_channels = [base_channels * m for m in channel_mults]
       
        self.encoder = Encoder(in_channels, base_channels, levels=levels, dropout=dropout, channel_mults=channel_mults)
       
        self.attn_blocks = nn.ModuleList()
        for i, ch in enumerate(self.encoder_channels):
            res = img_size // (2**(i+1))
            if attention_resolutions is not None and res in attention_resolutions:
                self.attn_blocks.append(EnhancedAttentionBlock(ch, n_heads=n_heads if n_heads is not None else 8))
            else:
                self.attn_blocks.append(nn.Identity())
       
        bottleneck_in = self.encoder.out_channels
        self.bottleneck_conv = ConvBlock(bottleneck_in, bottleneck_in, dropout)
        self.transformer_bottleneck = TransformerBottleneck(
            bottleneck_in,
            emb_dim=bottleneck_in,
            num_heads=n_heads if n_heads is not None else 16,
            depth=3,
            patch_size=1
        )
       
        self.decoder = UNetPlusPlusDecoder(
            encoder_channels=self.encoder_channels,
            base_channels=base_channels,
            dropout=dropout,
            out_channels=in_channels
        )


    def forward(self, x, t=None):
        features, bottleneck_input = self.encoder(x)
        for i in range(len(features)):
            features[i] = self.attn_blocks[i](features[i])
        bneck = self.bottleneck_conv(bottleneck_input)
        bneck = self.transformer_bottleneck(bneck)
        features.append(bneck)
        out = self.decoder(features)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return out


# =============================================================================
# 其他工具函數
# =============================================================================




# =============================================================================
# 其他工具函數
# =============================================================================


class GroupNorm32(nn.GroupNorm):
    """
    使用 float32 計算的 GroupNorm，並在輸出時恢復原始資料型態。
    注意：若 channels 較少，建議檢查 groups 設定是否合理。
    """
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def zero_module(module):
    """
    將 module 中的所有參數歸零，用於初始穩定化。
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def update_ema_params(target, source, decay_rate=0.9999):
    """
    使用指數移動平均 (EMA) 更新 target 模型的參數。
    """
    targParams = dict(target.named_parameters())
    srcParams = dict(source.named_parameters())
    for k in targParams:
        targParams[k].data.mul_(decay_rate).add_(srcParams[k].data, alpha=1 - decay_rate)
