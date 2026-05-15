import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- 正则化组件 --------------------
class DropPath(nn.Module):
    """Stochastic Depth (DropPath) 实现，用于随机丢弃整个残差分支"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # 生成与 x 相同 batch_size 的掩码，并扩展维度使其可广播
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 二值化
        output = x.div(keep_prob) * random_tensor
        return output

# -------------------- 基础组件（增加 Dropout）--------------------
class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, use_act=True, drop_rate=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True) if use_act else nn.Identity()
        # 对卷积输出施加 2D Dropout（在激活函数之后更常见，但这里在激活前也可，视习惯而定）
        # 选择在激活后使用 Dropout2d，能随机丢弃整个通道的特征图位置
        self.drop = nn.Dropout2d(drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class ResidualBlock(nn.Module):
    """简单残差块，支持 Dropout"""
    def __init__(self, ch, drop_rate=0.0):
        super().__init__()
        self.conv1 = ConvLayer(ch, ch, drop_rate=drop_rate)
        self.conv2 = ConvLayer(ch, ch, use_act=False, drop_rate=drop_rate)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.conv2(self.conv1(x)))

class ChannelAttention(nn.Module):
    def __init__(self, ch, reduction=16, drop_rate=0.0):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),          # 在 FC 中间加 Dropout
            nn.Linear(ch // reduction, ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# -------------------- Swin Transformer 模块（增加 Dropout + DropPath）--------------------
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# -------------------- Uformer 局部增强注意力 --------------------
class LeWinAttention(nn.Module):
    """Uformer 的局部增强窗口自注意力 (Locally-Enhanced Window Attention)"""
    def __init__(self, dim, window_size, num_heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # QKV 投影
        self.qkv = nn.Linear(dim, dim * 3)
        # 对 V 的深度卷积以注入局部上下文（Uformer 核心改动）
        self.v_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        # x: (B_, N, C), N = window_size * window_size
        B_, N, C = x.shape
        H = W = self.window_size

        # 1) 线性投影得到 Q、K、V
        qkv = self.qkv(x).reshape(B_, N, 3, C).permute(2, 0, 1, 3)  # (3, B_, N, C)
        q, k, v = qkv[0], qkv[1], qkv[2]                             # 每个: (B_, N, C)

        # 2) 对 V 进行局部增强：3x3 深度可分离卷积
        v = v.transpose(1, 2).reshape(B_, C, H, W)   # (B_, C, H, W)
        v = self.v_conv(v)                           # 深度卷积（groups=C）
        v = v.reshape(B_, C, N).transpose(1, 2)      # 恢复为 (B_, N, C)

        # 3) 多头分割
        q = q.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (B_, head, N, d)
        k = k.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 4) 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B_, head, N, N)
        if mask is not None:
            nW = mask.shape[0]
            B = B_ // nW
            mask = mask.unsqueeze(1).unsqueeze(0)           # (1, nW, 1, N, N)
            mask = mask.expand(B, -1, self.num_heads, -1, -1)  # (B, nW, head, N, N)
            mask = mask.reshape(B_, self.num_heads, N, N)
            attn = attn + mask

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # 5) 加权聚合 + 输出投影
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # (B_, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# -------------------- Uformer 局部增强 Transformer 块 --------------------
class UformerTransformerBlock(nn.Module):
    """基于 Uformer 的 Local-Enhanced Transformer 块，替代 SwinTransformerBlock"""
    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., drop_rate=0.0, attn_drop=0.0, drop_path_rate=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        # 核心：使用 LeWinAttention 替换原来的 WindowAttention
        self.attn = LeWinAttention(dim, window_size, num_heads,
                                   attn_drop=attn_drop, proj_drop=drop_rate)
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop_rate)
        )
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def _generate_mask(self, H, W, device):
        if self.shift_size == 0:
            return None
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)  # (nW, ws, ws, 1)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.window_size == 0 and W % self.window_size == 0, \
            f"特征图尺寸 ({H}, {W}) 必须能被 window_size ({self.window_size}) 整除"
        shortcut = x
        x = self.norm1(x.permute(0, 2, 3, 1))  # (B, H, W, C)

        # 移位窗口
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 窗口划分
        x_windows = window_partition(shifted_x, self.window_size)  # (nW*B, ws, ws, C)
        nW = x_windows.shape[0] // B
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # (B*nW, N, C)

        attn_mask = self._generate_mask(H, W, x.device)

        # 局部增强注意力
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # 恢复移位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # 第一个残差连接（保留与原模型一致的 0.5 缩放）
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = shortcut + self.drop_path1(x) * 0.5

        # MLP + 第二个残差连接
        shortcut = x
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = shortcut + self.drop_path2(self.mlp(self.norm2(x)).permute(0, 3, 1, 2)) * 0.5
        return x

# -------------------- 编码器双分支（传递 dropout 参数）--------------------
class SimpleBranch(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks=3, drop_rate=0.0):
        super().__init__()
        self.conv_in = ConvLayer(in_ch, out_ch, drop_rate=drop_rate)
        self.blocks = nn.Sequential(
            *[ResidualBlock(out_ch, drop_rate=drop_rate) for _ in range(num_blocks)]
        )

    def forward(self, x):
        return self.blocks(self.conv_in(x))

class ComplexBranch(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks=2, window_size=8, num_heads=4,
                 mlp_ratio=4., drop_rate=0.0, attn_drop=0.0, drop_path_rate=0.0):
        super().__init__()
        self.conv_in = ConvLayer(in_ch, out_ch, drop_rate=drop_rate)
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            shift_size = 0 if (i % 2 == 0) else window_size // 2
            self.blocks.append(
                UformerTransformerBlock(
                    dim=out_ch,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size,
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                    attn_drop=attn_drop,
                    drop_path_rate=drop_path_rate
                )
            )
        self.channel_attn = ChannelAttention(out_ch, drop_rate=drop_rate)

    def forward(self, x):
        x = self.conv_in(x)
        for blk in self.blocks:
            x = blk(x)
        return self.channel_attn(x)

class EncoderStage(nn.Module):
    def __init__(self, in_ch, out_ch, simple_blocks=3, complex_blocks=2,
                 window_size=8, mlp_ratio=4., drop_rate=0.0, attn_drop=0.0, drop_path_rate=0.0):
        super().__init__()
        self.downsample = nn.Conv2d(in_ch, out_ch, 3, 2, 1) if in_ch != out_ch else nn.Identity()
        self.simple = SimpleBranch(out_ch, out_ch, simple_blocks, drop_rate=drop_rate)
        num_heads = max(2, out_ch // 32)          # 自适应头数
        self.complex = ComplexBranch(out_ch, out_ch, complex_blocks, window_size,
                                     num_heads, mlp_ratio, drop_rate, attn_drop, drop_path_rate)
        self.fuse = ConvLayer(out_ch * 2, out_ch, kernel_size=1, padding=0, drop_rate=drop_rate)

    def forward(self, x):
        x = self.downsample(x)
        simple_out = self.simple(x)
        complex_out = self.complex(x)
        return self.fuse(torch.cat([simple_out, complex_out], dim=1))

# -------------------- 解码器阶段（传递 dropout 参数）--------------------
class DecoderStage(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch, window_size=8, num_heads=4,
                 mlp_ratio=4., drop_rate=0.0, attn_drop=0.0, drop_path_rate=0.0):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.fuse_conv = ConvLayer(out_ch + skip_ch, out_ch, 3, 1, drop_rate=drop_rate)
        self.swin = UformerTransformerBlock(
            dim=out_ch,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop=attn_drop,
            drop_path_rate=drop_path_rate
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse_conv(x)
        x = self.swin(x)
        return x

# -------------------- 完整模型 --------------------
class DSE(nn.Module):
    def __init__(self, img_ch=3, event_ch=1, base_ch=16, window_size=8,
                 drop_rate=0.02,        # 全局 Dropout 率
                 attn_drop=0.05,        # Attention 内部的 Dropout 率
                 drop_path_rate=0.02,   # DropPath 率（一般 0.1~0.2）
                 mlp_ratio=4.0):
        super().__init__()
        self.window_size = window_size
        self.stem_conv = ConvLayer(img_ch, base_ch, drop_rate=drop_rate)
        self.event_norm = nn.BatchNorm2d(event_ch)
        self.fuse_conv = ConvLayer(base_ch + event_ch, base_ch, drop_rate=drop_rate)

        # 编码器
        self.enc_stage1 = EncoderStage(base_ch, base_ch * 2,
                                       simple_blocks=2, complex_blocks=2,
                                       window_size=window_size, mlp_ratio=mlp_ratio,
                                       drop_rate=drop_rate, attn_drop=attn_drop,
                                       drop_path_rate=drop_path_rate)
        self.enc_stage2 = EncoderStage(base_ch * 2, base_ch * 4,
                                       simple_blocks=2, complex_blocks=2,
                                       window_size=window_size, mlp_ratio=mlp_ratio,
                                       drop_rate=drop_rate, attn_drop=attn_drop,
                                       drop_path_rate=drop_path_rate)
        # 瓶颈层（残差块也应用 Dropout）
        self.enc_bottleneck_main = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 8, 3, 2, 1),
            ResidualBlock(base_ch * 8, drop_rate=drop_rate),
            ResidualBlock(base_ch * 8, drop_rate=drop_rate),
            ResidualBlock(base_ch * 8, drop_rate=drop_rate),
            ResidualBlock(base_ch * 8, drop_rate=drop_rate)
        )
        self.enc_bottleneck_skip = nn.Conv2d(base_ch * 4, base_ch * 8, 3, 2, 1)

        # 解码器
        num_heads_dec1 = max(2, (base_ch * 4) // 32)
        num_heads_dec2 = max(2, (base_ch * 2) // 32)
        num_heads_dec3 = max(2, base_ch // 32)

        self.dec_stage1 = DecoderStage(base_ch * 8, base_ch * 4, skip_ch=base_ch * 4,
                                       window_size=window_size, num_heads=num_heads_dec1,
                                       mlp_ratio=mlp_ratio, drop_rate=drop_rate,
                                       attn_drop=attn_drop, drop_path_rate=drop_path_rate)
        self.dec_stage2 = DecoderStage(base_ch * 4, base_ch * 2, skip_ch=base_ch * 2,
                                       window_size=window_size, num_heads=num_heads_dec2,
                                       mlp_ratio=mlp_ratio, drop_rate=drop_rate,
                                       attn_drop=attn_drop, drop_path_rate=drop_path_rate)
        self.dec_stage3 = DecoderStage(base_ch * 2, base_ch, skip_ch=base_ch,
                                       window_size=window_size, num_heads=num_heads_dec3,
                                       mlp_ratio=mlp_ratio, drop_rate=drop_rate,
                                       attn_drop=attn_drop, drop_path_rate=drop_path_rate)

        self.output_conv = nn.Sequential(
            nn.BatchNorm2d(base_ch),
            nn.Conv2d(base_ch, img_ch, kernel_size=1)
        )

    def _pad_to_multiple(self, x):
        H, W = x.shape[-2:]
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0, 0, 0)
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x, (H, W, pad_h, pad_w)

    def _crop_back(self, x, crop_info):
        H, W, pad_h, pad_w = crop_info
        if pad_h == 0 and pad_w == 0:
            return x
        return x[:, :, :H, :W]

    def forward(self, blur_img, event_voxel):
        blur_img_pad, crop_info = self._pad_to_multiple(blur_img)
        event_voxel_pad, _ = self._pad_to_multiple(event_voxel)

        img_feat = self.stem_conv(blur_img_pad)
        event_voxel_pad = self.event_norm(event_voxel_pad)
        fused = torch.cat([img_feat, event_voxel_pad], dim=1)
        x = self.fuse_conv(fused)
        #x = torch.clamp(x, max=3.0)
        #print(f"[fuse_conv]       min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}, std={x.std().item():.4f}")
        skip1 = x

        x = self.enc_stage1(x)
        #x = torch.clamp(x, max=3.0)
        #print(f"[enc_stage1]      min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}, std={x.std().item():.4f}")
        skip2 = x
        x = self.enc_stage2(x)
        #x = torch.clamp(x, max=3.0)
        #print(f"[enc_stage2]      min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}, std={x.std().item():.4f}")
        skip3 = x
        main = self.enc_bottleneck_main(x)
        shortcut = self.enc_bottleneck_skip(x)
        x = main + shortcut   
        #x = torch.clamp(x, max=3.0)
        #print(f"[enc_bottleneck]  min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}, std={x.std().item():.4f}")

        x = self.dec_stage1(x, skip3)
        #x = torch.clamp(x, max=3.0)
        #print(f"[dec_stage1]      min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}, std={x.std().item():.4f}")
        x = self.dec_stage2(x, skip2)
        #x = torch.clamp(x, max=3.0)
        #print(f"[dec_stage2]      min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}, std={x.std().item():.4f}")
        x = self.dec_stage3(x, skip1)
        #x = torch.clamp(x, max=3.0)
        #print(f"[dec_stage3]      min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}, std={x.std().item():.4f}")
        residual = self.output_conv(x) 
        residual = self._crop_back(residual, crop_info)
        out = blur_img + residual
        return out

# -------------------- 训练示例（Weight Decay 设置）--------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型，启用所有正则化（drop_rate=0.1, drop_path=0.1）
    model = DSE(
        img_ch=3, event_ch=6, base_ch=64, window_size=8,
        drop_rate=0.1, attn_drop=0.1, drop_path_rate=0.1
    ).to(device)

    # ========= 关键：优化器中使用 Weight Decay =========
    # AdamW 能将 weight decay 与自适应学习率解耦，推荐用于 Transformer+CNN 混合模型
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        weight_decay=0.01  # 典型值：0.01~0.1，过拟合越严重可适当加大
    )

    # 模拟一次前向传播，确认形状正确
    B = torch.randn(1, 3, 640, 480).to(device)
    M = torch.randn(1, 6, 640, 480).to(device)

    with torch.no_grad():
        output = model(B, M)
        print("输出形状:", output.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params / 1e6:.2f} M")