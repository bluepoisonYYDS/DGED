import torch
import torch.nn as nn

# ----------------------------
# 基础 ConvLSTM 单元（不变）
# ----------------------------
class ConvLSTMCell(nn.Module):
    """简单 ConvLSTM 单元"""
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim,
                              kernel_size, padding=padding, bias=True)

    def forward(self, x, h_cur, c_cur):
        # x: (B, C, H, W), h_cur/c_cur: (B, hidden_dim, H, W)
        combined = torch.cat([x, h_cur], dim=1)
        gates = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


# ----------------------------
# 残差块（用于增加参数量）
# ----------------------------
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.conv(x))


# ----------------------------
# 加重版循环稠密化网络
# ----------------------------
class DV(nn.Module):
    """
    将稀疏事件体素 + 模糊图像 -> 稠密边缘图
    增加参数量的手段：
    - hidden_dim = 128
    - 图像特征分支：先提取 blur 的深层特征
    - 输出模块：几个残差块 + 最终输出层
    """
    def __init__(self, event_bins=16, hidden_dim=128, img_feat_dim=32):
        super().__init__()
        self.event_bins = event_bins
        self.hidden_dim = hidden_dim

        # 模糊图像浅层特征提取（增加参数）
        self.img_feat = nn.Sequential(
            nn.Conv2d(3, img_feat_dim, 3, padding=1), nn.ReLU(True),
            ResBlock(img_feat_dim),
            nn.Conv2d(img_feat_dim, img_feat_dim, 3, padding=1), nn.ReLU(True)
        )

        # ConvLSTM 输入: 单bin事件 (1) + 图像特征 (img_feat_dim)
        self.lstm_cell = ConvLSTMCell(input_dim=1 + img_feat_dim, hidden_dim=hidden_dim)

        # 输出稠密边缘图：融合隐藏状态 + 图像特征，过几个残差块
        out_in_ch = hidden_dim + img_feat_dim
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_in_ch, hidden_dim, 3, padding=1), nn.ReLU(True),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),   # 多加残差块，进一步增加参数
            nn.Conv2d(hidden_dim, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, event_voxel, blur_img):
        """
        event_voxel: (B, event_bins, H, W) 稀疏体素
        blur_img:    (B, 3, H, W)
        返回: (B, 1, H, W) 稠密边缘图
        """
        B, T, H, W = event_voxel.shape
        # 提取模糊图像特征（在所有时间步共享）
        feat_img = self.img_feat(blur_img)   # (B, img_feat_dim, H, W)

        # 初始化隐藏状态和细胞状态
        h = torch.zeros(B, self.hidden_dim, H, W, device=event_voxel.device)
        c = torch.zeros_like(h)

        # 逐时间步迭代
        for t in range(T):
            ev_t = event_voxel[:, t:t+1, :, :]            # (B, 1, H, W)
            x_t = torch.cat([ev_t, feat_img], dim=1)      # (B, 1+img_feat_dim, H, W)
            h, c = self.lstm_cell(x_t, h, c)

        # 融合最终状态与图像特征，生成稠密图
        fusion = torch.cat([h, feat_img], dim=1)           # (B, hidden_dim+img_feat_dim, H, W)
        edge_map = self.out_conv(fusion)                  # (B, 1, H, W)
        return edge_map


# ----------------------------
# 测试：计算参数量并检查输出形状
# ----------------------------
if __name__ == "__main__":
    # 参数设置
    B, T, H, W = 2, 6, 256, 256
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 构建加重版模型
    model = HeavyE_CIR(event_bins=T, hidden_dim=128, img_feat_dim=32).to(device)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量（全部）: {total_params:,}")
    print(f"模型参数量（可训练）: {trainable_params:,}")

    # 模拟输入：事件体素（故意做成稀疏）、模糊图像
    event = torch.randn(B, T, H, W, device=device)
    # 稀疏性：90% 位置置零
    mask = (torch.rand(B, T, H, W, device=device) > 0.9).float()
    event = event * mask

    blur = torch.randn(B, 3, H, W, device=device)

    # 前向传播
    with torch.no_grad():
        edge = model(event, blur)

    print(f"输入事件体素形状: {event.shape}")
    print(f"输入模糊图像形状: {blur.shape}")
    print(f"输出稠密边缘图形状: {edge.shape}   (预期: [{B}, 1, {H}, {W}])")

    # 验证不再稀疏（无大量精确零值）
    print(f"输出中绝对值 < 1e-5 的比例: {(edge.abs() < 1e-5).float().mean().item():.4f}")