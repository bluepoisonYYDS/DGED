import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]       # (B, half_dim)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:   # 奇数额外补零
            emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=device)], dim=-1)
        return emb
# model


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, dim), dim)
        self.act = nn.SiLU()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)
    def forward(self, x):
        return self.conv(self.act(self.norm(x)))

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.norm = nn.GroupNorm(min(32, dim), dim)
        self.act = nn.SiLU()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)
    def forward(self, x):
        return self.conv(self.act(self.norm(self.up(x))))


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, dropout=0):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, dim), dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()
        self.conv = nn.Conv2d(dim, dim_out, 3, padding=1)
    def forward(self, x):
        return self.conv(self.drop(self.act(self.norm(x))))

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, dropout=0):
        super().__init__()
        # 第一段：norm -> act -> conv
        self.norm1 = nn.GroupNorm(min(32, dim), dim)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(dim, dim_out, 3, padding=1)

        # 时间调制
        if exists(time_emb_dim):
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, time_emb_dim * 4),
                nn.LayerNorm(time_emb_dim * 4),
                nn.SiLU(),
                nn.Linear(time_emb_dim * 4, dim_out * 2)
            )
        else:
            self.time_mlp = None

        # 第二段：norm -> act -> drop -> conv
        self.norm2 = nn.GroupNorm(min(32, dim_out), dim_out)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(dropout) if dropout else nn.Identity()
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, padding=1)

        # 捷径投影（若通道变化）
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        # 捷径专用归一化（关键！）
        self.shortcut_norm = nn.GroupNorm(min(32, dim_out), dim_out)

        # 主路径可学习缩放，初始为 0（训练初期完全恒等，杜绝方差）
        self.gamma = nn.Parameter(torch.zeros(1, dim_out, 1, 1))

    def forward(self, x, time_emb):
        # 主路径（Pre-Activation）
        h = self.norm1(x)           # x 未经归一化，这里先规范
        h = self.act1(h)
        h = self.conv1(h)

        if self.time_mlp is not None:
            t = self.time_mlp(time_emb)
            scale, shift = t.chunk(2, dim=1)
            h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

        h = self.norm2(h)
        h = self.act2(h)
        h = self.drop2(h)
        h = self.conv2(h)

        # 捷径：投影 + 归一化，确保分布稳定
        shortcut = self.shortcut_norm(self.res_conv(x))

        return h * self.gamma + shortcut


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, time_emb_dim, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x

def _init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.GroupNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class UNet(nn.Module):
    def __init__(
        self,
        in_channel=12,
        out_channel=6,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=((40, 30),),   # 改为元组列表，例如在 40×30 时启用注意力
        res_blocks=3,
        dropout=0,
        with_time_emb=True,
        image_size=(640, 480)   # 修改为 (H, W)
    ):
        super().__init__()

        if with_time_emb:
            time_dim = inner_channel
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(inner_channel),      # 稳定且无关学习
                nn.Linear(inner_channel, inner_channel * 4),
                nn.SiLU(),                            # Swish
                nn.Linear(inner_channel * 4, inner_channel),
                nn.LayerNorm(inner_channel)           # 关键：稳定输出方差
            )
        else:
            time_dim = None
            self.time_mlp = None
            
        #self.time_mlp.requires_grad_(False)

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_h, now_w = image_size  # 分别保存高和宽
        downs = [
            nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, inner_channel), inner_channel),
            nn.SiLU()
        ]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            # 判断当前分辨率是否在 attn_res 中
            use_attn = ((now_h, now_w) in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, time_emb_dim=time_dim,
                    norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_h //= 2
                now_w //= 2
        self.downs = nn.ModuleList(downs)

        # 中间层
        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, time_emb_dim=time_dim,
                               norm_groups=norm_groups, dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, time_emb_dim=time_dim,
                               norm_groups=norm_groups, dropout=dropout, with_attn=False)
        ])

        # 上采样路径
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = ((now_h, now_w) in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(res_blocks + 1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel + feat_channels.pop(), channel_mult, time_emb_dim=time_dim,
                    dropout=dropout, norm_groups=norm_groups, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_h *= 2
                now_w *= 2

        self.ups = nn.ModuleList(ups)
        self.final_conv = Block(pre_channel, default(out_channel, in_channel))
        self.apply(_init_weights)
        for m in self.modules():
            if isinstance(m, nn.Linear) and hasattr(m, 'weight'):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, time):
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channel=12, out_channel=6, inner_channel=32, channel_mults=(1, 2, 4), attn_res=((40, 30),), res_blocks=2, dropout=0.1).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 构造一个 batch 的假数据
    batch_size = 2  # 为节省显存，可设为 2
    x = torch.randn(batch_size, 12, 640, 480, device=device, dtype=torch.float32)
    t = torch.randint(0, 1000, (batch_size,), device=device, dtype=torch.float32)
    # 假设目标是去噪输出，out_channel=6
    target = torch.randn(batch_size, 6, 640, 480, device=device, dtype=torch.float32)

    # 纯 FP32 训练一步（确保没有 autocast 影响）
    model.train()
    optimizer.zero_grad()
    output = model(x, t)
    loss = F.mse_loss(output, target)
    loss.backward()

    # 需要检查的层名（你在问题中列出的）
    target_layers = [
        'ups.16.res_block.mlp.1.weight',      # 注意：这里没有前缀 denoise_fn. 因为模型直接就是 model
        'ups.17.res_block.mlp.1.weight',
        'ups.18.res_block.mlp.1.weight',
        'downs.2.res_block.mlp.1.weight',
        'downs.1.res_block.mlp.1.weight',
        'mid.0.res_block.mlp.1.weight',
        'mid.0.res_block.block1.block.3.weight',
        'mid.1.res_block.block1.block.0.weight',
        'mid.1.res_block.block1.block.3.weight',
        'mid.1.res_block.mlp.1.weight',
        'mid.0.res_block.block1.block.0.weight',
    ]

    print("\n========== FP32 梯度检查 ==========")
    for name, param in model.named_parameters():
        # 为了匹配你的原始命名，可加上前缀 "denoise_fn." 或不加
        if name in target_layers:
            if param.grad is not None:
                grad_mean = param.grad.mean().item()
                grad_max  = param.grad.max().item()
                print(f"{name:60s} | 平均梯度: {grad_mean:12.8f} | 最大梯度: {grad_max:12.8f}")
            else:
                print(f"{name:60s} | 梯度为 None")

if __name__ == "__main__":
    main()