import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
import gc

import math
import argparse
import numpy as np
import h5py
import torch
gc.collect()
torch.cuda.empty_cache()
torch.cuda.init()
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import glob
from PIL import Image

# 导入扩散模型相关类
from diffusion_improved import FastDiffusion
from unet import UNet

# ============ 新增：导入冻结的 E‑CIR 模型 ============
from DV import HeavyE_CIR   # 假设文件名为 E_CIR.py，类名 HeavyE_CIR

class CenterCropTransform:
    """对 blur, sharp, voxel 进行同步中心裁剪"""
    def __init__(self, crop_size):
        self.crop_size = crop_size          # (H, W) 或 int

    def __call__(self, blur, sharp, voxel):
        H, W = blur.shape[-2], blur.shape[-1]
        if isinstance(self.crop_size, int):
            crop_h, crop_w = self.crop_size, self.crop_size
        else:
            crop_h, crop_w = self.crop_size
        top = (H - crop_h) // 2
        left = (W - crop_w) // 2
        blur = blur[..., top:top+crop_h, left:left+crop_w]
        sharp = sharp[..., top:top+crop_h, left:left+crop_w]
        voxel = voxel[..., top:top+crop_h, left:left+crop_w]
        return blur, sharp, voxel

# -------------------- 数据集定义 --------------------
class LoadDataset(Dataset):
    """
    从文件夹读取 blur (PNG), sharp (PNG), voxel (NPZ)。
    目录结构要求：
        root/
          train/
            blur/        # *.png
            sharp/       # *.png
            voxel/       # *.npz (内部数组形状 (H,W,6) 或 (6,H,W))
          val/
            ... (同上)
    transform: 接受 (blur, sharp, voxel) 并返回增强后的三元组。
    """
    def __init__(self, root, split='train', transform=None):
        super().__init__()
        self.root = os.path.join(root, split)
        self.transform = transform

        blur_dir  = os.path.join(self.root, 'blur')
        sharp_dir = os.path.join(self.root, 'sharp')
        voxel_dir = os.path.join(self.root, 'voxel')

        blur_files = [f for f in os.listdir(blur_dir) if f.lower().endswith('.png')]
        self.samples = []
        for fname in sorted(blur_files):
            name = os.path.splitext(fname)[0]
            blur_path  = os.path.join(blur_dir, f'{name}.png')
            sharp_path = os.path.join(sharp_dir, f'{name}.png')
            voxel_path = os.path.join(voxel_dir, f'{name}.npz')
            if os.path.exists(blur_path) and os.path.exists(sharp_path) and os.path.exists(voxel_path):
                self.samples.append((blur_path, sharp_path, voxel_path))
            else:
                print(f"警告：样本 {name} 缺失文件，已跳过")

        print(f"{split} 集共扫描到 {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path):
        """加载 PNG 并返回 RGB numpy 数组 (H, W, 3)，值域 [0,255]"""
        img = Image.open(path).convert('RGB')
        return np.array(img, dtype=np.float32)

    def _load_voxel(self, path):
        """加载 NPZ 体素，返回 numpy 数组 (H, W, 6) 或 (6, H, W)"""
        data = np.load(path)
        if 'voxel' in data:
            voxel = data['voxel']
        elif 'arr_0' in data:
            voxel = data['arr_0']
        else:
            voxel = data[list(data.keys())[0]]
        return voxel.astype(np.float32)

    def _down_up(self, img_tensor):
        """下采样再上采样回原尺寸，用于生成 T(B)"""
        _, h, w = img_tensor.shape
        small = torch.nn.functional.interpolate(
            img_tensor.unsqueeze(0), scale_factor=0.25, mode='bilinear', align_corners=False
        )
        up = torch.nn.functional.interpolate(
            small, size=(h, w), mode='bilinear', align_corners=False
        )
        return up.squeeze(0)

    def __getitem__(self, idx):
        blur_path, sharp_path, voxel_path = self.samples[idx]

        # 1. 读取原始数据
        blur_np  = self._load_image(blur_path)    # (H, W, 3), [0,255]
        sharp_np = self._load_image(sharp_path)   # (H, W, 3), [0,255]
        voxel_np = self._load_voxel(voxel_path)   # 可能是 (H, W, 6) 或 (6, H, W)

        # 2. 归一化到 [-1, 1]
        blur_np  = blur_np.astype(np.float32) / 127.5 - 1
        sharp_np = sharp_np.astype(np.float32) / 127.5 - 1
        voxel_np = normalize_voxel(voxel_np).astype(np.float32)

        # 3. 转为 Tensor 并调整到 (C, H, W)
        blur  = torch.from_numpy(blur_np).permute(2, 0, 1)  # (3, H, W)
        sharp = torch.from_numpy(sharp_np).permute(2, 0, 1) # (3, H, W)
        voxel = torch.from_numpy(voxel_np).permute(2, 0, 1) # (6, H, W)

        if self.transform is not None:
            blur, sharp, voxel = self.transform(blur, sharp, voxel)

        # 计算细节引导 P = blur - T(blur)
        T_blur = self._down_up(blur)
        P = blur - T_blur   # (3, H, W)
        condition = torch.cat([blur, P], dim=0)  # (6, H, W)

        # 返回字典（HR 依然是 6 通道 voxel，后续在训练循环中会被 E‑CIR 转换）
        return {
            'condition': condition,   # 6 通道：blur 3 + 细节引导 3
            'HR': voxel               # 6 通道事件体素，值域 [-1,1]
        }

def normalize_voxel(voxel, max_val=None, min_clip=1e-6):
    if max_val is None:
        v_flat = voxel.flatten()
        clip_val = np.percentile(v_flat, 99.9)
        if clip_val < min_clip:
            clip_val = v_flat.max()
        if clip_val < min_clip:
            clip_val = 1.0
    else:
        clip_val = max_val
    voxel_norm = np.clip(voxel, 0, clip_val) / clip_val
    return voxel_norm

def check_gradient_flow(model, save_path='gradient_flow.txt'):
    ave_grads, max_grads, layers = [], [], []
    for n, p in model.named_parameters():
        if p.grad is not None and 'bias' not in n:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())
    if not ave_grads:
        print("未找到可检查的梯度")
        return
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("梯度流动分析报告\n")
        f.write("="*100 + "\n\n")
        sorted_indices = np.argsort(ave_grads)
        dead_layers = [(layers[i], ave_grads[i]) for i in sorted_indices if ave_grads[i] < 1e-8]
        weak_layers = [(layers[i], ave_grads[i]) for i in sorted_indices if 1e-8 <= ave_grads[i] < 1e-6]
        normal_layers = [(layers[i], ave_grads[i]) for i in sorted_indices if ave_grads[i] >= 1e-6]
        f.write(f"💀 完全消失的层 (avg < 1e-8): {len(dead_layers)} 层\n")
        f.write(f"⚠️  梯度极弱的层 (1e-8 ≤ avg < 1e-6): {len(weak_layers)} 层\n")
        f.write(f"✅ 梯度正常的层 (avg ≥ 1e-6): {len(normal_layers)} 层\n")
        f.write(f"📊 总检查层数: {len(layers)}\n\n")
        if dead_layers:
            f.write("-"*100 + "\n💀 完全消失的层详情:\n" + "-"*100 + "\n")
            for i, (name, avg) in enumerate(dead_layers):
                idx = sorted_indices[i]
                f.write(f"  {i+1:3d}. {name:70s} | 平均梯度: {avg:12.8f} | 最大梯度: {max_grads[idx]:12.8f}\n")
            f.write("\n")
        if weak_layers:
            f.write("-"*100 + "\n⚠️  梯度极弱的层详情:\n" + "-"*100 + "\n")
            start_idx = len(dead_layers)
            for i, (name, avg) in enumerate(weak_layers):
                idx = sorted_indices[start_idx + i]
                f.write(f"  {start_idx+i+1:3d}. {name:70s} | 平均梯度: {avg:12.8f} | 最大梯度: {max_grads[idx]:12.8f}\n")
            f.write("\n")
        f.write("-"*100 + "\n梯度最小的前 30 层（升序）:\n" + "-"*100 + "\n")
        top_n = min(30, len(layers))
        for i in range(top_n):
            idx = sorted_indices[i]
            status = ""
            if ave_grads[idx] < 1e-8:
                status = "💀 消失"
            elif ave_grads[idx] < 1e-6:
                status = "⚠️ 极弱"
            f.write(f"  {i+1:3d}. {layers[idx]:70s} | 平均: {ave_grads[idx]:12.8f} | 最大: {max_grads[idx]:12.8f} | {status}\n")
        f.write("\n" + "-"*100 + "\n梯度最大的前 10 层（降序）:\n" + "-"*100 + "\n")
        sorted_indices_desc = np.argsort(ave_grads)[::-1]
        for i in range(min(10, len(layers))):
            idx = sorted_indices_desc[i]
            f.write(f"  {i+1:3d}. {layers[idx]:70s} | 平均: {ave_grads[idx]:12.8f}\n")
        f.write("\n" + "="*100 + "\n报告结束\n")
    print(f"\n✅ 梯度流动分析报告已保存至: {save_path}")
    print(f"   💀 完全消失: {len(dead_layers)} 层")
    print(f"   ⚠️  梯度极弱: {len(weak_layers)} 层")
    print(f"   ✅ 梯度正常: {len(normal_layers)} 层")

# -------------------- 模型构建（扩散部分，不变）--------------------
def build_model(args, device):
    unet = UNet(
        in_channel=7,           # condition(6) + densemap(1)
        out_channel=1,          # densemap 通道数
        inner_channel=args.inner_channel,
        norm_groups=args.norm_groups,
        channel_mults=args.channel_mults,
        attn_res=args.attn_res,
        res_blocks=args.res_blocks,
        dropout=args.dropout,
        with_time_emb=True,
        image_size=args.image_size
    ).to(device)

    diffusion = FastDiffusion(
        denoise_fn=unet,
        channels=1,             # densemap 通道数
        image_size=args.image_size,
        loss_type=args.loss_type,
        conditional=True,
        schedule_opt=None
    ).to(device)

    schedule_opt = {
        'schedule': 'cosine',
        'n_timestep': 1000,
        'linear_start': 1e-4,
        'linear_end': 2e-2,
    }
    diffusion.set_new_noise_schedule(schedule_opt, device)
    diffusion.set_loss(device)
    return diffusion

# -------------------- 训练主函数 --------------------
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = args.image_size
    transform = CenterCropTransform(image_size)

    # 1. 数据加载
    train_dataset = LoadDataset(root=args.data_root, split='train', transform=transform)
    val_dataset = LoadDataset(root=args.data_root, split='val', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # ============ 2. 加载冻结的 E‑CIR 模型 ============
    print("Loading frozen E-CIR model...")
    e_cir = HeavyE_CIR(event_bins=6, hidden_dim=128, img_feat_dim=32).to(device)
    if args.e_cir_weights and os.path.exists(args.e_cir_weights):
        e_cir.load_state_dict(torch.load(args.e_cir_weights, map_location=device))
        print(f"Loaded E-CIR weights from {args.e_cir_weights}")
    else:
        print("Warning: E-CIR weights not provided or not found, using random initialization (will produce meaningless edges).")
    e_cir.eval()
    for param in e_cir.parameters():
        param.requires_grad = False
    # ===============================================

    # 3. 构建扩散模型
    model = build_model(args, device)
    scaler = GradScaler()

    # 4. 优化器与调度器
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None

    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', best_val_loss)
        print(f"从 epoch {start_epoch} 恢复训练")

    os.makedirs(args.save_dir, exist_ok=True)
    loss_log_path = args.loss_log if args.loss_log else os.path.join(args.save_dir, 'loss_log.txt')
    if start_epoch == 0:
        with open(loss_log_path, 'w') as f:
            f.write("Epoch\tTrain Loss\tval Loss\n")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
        for step, batch in enumerate(pbar):
            condition = batch['condition'].to(device, non_blocking=True)  # (B,6,H,W)
            hr_voxel = batch['HR'].to(device, non_blocking=True)         # (B,6,H,W)

            # ---------- 关键修改：将稀疏体素转换为稠密边缘图 ----------
            blur = condition[:, :3, :, :]   # 提取模糊图像
            with torch.no_grad():
                edge_map = e_cir(hr_voxel, blur)   # (B,1,H,W) 稠密边缘
            # --------------------------------------------------------

            optimizer.zero_grad(set_to_none=True)
            with autocast(dtype=torch.bfloat16):
                loss = model({'condition': condition, 'HR': edge_map})

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
            if step % 50 == 0:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        avg_train_loss = np.mean(train_losses)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # 验证
        model.eval()
        torch.cuda.empty_cache()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]'):
                condition = batch['condition'].to(device)
                hr_voxel = batch['HR'].to(device)

                # 同样用 E‑CIR 生成边缘图作为目标
                blur = condition[:, :3, :, :]
                edge_map = e_cir(hr_voxel, blur)

                loss = model({'condition': condition, 'HR': edge_map})
                val_losses.append(loss.item())

                del condition, hr_voxel, blur, edge_map, loss
                torch.cuda.synchronize()
                if len(val_losses) % 50 == 0:
                    torch.cuda.empty_cache()

        avg_val_loss = np.mean(val_losses)
        if scheduler is not None:
            scheduler.step()

        print(f'Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {optimizer.param_groups[0]["lr"]:.2e}')
        with open(loss_log_path, 'a') as f:
            f.write(f"{epoch+1}\t{avg_train_loss:.6f}\t{avg_val_loss:.6f}\n")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f'  -> Saved best model (val_loss={best_val_loss:.6f})')

        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(args.save_dir, f'checkpoint_epoch{epoch}.pth'))

        if epoch == start_epoch or (epoch + 1) % 10 == 0:
            save_path = os.path.join(args.save_dir, f'gradient_flow_epoch{epoch+1}.txt')
            check_gradient_flow(model, save_path=save_path)

    print('Training finished.')


# -------------------- 参数解析 --------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train diffusion model for blur->voxel (with frozen E-CIR)')
    # 数据
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--image_size', type=int, nargs=2, default=[320, 320])
    # 模型
    parser.add_argument('--inner_channel', type=int, default=32)
    parser.add_argument('--norm_groups', type=int, default=32)
    parser.add_argument('--channel_mults', type=int, nargs='+', default=[1,2,4,8,8])
    parser.add_argument('--attn_res', type=int, nargs='+', default=[16])
    parser.add_argument('--res_blocks', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)
    # 扩散过程
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--beta_schedule', type=str, default='cosine')
    parser.add_argument('--linear_start', type=float, default=1e-4)
    parser.add_argument('--linear_end', type=float, default=2e-2)
    parser.add_argument('--loss_type', type=str, default='l2', choices=['l1', 'l2', 'mixed'])
    # 训练
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=16)
    # 保存与恢复
    parser.add_argument('--resume', type=str, default='', help='恢复训练的检查点路径')
    parser.add_argument('--loss_log', type=str, default='/root/autodl-tmp/Mymodel_Improved/diff_ckpt/loss_log.txt', help='损失记录文件')
    parser.add_argument('--save_dir', type=str, default='/root/autodl-tmp/Mymodel_Improved/diff_ckpt')
    parser.add_argument('--save_every', type=int, default=5)
    # ============ 新增：E‑CIR 权重路径 ============
    parser.add_argument('--e_cir_weights', type=str, default='/root/autodl-tmp/Mymodel_Improved/model_large_ckpt/e_cir_best.pth', help='Path to pre-trained HeavyE_CIR weights (.pth)')
    # =============================================

    args = parser.parse_args()
    args.image_size = tuple(args.image_size)
    os.makedirs(args.save_dir, exist_ok=True)

    train(args)