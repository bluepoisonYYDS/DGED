import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms.functional as TF
import torchvision.models as models
from torchvision.models import vgg19, VGG19_Weights

# 导入原始大模型，稍后将被包装
from DSE import DSE
from DV import DV
from eval import evaluate_psnr_ssim

class JointDeblurModel(nn.Module):
    """
    先使用 DV 将事件体素转化为稠密边缘图，
    再将边缘图与模糊图像一起送入去模糊大模型。
    """
    def __init__(self, deblur_net_args):
        super().__init__()
        # 稠密化网络（固定 event_bins=6，与原数据集一致）
        self.densifier = DV(event_bins=6, hidden_dim=128, img_feat_dim=32)

        # 修改大模型的 event_ch 为 1，使其接收边缘图
        deblur_net_args['event_ch'] = 1
        self.deblur_net = DSE(**deblur_net_args)

    def forward(self, blur_img, event_voxel):
        # 1. 生成稠密边缘图
        edge = self.densifier(event_voxel, blur_img)   # (B, 1, H, W)
        # 2. 将边缘图送入去模糊主干
        return self.deblur_net(blur_img, edge)


# ============================================================
# 3. 数据增强与数据集（沿用原 train_large.py，无需修改）
# ============================================================
class SyncAugmentation:
    """同步空间增强"""
    def __init__(self, crop_size=None, hflip_prob=0.5):
        self.crop_size = crop_size
        self.hflip_prob = hflip_prob

    def __call__(self, blur, sharp, voxel):
        if torch.rand(1).item() < self.hflip_prob:
            blur = torch.flip(blur, dims=[-1])
            sharp = torch.flip(sharp, dims=[-1])
            voxel = torch.flip(voxel, dims=[-1])
        if self.crop_size is not None:
            H, W = blur.shape[-2], blur.shape[-1]
            crop_h, crop_w = self.crop_size
            if crop_h <= H and crop_w <= W:
                top = torch.randint(0, H - crop_h + 1, (1,)).item()
                left = torch.randint(0, W - crop_w + 1, (1,)).item()
                blur = blur[..., top:top+crop_h, left:left+crop_w]
                sharp = sharp[..., top:top+crop_h, left:left+crop_w]
                voxel = voxel[..., top:top+crop_h, left:left+crop_w]
        return blur, sharp, voxel


class CenterCropTransform:
    """同步中心裁剪（验证集用）"""
    def __init__(self, crop_size):
        self.crop_size = crop_size

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


class LoadDataset(Dataset):
    """沿用原数据加载器，无需更改"""
    def __init__(self, root, split='train', transform=None):
        super().__init__()
        self.root = os.path.join(root, split)
        self.transform = transform

        blur_dir = os.path.join(self.root, 'blur')
        sharp_dir = os.path.join(self.root, 'sharp')
        voxel_dir = os.path.join(self.root, 'voxel')

        blur_files = [f for f in os.listdir(blur_dir) if f.lower().endswith('.png')]
        self.samples = []
        for fname in sorted(blur_files):
            name = os.path.splitext(fname)[0]
            blur_path = os.path.join(blur_dir, f'{name}.png')
            sharp_path = os.path.join(sharp_dir, f'{name}.png')
            voxel_path = os.path.join(voxel_dir, f'{name}.npz')
            if os.path.exists(blur_path) and os.path.exists(sharp_path) and os.path.exists(voxel_path):
                self.samples.append((blur_path, sharp_path, voxel_path))
            else:
                print(f"警告：样本 {name} 缺失文件，已跳过")
        print(f"{split} 集共 {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path):
        img = Image.open(path).convert('RGB')
        return np.array(img, dtype=np.float32)

    def _load_voxel(self, path):
        data = np.load(path)
        if 'voxel' in data:
            voxel = data['voxel']
        elif 'arr_0' in data:
            voxel = data['arr_0']
        else:
            voxel = data[list(data.keys())[0]]
        return voxel.astype(np.float32)

    def __getitem__(self, idx):
        blur_path, sharp_path, voxel_path = self.samples[idx]
        blur_np = self._load_image(blur_path) / 255.0
        sharp_np = self._load_image(sharp_path) / 255.0
        voxel_np = normalize_voxel(self._load_voxel(voxel_path))

        blur = torch.from_numpy(blur_np).permute(2, 0, 1)   # (3, H, W)
        sharp = torch.from_numpy(sharp_np).permute(2, 0, 1)
        voxel = torch.from_numpy(voxel_np).permute(2, 0, 1) # (6, H, W)

        if self.transform is not None:
            blur, sharp, voxel = self.transform(blur, sharp, voxel)
        return {'blur': blur, 'events': voxel, 'sharp': sharp}


def normalize_voxel(voxel, max_val=None, min_clip=1e-6):
    """事件体素归一化到 [0,1]"""
    if max_val is None:
        v_flat = voxel.flatten()
        clip_val = np.percentile(v_flat, 99.9)
        if clip_val < min_clip:
            clip_val = v_flat.max()
        if clip_val < min_clip:
            clip_val = 1.0
    else:
        clip_val = max_val
    return np.clip(voxel, 0, clip_val) / clip_val


# ============================================================
# 4. 感知损失（与原训练一致）
# ============================================================
class PerceptualLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        self.vgg = vgg
        self.layers = [9, 22]
        self.loss_fn = nn.L1Loss()

    def forward(self, pred, target):
        pred_feats = self._extract(pred)
        target_feats = self._extract(target)
        loss = 0.0
        for p, t in zip(pred_feats, target_feats):
            loss += self.loss_fn(p, t)
        return loss / len(self.layers)

    def _extract(self, x):
        feats = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers:
                feats.append(x)
        return feats


# ============================================================
# 5. 训练/验证循环
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion_l1, criterion_l2,
                    device, scaler, percep_loss, lambda_p, k=0.8):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc='Train')
    for batch in pbar:
        blur = batch['blur'].to(device)
        events = batch['events'].to(device)
        sharp = batch['sharp'].to(device)

        optimizer.zero_grad()
        with autocast(enabled=scaler is not None):
            pred = model(blur, events)               # 组合模型前向
            loss_l1 = criterion_l1(pred, sharp)
            loss_l2 = criterion_l2(pred, sharp)
            loss = k * loss_l1 + (1 - k) * loss_l2
            if percep_loss is not None:
                loss_percep = percep_loss(pred, sharp)
                loss = loss + lambda_p * loss_percep

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    return total_loss / len(loader)


def validate(model, loader, criterion_l1, criterion_l2, device, percep_loss, lambda_p, k=0.8):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc='val'):
            blur = batch['blur'].to(device)
            events = batch['events'].to(device)
            sharp = batch['sharp'].to(device)

            pred = model(blur, events)
            loss_l1 = criterion_l1(pred, sharp)
            loss_l2 = criterion_l2(pred, sharp)
            loss = k * loss_l1 + (1 - k) * loss_l2
            if percep_loss is not None:
                loss_percep = percep_loss(pred, sharp)
                loss = loss + lambda_p * loss_percep
            total_loss += loss.item()
    return total_loss / len(loader)
def check_gradient_flow(model, save_path='gradient_flow.txt'):
    """
    检查模型各层的梯度平均大小，定位梯度消失的层级，并保存到文件
    """
    ave_grads = []
    max_grads = []
    layers = []

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

        # 按平均梯度升序排序（最小的在前，即最可能消失的层）
        sorted_indices = np.argsort(ave_grads)

        # 统计信息
        dead_layers = [(layers[i], ave_grads[i]) for i in sorted_indices if ave_grads[i] < 1e-8]
        weak_layers = [(layers[i], ave_grads[i]) for i in sorted_indices if 1e-8 <= ave_grads[i] < 1e-6]
        normal_layers = [(layers[i], ave_grads[i]) for i in sorted_indices if ave_grads[i] >= 1e-6]

        f.write(f"💀 完全消失的层 (avg < 1e-8): {len(dead_layers)} 层\n")
        f.write(f"⚠️  梯度极弱的层 (1e-8 ≤ avg < 1e-6): {len(weak_layers)} 层\n")
        f.write(f"✅ 梯度正常的层 (avg ≥ 1e-6): {len(normal_layers)} 层\n")
        f.write(f"📊 总检查层数: {len(layers)}\n\n")

        # 打印完全消失的层
        if dead_layers:
            f.write("-"*100 + "\n")
            f.write("💀 完全消失的层详情:\n")
            f.write("-"*100 + "\n")
            for i, (name, avg) in enumerate(dead_layers):
                idx = sorted_indices[i]
                f.write(f"  {i+1:3d}. {name:70s} | 平均梯度: {avg:12.8f} | 最大梯度: {max_grads[idx]:12.8f}\n")
            f.write("\n")

        # 打印梯度极弱的层
        if weak_layers:
            f.write("-"*100 + "\n")
            f.write("⚠️  梯度极弱的层详情:\n")
            f.write("-"*100 + "\n")
            start_idx = len(dead_layers)
            for i, (name, avg) in enumerate(weak_layers):
                idx = sorted_indices[start_idx + i]
                f.write(f"  {start_idx+i+1:3d}. {name:70s} | 平均梯度: {avg:12.8f} | 最大梯度: {max_grads[idx]:12.8f}\n")
            f.write("\n")

        # 打印梯度最小的前 30 层（无论状态）
        f.write("-"*100 + "\n")
        f.write("梯度最小的前 30 层（升序）:\n")
        f.write("-"*100 + "\n")
        top_n = min(30, len(layers))
        for i in range(top_n):
            idx = sorted_indices[i]
            status = ""
            if ave_grads[idx] < 1e-8:
                status = "💀 消失"
            elif ave_grads[idx] < 1e-6:
                status = "⚠️ 极弱"
            f.write(f"  {i+1:3d}. {layers[idx]:70s} | 平均: {ave_grads[idx]:12.8f} | 最大: {max_grads[idx]:12.8f} | {status}\n")
        
        # 打印梯度最大的前 10 层
        f.write("\n" + "-"*100 + "\n")
        f.write("梯度最大的前 10 层（降序，可能是梯度爆炸候选）:\n")
        f.write("-"*100 + "\n")
        sorted_indices_desc = np.argsort(ave_grads)[::-1]
        for i in range(min(10, len(layers))):
            idx = sorted_indices_desc[i]
            f.write(f"  {i+1:3d}. {layers[idx]:70s} | 平均: {ave_grads[idx]:12.8f}\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("报告结束\n")

    print(f"\n✅ 梯度流动分析报告已保存至: {save_path}")

    # 同时在控制台输出摘要
    print(f"   💀 完全消失: {len(dead_layers)} 层")
    print(f"   ⚠️  梯度极弱: {len(weak_layers)} 层")
    print(f"   ✅ 梯度正常: {len(normal_layers)} 层")

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # --- 数据集 ---
    crop_size = args.crop_size
    if crop_size is not None and isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    train_transform = SyncAugmentation(crop_size=crop_size, hflip_prob=0) if not args.no_augment else None
    val_transform = CenterCropTransform(crop_size) if (crop_size is not None and not args.no_augment) else None

    train_dataset = LoadDataset(args.data_root, split='train', transform=train_transform)
    val_dataset = LoadDataset(args.data_root, split='val', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # --- 构建联合模型 ---
    # 传递原大模型所需的参数（event_ch 会在 JointDeblurModel 内部被设为1）
    deblur_net_args = dict(
        img_ch=3,
        base_ch=args.base_ch,
        up_ch=args.up_ch,
        window_size=args.window_size,
        drop_rate=args.drop_rate,
        attn_drop=args.attn_drop,
        drop_path_rate=args.drop_path_rate
    )
    model = JointDeblurModel(deblur_net_args).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"联合模型总参数量: {total_params / 1e6:.2f} M")

    # --- 损失与优化器 ---
    criterion_l1 = nn.L1Loss()
    criterion_l2 = nn.MSELoss()
    percep_loss = PerceptualLoss(device=device) if args.use_perceptual else None

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=2, eta_min=1e-6
    )

    # 学习率预热
    warmup_epochs = 5
    base_lr = args.lr
    warmup_lr_init = 1e-6
    def adjust_learning_rate(optimizer, epoch, warmup_epochs, base_lr, warmup_lr_init):
        if epoch < warmup_epochs:
            lr = warmup_lr_init + (base_lr - warmup_lr_init) * ((epoch + 1) / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        return base_lr

    # 混合精度
    scaler = GradScaler() if args.use_amp else None

    # 恢复训练
    start_epoch = 0
    dv_epoch = 0
    dse_epoch = 0
    best_val_loss = float('inf')
    if args.resume_DV and os.path.isfile(args.resume_DV):
        dv_state = torch.load(args.resume_DV, map_location=device)
        model.densifier.load_state_dict(dv_state['model_state_dict'])
        optimizer.load_state_dict(dv_state['optimizer_state_dict'])
        if scaler and 'scaler' in dv_state:
            scaler.load_state_dict(dv_state['scaler'])
        dv_epoch = dv_state['epoch'] + 1
        best_val_loss = dv_state.get('best_val_loss', best_val_loss)
        print(f"从 DV checkpoint 恢复，epoch: {dv_epoch}, val loss: {best_val_loss:.6f}")
    if args.resume_DSE and os.path.isfile(args.resume_DSE):
        dse_state = torch.load(args.resume_DSE, map_location=device)
        model.deblur_net.load_state_dict(dse_state['model_state_dict'])
        dse_epoch = dse_state['epoch'] + 1
        best_val_loss = min(best_val_loss, dse_state.get('best_val_loss', float('inf')))
        print(f"从 DSE checkpoint 恢复，epoch: {dse_epoch}, val loss: {best_val_loss:.6f}")   
    start_epoch = max(dv_epoch, dse_epoch)
    if(dv_epoch != dse_epoch):
        print(f"警告：DV checkpoint epoch ({dv_epoch}) 与 DSE checkpoint epoch ({dse_epoch}) 不一致，可能导致训练不连续！")

    os.makedirs(args.save_dir_DV, exist_ok=True)
    os.makedirs(args.save_dir_DSE, exist_ok=True)
    os.makedirs(args.save_dir_log, exist_ok=True)
    # 创建评估日志文件
    eval_log_path = os.path.join(args.save_dir_log, 'eval_metrics.txt')
    if start_epoch == 0:  # 仅在初始训练时创建并写入表头
        with open(eval_log_path, 'w') as f:
            f.write("Epoch\tPSNR(dB)\tSSIM\n")
    # 创建损失日志文件
    loss_log_path = os.path.join(args.save_dir_log, 'loss_log.txt')
    if start_epoch == 0:  # 仅在初始训练时创建并写入表头
        with open(loss_log_path, 'w') as f:
            f.write("Epoch\tTrain Loss\tval Loss\n")

    # --- 训练循环 ---
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        adjust_learning_rate(optimizer, epoch, warmup_epochs, base_lr, warmup_lr_init)

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_l1, criterion_l2,
                                     device, scaler, percep_loss, args.lambda_percep, args.k)
        val_loss = validate(model, val_loader, criterion_l1, criterion_l2,
                            device, percep_loss, args.lambda_percep, args.k)

        if scheduler is not None:
            scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        lr = adjust_learning_rate(optimizer, epoch, warmup_epochs, base_lr, warmup_lr_init)
        print(f"Train Loss: {train_loss:.6f} | val Loss: {val_loss:.6f} | LR: {lr:.6f}")

        #查看梯度消失情况
        if epoch == start_epoch or (epoch + 1) % 10 == 0:  # 每隔 10 个 epoch 检查一次梯度流
            save_path = os.path.join(args.save_dir_log, f'gradient_flow_epoch{epoch+1}.txt')
            check_gradient_flow(model, save_path=save_path)
        # 日志记录
        with open(loss_log_path, 'a') as f:
            f.write(f"{epoch + 1}\t{train_loss:.6f}\t{val_loss:.6f}\n")

        if (epoch + 1) % args.eval_every == 0:
            avg_psnr, avg_ssim = evaluate_psnr_ssim(model, val_loader, device)
            print(f"val PSNR: {avg_psnr:.2f} dB | val SSIM: {avg_ssim:.4f}")
            with open(eval_log_path, 'a') as f:
                f.write(f"{epoch + 1}\t{avg_psnr:.4f}\t{avg_ssim:.4f}\n")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.densifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'args': args,
            },os.path.join(args.save_dir_DV, 'DV_best.pth'))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.deblur_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'args': args,
            }, os.path.join(args.save_dir_DSE, 'DSE_best.pth'))
            print(f"保存最佳模型，测试损失: {val_loss:.6f}")

        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.densifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'args': args,
            },os.path.join(args.save_dir_DV, f'DV_{epoch+1}.pth'))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.deblur_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'args': args,
            }, os.path.join(args.save_dir_DSE, f'DSE_{epoch+1}.pth'))

    print("联合训练完成！")


# ============================================================
# 7. 参数解析与入口
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--base_ch', type=int, default=64)
    parser.add_argument('--up_ch', type=int, default=16)
    parser.add_argument('--window_size', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=2e-4)
    parser.add_argument('--drop_rate', type=float, default=0.0)
    parser.add_argument('--attn_drop', type=float, default=0.0)
    parser.add_argument('--drop_path_rate', type=float, default=0.0)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--use_perceptual', action='store_true')
    parser.add_argument('--lambda_percep', type=float, default=0.1)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--save_dir_DSE', type=str, default='/root/autodl-tmp/Mymodel_Improved/DSE_ckpt')
    parser.add_argument('--save_dir_DV', type=str, default='/root/autodl-tmp/Mymodel_Improved/DV_ckpt')
    parser.add_argument('--save_dir_log', type=str, default='/root/autodl-tmp/Mymodel_Improved/DV-DSE_log')
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--resume_DSE', type=str, default='')
    parser.add_argument('--resume_DV', type=str, default='')
    parser.add_argument('--k', type=float, default=0.5)
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--crop_size', type=int, default=320)
    parser.add_argument('--no_augment', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.save_dir_DSE, exist_ok=True)
    os.makedirs(args.save_dir_DV, exist_ok=True)
    os.makedirs(args.save_dir_log, exist_ok=True)
    main(args)

'''
python /root/autodl-tmp/Mymodel_Improved/train_joint.py --data_root root_to_your_dataset --use_perceptual --use_amp
'''