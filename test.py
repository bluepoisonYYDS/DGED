"""
eval_deblur.py
读取联合去模糊模型权重，在指定数据集上进行评估，输出损失与图像质量指标（PSNR、SSIM）。
"""
import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.models as models
from torchvision.models import vgg19, VGG19_Weights
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 导入模型定义（确保这些模块与训练时一致，且在同一目录或PYTHONPATH中）
from model_large import EventImageDeblurNet
from DV import HeavyE_CIR

# ============================================================
# 1. 模型定义（与训练代码完全一致）
# ============================================================
class JointDeblurModel(nn.Module):
    """同训练脚本，先稠密化事件体素，再送入大模型去模糊"""
    def __init__(self, deblur_net_args):
        super().__init__()
        self.densifier = HeavyE_CIR(event_bins=6, hidden_dim=128, img_feat_dim=32)
        deblur_net_args['event_ch'] = 1   # 将事件通道数固定为1，接收边缘图
        self.deblur_net = EventImageDeblurNet(**deblur_net_args)

    def forward(self, blur_img, event_voxel):
        edge = self.densifier(event_voxel, blur_img)   # (B,1,H,W)
        return self.deblur_net(blur_img, edge)

# ============================================================
# 2. 感知损失（与训练代码一致，仅用于损失评估）
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
# 3. 数据集（验证/测试集，无数据增强）
# ============================================================
class CenterCropTransform:
    """同步中心裁剪（尺寸与训练一致）"""
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
    """数据加载器，与训练代码一致"""
    def __init__(self, root, split='val', transform=None):
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
        voxel_np = self._normalize_voxel(self._load_voxel(voxel_path))

        blur = torch.from_numpy(blur_np).permute(2, 0, 1)   # (3, H, W)
        sharp = torch.from_numpy(sharp_np).permute(2, 0, 1)
        voxel = torch.from_numpy(voxel_np).permute(2, 0, 1) # (6, H, W)

        if self.transform is not None:
            blur, sharp, voxel = self.transform(blur, sharp, voxel)
        return {'blur': blur, 'events': voxel, 'sharp': sharp, 'name': os.path.splitext(os.path.basename(blur_path))[0]}

    @staticmethod
    def _normalize_voxel(voxel, max_val=None, min_clip=1e-6):
        """事件体素归一化到[0,1]"""
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
# 4. 评估核心函数
# ============================================================
@torch.no_grad()
def evaluate(model, dataloader, device, crop_size, loss_mode='l1',
             lambda_percep=0.1, save_images=False, output_dir=None):
    """
    评估模型，返回平均损失、PSNR、SSIM
    loss_mode: 'l1', 'l2', 'l1+l2', 'all' (包含感知损失)
    """
    model.eval()

    criterion_l1 = nn.L1Loss()
    criterion_l2 = nn.MSELoss()
    percep_loss_fn = None
    if loss_mode == 'all':
        percep_loss_fn = PerceptualLoss(device=device)

    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0

    # 如果保存图像，创建输出目录
    if save_images and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for batch in tqdm(dataloader, desc='Evaluating'):
        blur = batch['blur'].to(device)
        events = batch['events'].to(device)
        sharp = batch['sharp'].to(device)
        names = batch['name']

        # 前向传播，得到去模糊图像
        pred = model(blur, events)

        # 计算损失
        loss = 0.0
        if loss_mode == 'l1':
            loss = criterion_l1(pred, sharp)
        elif loss_mode == 'l2':
            loss = criterion_l2(pred, sharp)
        elif loss_mode == 'l1+l2':
            loss = 0.8 * criterion_l1(pred, sharp) + 0.2 * criterion_l2(pred, sharp)
        elif loss_mode == 'all':
            l1 = criterion_l1(pred, sharp)
            l2 = criterion_l2(pred, sharp)
            percep = percep_loss_fn(pred, sharp)
            loss = 0.8 * l1 + 0.2 * l2 + lambda_percep * percep
        total_loss += loss.item()

        # 计算 PSNR 和 SSIM（逐图像）
        pred_np = pred.cpu().numpy().transpose(0, 2, 3, 1)   # (B, H, W, C)
        sharp_np = sharp.cpu().numpy().transpose(0, 2, 3, 1)

        for i in range(pred_np.shape[0]):
            pred_img = np.clip(pred_np[i], 0.0, 1.0)
            sharp_img = np.clip(sharp_np[i], 0.0, 1.0)

            # PSNR
            try:
                p = psnr(sharp_img, pred_img, data_range=1.0)
            except Exception:
                p = 0.0
            total_psnr += p

            # SSIM（多通道，逐通道平均）
            try:
                s = ssim(sharp_img, pred_img, data_range=1.0, channel_axis=-1,
                         win_size=min(7, min(pred_img.shape[0], pred_img.shape[1])))
            except Exception:
                s = 0.0
            total_ssim += s

            num_samples += 1

        # 可选：保存去模糊结果
        if save_images and output_dir:
            for i, name in enumerate(names):
                img_save = np.clip(pred_np[i] * 255.0, 0, 255).astype(np.uint8)
                Image.fromarray(img_save).save(os.path.join(output_dir, f"{name}_deblur.png"))

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    avg_psnr = total_psnr / num_samples if num_samples > 0 else 0
    avg_ssim = total_ssim / num_samples if num_samples > 0 else 0
    return avg_loss, avg_psnr, avg_ssim

# ============================================================
# 5. 主函数
# ============================================================
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---------- 数据集 ----------
    crop_size = args.crop_size
    if crop_size is not None and isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    transform = CenterCropTransform(crop_size) if crop_size is not None else None

    dataset = LoadDataset(args.data_root, split=args.split, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # ---------- 构建模型 ----------
    deblur_net_args = dict(
        img_ch=3,
        base_ch=args.base_ch,
        window_size=args.window_size,
        drop_rate=args.drop_rate,
        attn_drop=args.attn_drop,
        drop_path_rate=args.drop_path_rate
    )
    model = JointDeblurModel(deblur_net_args).to(device)

    # ---------- 加载权重 ----------
    if args.ckpt and os.path.isfile(args.ckpt):
        print(f"Loading full joint model checkpoint from {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device)
        # 兼容只保存了 model_state_dict 或整个字典
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt
        model.load_state_dict(state_dict, strict=True)
        print("Full model loaded successfully.")
    elif args.dense_ckpt and args.deblur_ckpt:
        print(f"Loading densifier from {args.dense_ckpt}")
        model.densifier.load_state_dict(torch.load(args.dense_ckpt, map_location=device))
        print(f"Loading deblur_net from {args.deblur_ckpt}")
        model.deblur_net.load_state_dict(torch.load(args.deblur_ckpt, map_location=device))
        print("Separate model parts loaded.")
    else:
        raise ValueError("必须指定 --ckpt（完整权重） 或同时提供 --dense_ckpt 和 --deblur_ckpt")

    model.eval()

    # ---------- 评估 ----------
    avg_loss, avg_psnr, avg_ssim = evaluate(
        model, loader, device, crop_size,
        loss_mode=args.loss_mode,
        lambda_percep=args.lambda_percep,
        save_images=args.save_images,
        output_dir=args.output_dir
    )

    print("\n================= Evaluation Results =================")
    print(f"Dataset split  : {args.split}")
    print(f"Loss mode      : {args.loss_mode}")
    print(f"Average Loss   : {avg_loss:.6f}")
    print(f"Average PSNR   : {avg_psnr:.4f} dB")
    print(f"Average SSIM   : {avg_ssim:.6f}")
    print("======================================================")

    # 保存结果到文件
    output_log = args.output_log
    if output_log:
        os.makedirs(os.path.dirname(output_log) if os.path.dirname(output_log) else '.', exist_ok=True)
        with open(output_log, 'w') as f:
            f.write(f"Split: {args.split}\n")
            f.write(f"Loss mode: {args.loss_mode}\n")
            f.write(f"Average Loss: {avg_loss:.6f}\n")
            f.write(f"Average PSNR: {avg_psnr:.4f}\n")
            f.write(f"Average SSIM: {avg_ssim:.6f}\n")
        print(f"Results saved to {output_log}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='评估联合去模糊模型')
    # 数据参数
    parser.add_argument('--data_root', type=str, required=True, help='数据集根目录，包含 train/val/test 子文件夹')
    parser.add_argument('--split', type=str, default='val', help='评估哪个子集 (val/test)')
    parser.add_argument('--crop_size', type=int, default=320, help='中心裁剪尺寸，若不需要则设为0')
    parser.add_argument('--batch_size', type=int, default=1, help='评估时的批大小')
    parser.add_argument('--num_workers', type=int, default=8)

    # 模型结构参数（需与训练时一致）
    parser.add_argument('--base_ch', type=int, default=64)
    parser.add_argument('--window_size', type=int, default=8)
    parser.add_argument('--drop_rate', type=float, default=0.1)
    parser.add_argument('--attn_drop', type=float, default=0.05)
    parser.add_argument('--drop_path_rate', type=float, default=0.05)

    # 权重文件
    parser.add_argument('--ckpt', type=str, default=None, help='完整联合模型权重文件 (例如 best_model_joint.pth)')
    parser.add_argument('--dense_ckpt', type=str, default=None, help='稠密化网络权重 (e_cir_best.pth)')
    parser.add_argument('--deblur_ckpt', type=str, default=None, help='去模糊大模型权重 (deblur_net_best.pth)')

    # 评估选项
    parser.add_argument('--loss_mode', type=str, default='l1', choices=['l1', 'l2', 'l1+l2', 'all'],
                        help='损失计算方式：l1, l2, l1+l2(0.8 0.2), all(含感知损失)')
    parser.add_argument('--lambda_percep', type=float, default=0.1, help='感知损失权重（仅在 loss_mode=all 时生效）')
    parser.add_argument('--save_images', action='store_true', help='是否保存去模糊后的图像')
    parser.add_argument('--output_dir', type=str, default='./output_deblur', help='保存去模糊图像的目录')
    parser.add_argument('--output_log', type=str, default='./eval_results.txt', help='保存评估指标的文件路径')

    args = parser.parse_args()
    main(args)