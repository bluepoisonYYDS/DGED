import os
import argparse
import random
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------
# 原始模型模块（请确保路径正确）
# ------------------------------------------------------------
from model_large import EventImageDeblurNet
from DV import HeavyE_CIR
from eval import evaluate_psnr_ssim          # 原训练中的评估接口


# ------------------------------------------------------------
# 联合模型定义（与训练代码完全一致）
# ------------------------------------------------------------
class JointDeblurModel(nn.Module):
    def __init__(self, deblur_net_args):
        super().__init__()
        self.densifier = HeavyE_CIR(event_bins=6, hidden_dim=128, img_feat_dim=32)
        deblur_net_args['event_ch'] = 1
        self.deblur_net = EventImageDeblurNet(**deblur_net_args)

    def forward(self, blur_img, event_voxel):
        edge = self.densifier(event_voxel, blur_img)   # (B,1,H,W)
        #edge = torch.zeros_like(edge)
        return self.deblur_net(blur_img, edge)


# ------------------------------------------------------------
# 测试数据集（必须包含清晰图像用于计算指标）
# ------------------------------------------------------------
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
    return np.clip(voxel, 0, clip_val) / clip_val


class TestDataset(Dataset):
    """测试数据集：读取模糊图像、事件体素和清晰图像"""
    def __init__(self, root, split='test', crop_size=None):
        super().__init__()
        self.root = os.path.join(root, split)
        self.crop_size = crop_size
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
                self.samples.append((blur_path, sharp_path, voxel_path, name))
            else:
                print(f"警告：样本 {name} 文件缺失，已跳过")
        print(f"{split} 集共 {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def _center_crop(self, img_tensor, crop_h, crop_w):
        """对 (C, H, W) 张量进行中心裁剪"""
        H, W = img_tensor.shape[-2], img_tensor.shape[-1]
        top = (H - crop_h) // 2
        left = (W - crop_w) // 2
        return img_tensor[..., top:top+crop_h, left:left+crop_w]

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
        blur_path, sharp_path, voxel_path, name = self.samples[idx]
        blur_np = self._load_image(blur_path) / 255.0
        sharp_np = self._load_image(sharp_path) / 255.0
        voxel_np = normalize_voxel(self._load_voxel(voxel_path))

        blur = torch.from_numpy(blur_np).permute(2, 0, 1)
        sharp = torch.from_numpy(sharp_np).permute(2, 0, 1)
        voxel = torch.from_numpy(voxel_np).permute(2, 0, 1)

        if self.crop_size is not None:
            crop_h, crop_w = self.crop_size, self.crop_size  # 正方形
            blur = self._center_crop(blur, crop_h, crop_w)
            sharp = self._center_crop(sharp, crop_h, crop_w)
            voxel = self._center_crop(voxel, crop_h, crop_w)

        return {'blur': blur, 'events': voxel, 'sharp': sharp, 'name': name}

# ------------------------------------------------------------
# 测试主函数
# ------------------------------------------------------------
def test(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ---------- 数据集 ----------
    crop = args.crop_size if args.crop_size > 0 else None
    dataset = TestDataset(args.data_root, split=args.split, crop_size=crop)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # ---------- 构建模型并加载权重 ----------
    deblur_args = dict(
        img_ch=3,
        base_ch=args.base_ch,
        window_size=args.window_size,
        drop_rate=args.drop_rate,
        attn_drop=args.attn_drop,
        drop_path_rate=args.drop_path_rate
    )
    model = JointDeblurModel(deblur_args).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载联合模型权重，epoch: {checkpoint.get('epoch', '未知')}, "
              f"最佳val loss: {checkpoint.get('best_val_loss', '未知')}")
    else:
        model.load_state_dict(checkpoint)
        print("加载纯 state_dict 权重")
    model.eval()

    # ---------- 输出目录 ----------
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, 'vis')
    if args.save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    # ---------- 第一步：使用原接口计算 PSNR / SSIM ----------
    print("正在计算 PSNR 和 SSIM ...")
    avg_psnr, avg_ssim = evaluate_psnr_ssim(model, loader, device)
    print(f"\n===== 测试集评估结果 =====")
    print(f"平均 PSNR: {avg_psnr:.4f} dB")
    print(f"平均 SSIM: {avg_ssim:.4f}")

    # 保存评估日志
    log_path = os.path.join(args.output_dir, 'test_metrics.txt')
    with open(log_path, 'w') as f:
        f.write(f"Average PSNR: {avg_psnr:.4f} dB\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
    print(f"评估日志已保存至 {log_path}")

    # ---------- 第二步：再次推理，保存所有生成图像和可视化样本 ----------
    # 确定需要可视化的 batch 索引（随机采样）
    total_batches = len(loader)
    num_vis = min(args.num_vis_batches, total_batches)
    if args.save_vis and num_vis > 0:
        rng = random.Random(42)  # 固定种子，保证可复现
        vis_indices = sorted(rng.sample(range(total_batches), num_vis))
        print(f"将在以下 batch 索引处进行可视化保存: {vis_indices}")
    else:
        vis_indices = set()

    print("保存去模糊图像和可视化结果 ...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc='Saving')):
            blur = batch['blur'].to(device)
            events = batch['events'].to(device)
            sharp = batch['sharp'].to(device)
            names = batch['name']

            pred = model(blur, events)  # (B,3,H,W)

            if args.save_vis and batch_idx in vis_indices:
                i = 0
                blur_img = (blur[i].cpu().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
                pred_img = (pred[i].cpu().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
                sharp_img = (sharp[i].cpu().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()

                #Image.fromarray(blur_img).save(os.path.join(vis_dir, f'batch{batch_idx:04d}_blur.png'))
                Image.fromarray(pred_img).save(os.path.join(vis_dir, f'batch{batch_idx:04d}_pred3.png'))
                #Image.fromarray(sharp_img).save(os.path.join(vis_dir, f'batch{batch_idx:04d}_sharp.png'))

    print(f"\n全部去模糊图像已保存至 {args.output_dir}")
    if args.save_vis:
        print(f"可视化拼接图已保存至 {vis_dir}")


# ------------------------------------------------------------
# 命令行参数
# ------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='联合事件去模糊模型测试（默认计算PSNR/SSIM）')
    parser.add_argument('--model_path', type=str, required=True,
                        help='联合模型权重路径 (如 best_model_joint.pth)')
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据根目录，应包含 test/(blur,sharp,voxel) 等子文件夹')
    parser.add_argument('--output_dir', type=str, default='./test_output',
                        help='去模糊图像及评测结果输出目录')
    parser.add_argument('--split', type=str, default='test',
                        help='数据集划分 (默认 test)')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')

    # 模型结构参数（必须与训练时一致）
    parser.add_argument('--base_ch', type=int, default=64)
    parser.add_argument('--window_size', type=int, default=8)
    parser.add_argument('--drop_rate', type=float, default=0.1)
    parser.add_argument('--attn_drop', type=float, default=0.05)
    parser.add_argument('--drop_path_rate', type=float, default=0.05)

    # 可视化
    parser.add_argument('--save_vis', action='store_true', default=True,
                        help='是否保存随机 batch 的可视化拼接图')
    parser.add_argument('--num_vis_batches', type=int, default=5,
                        help='需要可视化的随机 batch 数量')
    parser.add_argument('--crop_size', type=int, default=640,
                        help='中心正方形裁剪边长，默认640，设为0则不裁剪')

    args = parser.parse_args()
    test(args)

'''
python /root/autodl-tmp/Mymodel_Improved/test_large_model.py --model_path /root/autodl-tmp/Mymodel_Improved/model_large_ckpt/ckpt_joint_epoch140.pth --data_root /root/autodl-tmp/data --output_dir ./results  --num_vis_batches 10 --split test
'''