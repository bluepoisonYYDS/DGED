from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

def evaluate_psnr_ssim(model, dataloader, device):
    """
    在给定数据集上计算平均 PSNR 和 SSIM。
    注意：图像值域需为 [0, 1]，若模型输出未钳位，请手动 clamp。
    """
    model.eval()
    psnr_list = []
    ssim_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Test'):
            blur = batch['blur'].to(device)
            events = batch['events'].to(device)
            sharp = batch['sharp'].to(device)

            pred = model(blur, events)

            # 确保值在 [0, 1] 范围内
            pred = torch.clamp(pred, 0.0, 1.0)

            # 转换为 numpy 数组 (H, W, C)
            pred_np = pred.squeeze(0).cpu().permute(1, 2, 0).numpy()
            sharp_np = sharp.squeeze(0).cpu().permute(1, 2, 0).numpy()

            # 计算 PSNR（数据范围设为 1.0）
            psnr = peak_signal_noise_ratio(sharp_np, pred_np, data_range=1.0)

            # 计算 SSIM，多通道设置 channel_axis=-1
            ssim = structural_similarity(sharp_np, pred_np, channel_axis=-1, data_range=1.0)

            psnr_list.append(psnr)
            ssim_list.append(ssim)

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    return avg_psnr, avg_ssim