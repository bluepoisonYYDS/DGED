import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from diffusion import GaussianDiffusion
from unet import UNet

class FastDiffusion(GaussianDiffusion):
    """
    继承自已修正的 GaussianDiffusion（denoise_fn 直接预测 x0），
    添加 DDIM 与 DPM-Solver 的高效采样方法。
    """
    def _get_subsequence(self, steps):
        """生成采样子序列（DDIM 跳跃步思想）"""
        # 从 T-1 到 0 均匀取 steps 个时间点（含两端）
        indices = np.linspace(0, self.num_timesteps - 1, steps).astype(np.int64)[::-1]
        return torch.from_numpy(indices.copy()).long()

    @torch.no_grad()
    def fast_sample_loop(self, condition, steps=50, method='ddim', eta=0.0):
        """
        条件生成事件体素（快速采样）。
        Args:
            condition: (b, 6, h, w) 模糊图 + 细节引导 P
            steps: 采样步数（远小于 T，如 20~50）
            method: 'ddim' 或 'dpm_solver_2s'
            eta: DDIM 的随机性系数，0 为确定性，1 为原始 DDPM
        Returns:
            生成的 6 通道事件体素
        """
        device = self.betas.device
        b, c, h, w = condition.shape
        shape = (b, self.channels, h, w)       # 事件体素形状

        # 1. 构建跳跃时间子序列（DDIM 的核心）
        time_steps = self._get_subsequence(steps).to(device)    # [τ_s, τ_{s-1}, ..., τ_0]
        time_prev = torch.cat([time_steps[1:], torch.tensor([0], device=device)])  # 前一时刻

        # 2. 从纯噪声开始
        img = torch.randn(shape, device=device)

        for i, (t, t_prev) in enumerate(zip(time_steps, time_prev)):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            t_prev_batch = torch.full((b,), t_prev, device=device, dtype=torch.long)

            # 用网络预测 x0（利用条件）
            x0_pred = self.denoise_fn(torch.cat([condition, img], dim=1), t_batch)

            # 根据方法选择更新公式
            if method == 'ddim':
                img = self._ddim_step(img, x0_pred, t, t_prev, eta)
            elif method == 'dpm_solver_2s':
                img = self._dpm_solver_2s_step(img, x0_pred, condition, t_batch, t_prev_batch, i)
            else:
                raise ValueError("method must be 'ddim' or 'dpm_solver_2s'")

        return img

    # ---------- DDIM 单步更新 ----------
    def _ddim_step(self, x_t, x0_pred, t, t_prev, eta=0.0):
        """
        DDIM 更新（基于预测的 x0 与指定时间步 t -> t_prev）。
        公式：
            σ_t = η * sqrt( (1-α_bar_{t-1})/(1-α_bar_t) ) * sqrt(1 - α_bar_t/α_bar_{t-1} )
            方向指向 x_t  = sqrt(α_bar_{t-1}) * x0_pred + sqrt(1-α_bar_{t-1} - σ_t^2) * (x_t - sqrt(α_bar_t) x0_pred) / sqrt(1-α_bar_t)
            x_{t-1} = 方向 + σ_t * ε
        """
        alpha_bar_t = self.alphas_cumprod[t]
        alpha_bar_t_prev = self.alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0, device=x_t.device)

        sigma = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * \
                torch.sqrt(1 - alpha_bar_t / alpha_bar_t_prev)

        # 预测的“方向”分量
        pred_dir = torch.sqrt(alpha_bar_t_prev) * x0_pred + \
                   torch.sqrt(1 - alpha_bar_t_prev - sigma**2) * \
                   (x_t - torch.sqrt(alpha_bar_t) * x0_pred) / torch.sqrt(1 - alpha_bar_t)

        if eta > 0:
            noise = torch.randn_like(x_t)
            return pred_dir + sigma * noise
        else:
            return pred_dir

    # ---------- DPM-Solver 二阶更新 ----------
    def _dpm_solver_2s_step(self, x_s, x0_pred, condition, t_s, t_prev, step_idx):
        """
        DPM-Solver++(2S) 的单步更新（预测 x0 版本）。
        需要保留上一步的模型输出，因此外部维护一个缓存。

        原理：
            将扩散 ODE 转换为关于 x0 预测的形式，利用指数积分器得到二阶更新。
        这里实现简化版：第一次调用时只做一阶预测，后续步结合前一步的输出来做二阶校正。
        """
        if not hasattr(self, '_dpm_cache'):
            self._dpm_cache = {'prev_x': None, 'prev_t': None, 'prev_x0': None}

        # 计算 lambda_t = log(α_t / (1-α_t)) (SNR 的 log)
        alpha_bar_s = self.alphas_cumprod[t_s]
        alpha_bar_t = self.alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0, device=x_s.device)
        lambda_s = torch.log(alpha_bar_s / (1 - alpha_bar_s))
        lambda_t = torch.log(alpha_bar_t / (1 - alpha_bar_t))
        h = lambda_t - lambda_s

        # 一阶更新（首次步或作为基础）
        x_t_1st = torch.sqrt(alpha_bar_t / alpha_bar_s) * x_s - \
                  torch.sqrt(alpha_bar_t) * (torch.exp(-h) - 1) * x0_pred

        # 如果有前一步的缓存，进行二阶校正
        if self._dpm_cache['prev_x'] is not None:
            x_s_prev = self._dpm_cache['prev_x']
            t_prev_step = self._dpm_cache['prev_t']
            # 取出前一时刻的 x0 预测，需要再算一次（或者缓存 x0_prev）
            x0_prev = self._dpm_cache['prev_x0']

            lambda_s_prev = torch.log(self.alphas_cumprod[t_prev_step] / (1 - self.alphas_cumprod[t_prev_step]))
            h_prev = lambda_s - lambda_s_prev

            D1_s = (x0_prev - x0_pred) / h_prev  # 导数近似
            # 二阶校正项
            correction = -0.5 * torch.sqrt(alpha_bar_t) * (torch.exp(-h) - 1) * h * D1_s
            x_t = x_t_1st + correction
        else:
            x_t = x_t_1st

        # 更新缓存，用于下一步
        self._dpm_cache['prev_x'] = x_s
        self._dpm_cache['prev_t'] = t_s
        self._dpm_cache['prev_x0'] = x0_pred

        return x_t

    def clear_dpm_cache(self):
        self._dpm_cache = {'prev_x': None, 'prev_t': None, 'prev_x0': None}