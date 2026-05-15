import torch
import numpy as np
from tqdm import tqdm
from diffusion import GaussianDiffusion

class FastDiffusion(GaussianDiffusion):
    """
    继承自预测噪声的 GaussianDiffusion，添加 DDIM 与 DPM-Solver 快速采样。
    """
    def _get_subsequence(self, steps):
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
        shape = (b, self.channels, h, w)

        time_steps = self._get_subsequence(steps).to(device)
        time_prev = torch.cat([time_steps[1:], torch.tensor([0], device=device)])

        img = torch.randn(shape, device=device)

        for i, (t, t_prev) in enumerate(zip(time_steps, time_prev)):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            t_prev_batch = torch.full((b,), t_prev, device=device, dtype=torch.long)

            # 网络预测噪声 ε
            pred_noise = self.denoise_fn(torch.cat([condition, img], dim=1), t_batch)

            # 由预测噪声重建 x0
            x0_pred = self.predict_start_from_noise(img, t_batch, pred_noise)
            x0_pred.clamp_(0., 1.)  # 重建的 x0 限制在 [0, 1] 范围内

            if method == 'ddim':
                img = self._ddim_step(img, x0_pred, t, t_prev, eta)
            elif method == 'dpm_solver_2s':
                img = self._dpm_solver_2s_step(img, x0_pred, condition, t_batch, t_prev_batch, i)
            else:
                raise ValueError("method must be 'ddim' or 'dpm_solver_2s'")

        return img.clamp(0., 1.)

    def _ddim_step(self, x_t, x0_pred, t, t_prev, eta=0.0):
        """
        DDIM 更新（基于重建的 x0_pred）。
        """
        alpha_bar_t = self.alphas_cumprod[t]
        alpha_bar_t_prev = self.alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0, device=x_t.device)

        sigma = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * \
                torch.sqrt(1 - alpha_bar_t / alpha_bar_t_prev)

        pred_dir = torch.sqrt(alpha_bar_t_prev) * x0_pred + \
                   torch.sqrt(1 - alpha_bar_t_prev - sigma**2) * \
                   (x_t - torch.sqrt(alpha_bar_t) * x0_pred) / torch.sqrt(1 - alpha_bar_t)

        if eta > 0:
            noise = torch.randn_like(x_t)
            return pred_dir + sigma * noise
        else:
            return pred_dir

    def _dpm_solver_2s_step(self, x_s, x0_pred, condition, t_s, t_prev, step_idx):
        """
        DPM-Solver++(2S) 单步更新（基于重建的 x0_pred）。
        """
        if not hasattr(self, '_dpm_cache'):
            self._dpm_cache = {'prev_x': None, 'prev_t': None, 'prev_x0': None}

        alpha_bar_s = self.alphas_cumprod[t_s]
        alpha_bar_t = self.alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0, device=x_s.device)
        lambda_s = torch.log(alpha_bar_s / (1 - alpha_bar_s))
        lambda_t = torch.log(alpha_bar_t / (1 - alpha_bar_t))
        h = lambda_t - lambda_s

        x_t_1st = torch.sqrt(alpha_bar_t / alpha_bar_s) * x_s - \
                  torch.sqrt(alpha_bar_t) * (torch.exp(-h) - 1) * x0_pred

        if self._dpm_cache['prev_x'] is not None:
            x_s_prev = self._dpm_cache['prev_x']
            t_prev_step = self._dpm_cache['prev_t']
            x0_prev = self._dpm_cache['prev_x0']

            lambda_s_prev = torch.log(self.alphas_cumprod[t_prev_step] / (1 - self.alphas_cumprod[t_prev_step]))
            h_prev = lambda_s - lambda_s_prev

            D1_s = (x0_prev - x0_pred) / h_prev
            correction = -0.5 * torch.sqrt(alpha_bar_t) * (torch.exp(-h) - 1) * h * D1_s
            x_t = x_t_1st + correction
        else:
            x_t = x_t_1st

        self._dpm_cache['prev_x'] = x_s
        self._dpm_cache['prev_t'] = t_s
        self._dpm_cache['prev_x0'] = x0_pred

        return x_t

    def clear_dpm_cache(self):
        self._dpm_cache = {'prev_x': None, 'prev_t': None, 'prev_x0': None}