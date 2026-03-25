import math

import torch
import torch.nn as nn
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        objective="pred_noise",
        beta_schedule="sigmoid",
    ):
        super().__init__()

        # model 通常是一个 U-Net，输入 x_t 和时间步 t，输出噪声或 x_0 的估计。
        self.model = model
        self.channels = 3
        self.image_size = image_size
        self.objective = objective
        assert objective in {
            "pred_noise",
            "pred_x_start",
        }, "objective must be either pred_noise (predict noise) or pred_x_start (predict image start)"

        # register_buffer 用来存扩散过程中的常数表，让它们跟随设备移动但不参与训练。
        register_buffer = lambda name, val: self.register_buffer(name, val.float())

        # betas 决定每个时间步加多少噪声；不同 schedule 会影响训练和采样稳定性。
        betas = get_beta_schedule(beta_schedule, timesteps)
        self.num_timesteps = int(betas.shape[0])
        alphas = 1.0 - betas
        # alphas_cumprod = \bar{alpha}_t，表示从 0 一路扩散到 t 还保留多少原图信息。
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        register_buffer("betas", betas)
        register_buffer("alphas", alphas)
        register_buffer("alphas_cumprod", alphas_cumprod)

        # 这两个量会直接出现在 q(x_t | x_0) 的闭式表达里。
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

        # alphas_cumprod_prev 是 \bar{alpha}_{t-1}，t=0 时补 1。
        alphas_cumprod_prev = nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        # posterior_mean_coef1 / coef2 对应 q(x_{t-1} | x_t, x_0) 的均值公式里的两个系数。
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )
        # posterior_std 是反向一步采样时要乘的标准差。
        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_std = torch.sqrt(posterior_var.clamp(min=1e-20))
        register_buffer("posterior_std", posterior_std)

        # snr = signal-to-noise ratio；当 objective 是预测 x_0 时，用它给不同时间步加权。
        snr = alphas_cumprod / (1 - alphas_cumprod)
        loss_weight = torch.ones_like(snr) if objective == "pred_noise" else snr
        register_buffer("loss_weight", loss_weight)

    def normalize(self, img):
        # 把像素从 [0, 1] 映射到 [-1, 1]，这是 diffusion 模型里更常见的数值范围。
        return img * 2 - 1

    def unnormalize(self, img):
        # 采样结束后再映射回可视化常用的 [0, 1]。
        return (img + 1) * 0.5

    def predict_start_from_noise(self, x_t, t, noise):
        # 根据 q(x_t | x_0) 的公式，把噪声估计反推出原始图像 x_0。
        sqrt_alphas_cumprod = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        x_start = (x_t - sqrt_one_minus_alphas_cumprod * noise) / sqrt_alphas_cumprod
        return x_start

    def predict_noise_from_start(self, x_t, t, x_start):
        # 反过来，如果模型预测的是 x_0，就可以把它换算成噪声 epsilon。
        sqrt_alphas_cumprod = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        pred_noise = (x_t - sqrt_alphas_cumprod * x_start) / sqrt_one_minus_alphas_cumprod
        return pred_noise

    def q_posterior(self, x_start, x_t, t):
        # q(x_{t-1} | x_t, x_0) 是高斯分布，这里返回它的均值和标准差。
        c1 = extract(self.posterior_mean_coef1, t, x_t.shape)
        c2 = extract(self.posterior_mean_coef2, t, x_t.shape)
        posterior_mean = c1 * x_start + c2 * x_t
        posterior_std = extract(self.posterior_std, t, x_t.shape)
        return posterior_mean, posterior_std

    @torch.no_grad()
    def p_sample(self, x_t, t: int, model_kwargs={}):
        # 把标量时间步复制成 batch 维向量，方便一次并行处理一整批样本。
        t = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)

        # 模型输出要么是噪声估计 epsilon_theta，要么是 x_0 的估计。
        model_out = self.model(x_t, t, model_kwargs=model_kwargs)
        if self.objective == "pred_noise":
            pred_noise = model_out
            x_start = self.predict_start_from_noise(x_t, t, pred_noise)
        else:
            x_start = model_out

        # 对预测的 x_0 做裁剪，避免数值飘得太远，提升采样稳定性。
        x_start = x_start.clamp(-1.0, 1.0)
        posterior_mean, posterior_std = self.q_posterior(x_start, x_t, t)

        # t=0 时已经走到最终结果，不再加随机噪声。
        if int(t[0].item()) == 0:
            return posterior_mean
        # 其余时间步按 posterior 的均值和标准差采样得到 x_{t-1}。
        return posterior_mean + posterior_std * torch.randn_like(x_t)

    @torch.no_grad()
    def sample(self, batch_size=16, return_all_timesteps=False, model_kwargs={}):
        # 采样从纯高斯噪声 x_T 开始，一步步去噪回到图像空间。
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        img = torch.randn(shape, device=self.betas.device)
        imgs = [img]

        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            # 每一轮把 x_t 变成 x_{t-1}。
            img = self.p_sample(img, t, model_kwargs=model_kwargs)
            imgs.append(img)

        res = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        res = self.unnormalize(res)
        return res

    def q_sample(self, x_start, t, noise):
        # 前向扩散闭式公式：x_t = sqrt(\bar{alpha}_t) x_0 + sqrt(1-\bar{alpha}_t) epsilon。
        sqrt_alphas_cumprod = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        x_t = sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise
        return x_t

    def p_losses(self, x_start, model_kwargs={}):
        b, nts = x_start.shape[0], self.num_timesteps
        # 训练时对每张图随机采一个时间步 t，这样一个模型就能学会所有去噪阶段。
        t = torch.randint(0, nts, (b,), device=x_start.device).long()
        x_start = self.normalize(x_start)
        noise = torch.randn_like(x_start)
        # pred_noise 目标是还原 epsilon；pred_x_start 目标是直接还原原图。
        target = noise if self.objective == "pred_noise" else x_start
        loss_weight = extract(self.loss_weight, t, target.shape)

        x_t = self.q_sample(x_start, t, noise)
        model_out = self.model(x_t, t, model_kwargs=model_kwargs)
        # 这里是逐像素 MSE，再乘时间步权重。
        loss = ((model_out - target) ** 2 * loss_weight).mean()
        return loss


def extract(a, t, x_shape):
    # 从长度为 T 的系数表 a 中，按 batch 里的时间步 t 取出对应系数，并 reshape 成可广播形状。
    b, *_ = t.shape
    out = a.gather(-1, t)
    out = out.reshape(b, *((1,) * (len(x_shape) - 1)))
    return out


def linear_beta_schedule(timesteps):
    # 线性噪声日程：beta 从小到大均匀增长。
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    # cosine schedule 会让前期保留更多信号，实践中常比线性 schedule 更平滑。
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    # sigmoid schedule 在前中后期的噪声增长速度更柔和。
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def get_beta_schedule(beta_schedule, timesteps):
    # 根据字符串配置选择具体的 beta 生成函数。
    if beta_schedule == "linear":
        beta_schedule_fn = linear_beta_schedule
    elif beta_schedule == "cosine":
        beta_schedule_fn = cosine_beta_schedule
    elif beta_schedule == "sigmoid":
        beta_schedule_fn = sigmoid_beta_schedule
    else:
        raise ValueError(f"unknown beta schedule {beta_schedule}")

    return beta_schedule_fn(timesteps)
