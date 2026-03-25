import math
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import save_image
from tqdm.auto import tqdm


def cycle(dl):
    # 把有限的 DataLoader 包装成无限迭代器，训练时就不用手动重启 epoch。
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        device,
        *,
        train_batch_size=256,
        train_lr=1e-3,
        weight_decay=0.0,
        train_num_steps=100000,
        adam_betas=(0.9, 0.99),
        sample_every=1000,
        save_every=10000,
        results_folder=None,
    ):
        super().__init__()

        assert results_folder is not None, "must specify results folder"
        self.diffusion_model = diffusion_model

        self.device = device
        # 这里固定每次可视化采样 25 张，方便排成 5x5 网格。
        self.num_samples = 25
        self.save_every = save_every
        self.sample_every = sample_every
        self.batch_size = train_batch_size
        self.train_num_steps = train_num_steps

        # 构建训练 DataLoader，dataset 会返回图像和对应的条件信息。
        self.ds = dataset
        dl = DataLoader(
            self.ds,
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=0,
        )
        self.dl = cycle(dl)

        # DDPM 常见选择是 Adam / AdamW，这里使用 Adam。
        self.opt = Adam(
            diffusion_model.parameters(),
            lr=train_lr,
            betas=adam_betas,
            weight_decay=weight_decay,
        )

        self.results_folder = results_folder
        os.makedirs(self.results_folder, exist_ok=True)

        # step 记录已经训练了多少次参数更新。
        self.step = 0

    def save(self, milestone):
        # checkpoint 同时保存模型参数、优化器状态和当前 step。
        ckpt_path = os.path.join(self.results_folder, f"model-{milestone}.pt")
        print(f"saving model to {ckpt_path}.")
        data = {
            "step": self.step,
            "model": self.diffusion_model.state_dict(),
            "opt": self.opt.state_dict(),
        }

        torch.save(data, ckpt_path)

    def load(self, milestone):

        ckpt_path = os.path.join(self.results_folder, f"model-{milestone}.pt")
        print(f"loading model from {ckpt_path}.")
        data = torch.load(ckpt_path, map_location=self.device, weights_only=True)

        self.diffusion_model.load_state_dict(data["model"])
        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])

        # 载入后把优化器内部缓存张量也迁移到目标设备。
        device = self.device
        self.diffusion_model.to(device)
        for state in self.opt.state.values():
            if isinstance(state, torch.Tensor):
                state.data = state.data.to(device)
            elif isinstance(state, dict):
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

    def download_pretrained(self):
        # 如果本地没有预训练权重，就从课程服务器下载。
        ckpt_path = os.path.join(self.results_folder, f"model-70000.pt")
        if not os.path.exists(ckpt_path):
            print(f"Downloading...{ckpt_path}")
            import urllib.request
            fname = os.path.basename(ckpt_path)
            url = f"http://cs231n.stanford.edu/2025/storage/a3/{fname}"
            urllib.request.urlretrieve(url, ckpt_path)
            print("Download complete.")
        else:
            print(f"Pretrained model already downloaded.")

    def train(self):
        device = self.device
        self.diffusion_model.to(device)

        all_losses = []

        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:

            while self.step < self.train_num_steps:

                data, model_kwargs = next(self.dl)
                data = data.to(device)
                # text_emb 是条件扩散里给 U-Net 的文本条件向量。
                model_kwargs["text_emb"] = model_kwargs["text_emb"].to(device)

                self.opt.zero_grad()
                loss = self.diffusion_model.p_losses(data, model_kwargs=model_kwargs)
                loss.backward()
                # 梯度裁剪防止训练后期出现梯度爆炸。
                torch.nn.utils.clip_grad_norm_(self.diffusion_model.parameters(), 1.0)
                self.opt.step()

                pbar.set_description(f"loss: {loss.item():.4f}")
                all_losses.append(loss.item())

                self.step += 1

                if self.step % self.save_every == 0:
                    self.save(self.step)

                if self.step % self.sample_every == 0:
                    self.diffusion_model.eval()

                    with torch.no_grad():
                        # 随机抽一些文本条件，生成对应样本用于肉眼检查训练效果。
                        model_kwargs = self.ds.random_model_kwargs(self.num_samples)
                        model_kwargs["text_emb"] = model_kwargs["text_emb"].to(device)

                        all_images = self.diffusion_model.sample(
                            batch_size=self.num_samples, model_kwargs=model_kwargs
                        )

                    save_image(
                        all_images,
                        os.path.join(self.results_folder, f"sample-{self.step}.png"),
                        nrow=int(math.sqrt(self.num_samples)),
                    )

                pbar.update(1)

        return all_losses
