import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import joblib


import clip
from tqdm.auto import tqdm


def get_text_augs():
    # 读取每个文本描述对应的近义改写，供文本增强使用。
    fpath = os.path.join(os.path.dirname(__file__), "datasets/paraphrases_dict.pkl")
    paraphrases_dict = joblib.load(fpath)
    augs = {}
    for text, paraphrases in tqdm(paraphrases_dict.items()):
        paraphrases = paraphrases.split(",")
        text_augs = []
        for p in paraphrases:
            p = p.strip()
            if "\n" in p:
                # 某些条目里会混入多行文本，这里只保留最后一行有效句子。
                p = p.split("\n")[-1]
            text_augs.append(p)
        augs[text] = text_augs
    return augs


class ClipEmbed:
    def __init__(self, device):
        # 用 CLIP 文本编码器把自然语言条件转成 embedding。
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model = self.model.eval()
        self.device = device

    def embed(self, text):
        with torch.inference_mode():
            text = clip.tokenize(text).to(self.device)
            # encode_text 输出 batch 维，这里只取单条文本的 embedding。
            text_emb = self.model.encode_text(text)[0].cpu()
        return text_emb


class TextEmbedder:
    def __init__(self):
        self.loaded = None

    def load_processed(self, data_path):
        self.loaded = torch.load(data_path)

    def save_processed(self, all_texts, path):
        assert not os.path.exists(path)
        text_embedder = ClipEmbed(device="cuda")
        all_texts = list(set(all_texts))

        # 先把所有唯一文本都编码成 CLIP embedding，并记录文本到索引的映射。
        idx_mapping = {}
        text_embeddings = []
        for i, text in tqdm(enumerate(all_texts)):
            idx_mapping[text] = i
            text_embeddings.append(text_embedder.embed(text))
        text_embeddings = torch.stack(text_embeddings)

        # 再对 embedding 做 PCA，后续可以只保留前 num_pca 个主成分压缩条件维度。
        data = text_embeddings.float().numpy()
        mean = np.mean(data, axis=0)  # Compute mean vector
        centered_data = data - mean
        U, S, Vt = np.linalg.svd(centered_data, full_matrices=False)
        components = Vt  # Store all components
        components = torch.from_numpy(components).float()
        mean = torch.from_numpy(mean).float()

        # 保存原始 embedding、PCA 主成分和均值，后面可编码也可反解码。
        torch.save(
            {
                "idx_mapping": idx_mapping,
                "embs": text_embeddings,
                "pca_components": components,
                "mean": mean,
            },
            path,
        )

    def embed(self, *, text=None, emb=None, num_pca=None):

        assert (text is None) ^ (emb is None)

        if emb is None:
            # 如果传入的是文本，就先查索引再取预先缓存好的 embedding。
            emb_idx = self.loaded["idx_mapping"][text]
            emb = self.loaded["embs"][emb_idx].float()

        if num_pca is not None:
            # 可选：仅保留前 num_pca 个主成分，降低条件向量维度。
            emb = self.encode_pca(emb, num_pca)

        return emb

    def encode_pca(self, emb, num_pca):
        # PCA 编码：先中心化，再投影到前几个主成分上。
        emb = emb - self.loaded["mean"]
        emb = self.loaded["pca_components"][:num_pca] @ emb
        return emb

    def decode_pca(self, emb):
        # PCA 解码：从低维主成分空间重建回原 embedding 空间。
        num_pca = emb.shape[0]
        emb = self.loaded["pca_components"][:num_pca].T @ emb
        emb = emb + self.loaded["mean"]
        return emb


def download_data(fpath):
    # 课程提供的数据文件如果本地不存在，就自动下载。
    if not os.path.exists(fpath):
        print(f"Downloading...{fpath}")
        import urllib.request
        fname = os.path.basename(fpath)
        url = f"http://cs231n.stanford.edu/2025/storage/a3/{fname}"
        urllib.request.urlretrieve(url, fpath)
        print("Download complete.")
    else:
        fname = os.path.basename(fpath)
        print(f"{fname} already downloaded.")


class EmojiDataset(Dataset):
    def __init__(
        self,
        image_size,
        data_path="data/emoji_data.npz",
        text_emb_path="data/text_embeddings.pt",
        num_text_emb_pca=None,
    ):
        # 这里强制使用 cs231n/datasets 下的课程数据路径。
        data_path = os.path.join(os.path.dirname(__file__), "datasets/emoji_data.npz")
        text_emb_path = os.path.join(os.path.dirname(__file__), "datasets/text_embeddings.pt")
        download_data(data_path)
        download_data(text_emb_path)

        self.load_augs = False
        if self.load_augs:
            print("LOADING AUGS")
            self.text_augs = get_text_augs()
            text_emb_path = "data/text_embeddings_augs.pt"

        loaded = np.load(data_path, allow_pickle=True)
        # npz 每个 key 存一条样本记录，item() 后得到包含 images / texts 的字典。
        self.data = [loaded[key].item() for key in loaded]

        if self.load_augs:
            for d in self.data:
                texts = []
                for t in d["texts"]:
                    texts.extend(self.text_augs[t])
                texts = d["texts"] + texts
                d["texts"] = texts

        self.transform = T.Compose(
            [T.Resize(image_size), T.CenterCrop(image_size), T.ToTensor()]
        )
        self.num_text_emb_pca = num_text_emb_pca
        self.text_embedder = TextEmbedder()
        self.text_embedder.load_processed(text_emb_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        imgs = self.data[idx]["images"]
        texts = self.data[idx]["texts"]

        # 同一个 emoji 概念可能有多张图片，这里随机挑一张做训练样本。
        img_idx = np.random.choice(len(imgs))
        img = imgs[img_idx]

        # 图像转成 tensor，范围会变到 [0, 1]。
        img = Image.fromarray(img)
        img = self.transform(img)

        # 同理，文本描述也随机挑一个，增强“图像 <-> 文本条件”的多样性。
        text = np.random.choice(texts)
        text_emb = self.text_embedder.embed(text=text, num_pca=self.num_text_emb_pca)
        model_kwargs = {"text_emb": text_emb, "text": text}
        return img, model_kwargs

    def random_model_kwargs(self, n):

        # 采样阶段只需要条件信息，不需要把图像也返回出去。
        idxs = np.random.choice(len(self), n)
        samples = [self.__getitem__(idx) for idx in idxs]
        imgs, model_kwargs = torch.utils.data.default_collate(samples)

        return model_kwargs

    def embed_new_text(self, text, clip_embed):
        # 对训练集里没有见过的新文本，也能动态编码成条件向量。
        text_emb = clip_embed.embed(text).float().cpu()
        if self.num_text_emb_pca is not None:
            text_emb = self.text_embedder.encode_pca(text_emb, self.num_text_emb_pca)
        return text_emb
