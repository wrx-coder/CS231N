import cv2
import clip
import numpy as np
import tensorflow_datasets as tfds
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T


def get_similarity_no_loop(text_features, image_features):
    """
    Computes the pairwise cosine similarity between text and image feature vectors.
    """
    # 先做 L2 normalize，这样点积就等价于 cosine similarity。
    text_norm = text_features / text_features.norm(dim=1, keepdim=True)
    image_norm = image_features / image_features.norm(dim=1, keepdim=True)
    # 相似度矩阵形状是 (num_text, num_image)。
    similarity = text_norm @ image_norm.t()
    return similarity


@torch.no_grad()
def clip_zero_shot_classifier(clip_model, clip_preprocess, images, class_texts, device):
    pred_classes = []

    # 把类别文本 prompt 编码成文本特征库。
    text_tokens = clip.tokenize(class_texts).to(device)
    text_features = clip_model.encode_text(text_tokens).float()
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # 把待分类图片编码成图像特征。
    image_inputs = torch.stack(
        [clip_preprocess(Image.fromarray(image)) for image in images]
    ).to(device)
    image_features = clip_model.encode_image(image_inputs).float()
    image_features = image_features / image_features.norm(dim=1, keepdim=True)

    # 每张图选和它最相似的文本类别，就是 zero-shot 分类结果。
    similarities = get_similarity_no_loop(text_features, image_features)
    pred_indices = similarities.argmax(dim=0).tolist()
    pred_classes = [class_texts[idx] for idx in pred_indices]
    return pred_classes


class CLIPImageRetriever:
    """
    A simple image retrieval system using CLIP.
    """

    @torch.no_grad()
    def __init__(self, clip_model, clip_preprocess, images, device):
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.images = images
        self.device = device

        # 初始化时一次性把图库图片全部编码好，后面检索只需要编码 query 文本。
        image_inputs = torch.stack(
            [clip_preprocess(Image.fromarray(image)) for image in images]
        ).to(device)
        image_features = clip_model.encode_image(image_inputs).float()
        self.image_features = image_features / image_features.norm(dim=1, keepdim=True)

    @torch.no_grad()
    def retrieve(self, query: str, k: int = 2):
        # 检索流程：文本 query 编码 -> 与图库特征算相似度 -> 取 top-k。
        text_tokens = clip.tokenize([query]).to(self.device)
        text_features = self.clip_model.encode_text(text_tokens).float()
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        similarities = text_features @ self.image_features.t()
        top_indices = similarities.squeeze(0).topk(k=min(k, len(self.images))).indices.tolist()
        return top_indices


class DavisDataset:
    def __init__(self):
        # DAVIS 是视频分割数据集，这里读取验证集做 DINO patch-level segmentation。
        self.davis = tfds.load("davis/480p", split="validation", as_supervised=False)
        self.img_tsfm = T.Compose(
            [
                T.Resize((480, 480)),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def get_sample(self, index):
        assert index < len(self.davis)
        # tfds 的这个 split 需要顺序迭代到指定样本。
        ds_iter = iter(tfds.as_numpy(self.davis))
        for _ in range(index + 1):
            video = next(ds_iter)
        frames, masks = video["video"]["frames"], video["video"]["segmentations"]
        print(f"video {video['metadata']['video_name'].decode()}  {len(frames)} frames")
        return frames, masks

    def process_frames(self, frames, dino_model, device):
        res = []
        for f in frames:
            f = self.img_tsfm(Image.fromarray(f))[None].to(device)
            with torch.no_grad():
                # DINO 输出包含 cls token 和 patch token；这里丢掉 cls，只保留每个 patch 的特征。
                tok = dino_model.get_intermediate_layers(f, n=1)[0]
            res.append(tok[0, 1:])

        # 输出形状通常是 (num_frames, num_patches, feature_dim)。
        res = torch.stack(res)
        return res

    def process_masks(self, masks, device):
        res = []
        for m in masks:
            # 把像素级 mask 缩到和 patch 网格一致的 60x60，再展平成 patch 标签。
            m = cv2.resize(m, (60, 60), cv2.INTER_NEAREST)
            res.append(torch.from_numpy(m).long().flatten(-2, -1))
        res = torch.stack(res).to(device)
        return res

    def mask_frame_overlay(self, processed_mask, frame):
        H, W = frame.shape[:2]
        mask = processed_mask.detach().cpu().numpy()
        mask = mask.reshape((60, 60))
        # 把 patch 级 mask 再放回原图大小，方便可视化。
        mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        overlay = create_segmentation_overlay(mask, frame.copy())
        return overlay


def create_segmentation_overlay(segmentation_mask, image, alpha=0.5):
    assert segmentation_mask.shape[:2] == image.shape[:2], "Segmentation and image size mismatch"
    assert image.dtype == np.uint8, "Image must be of type uint8"

    def generate_colormap(n):
        # 固定随机种子，保证每次可视化类别颜色一致。
        np.random.seed(42)
        return np.random.randint(0, 256, size=(n, 3), dtype=np.uint8)

    colormap = generate_colormap(10)
    seg_color = colormap[segmentation_mask]
    overlay = cv2.addWeighted(image, 1 - alpha, seg_color, alpha, 0)
    return overlay


def compute_iou(pred, gt, num_classes):
    iou = 0
    for ci in range(num_classes):
        # IoU = 交集 / 并集，最后对所有类别取平均。
        p = pred == ci
        g = gt == ci
        iou += (p & g).sum() / ((p | g).sum() + 1e-8)
    return iou / num_classes


class DINOSegmentation:
    def __init__(self, device, num_classes: int, inp_dim: int = 384):
        self.device = device
        # 这里把 DINO 提取的每个 patch 特征当作输入，用线性层做逐 patch 分类。
        self.model = nn.Linear(inp_dim, num_classes).to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-2, weight_decay=1e-4
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self, X_train, Y_train, num_iters=500):
        self.model.train()
        X_train = X_train.to(self.device)
        Y_train = Y_train.to(self.device)

        for _ in range(num_iters):
            # logits 形状一般是 (num_patches_total, num_classes)。
            logits = self.model(X_train)
            loss = self.loss_fn(logits, Y_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def inference(self, X_test):
        self.model.eval()
        logits = self.model(X_test.to(self.device))
        # 每个 patch 取最大 logit 对应的类别。
        pred_classes = logits.argmax(dim=1)
        return pred_classes
