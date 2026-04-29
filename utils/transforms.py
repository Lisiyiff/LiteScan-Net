"""
专门为耕地变化检测重构的数据增强模块 (Numpy Flow Refactored)
特点：
1. 纯 Numpy 实现，移除中间环节的 Tensor 转换，大幅减少代码冗余。
2. 严格的几何一致性：img1, img2, label 共享同一组几何变换参数。
3. 严格的光谱独立性：img1, img2 的颜色变换独立进行。
4. 支持多光谱数据（不限于3通道）。
"""

import math
import random
import numpy as np
import cv2
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter

# ==================== 基础几何变换 (严格同步) ====================

class RandomHorizontalFlipTemporal:
    """随机水平翻转（严格同步）"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1, img2, label=None):
        if random.random() < self.p:
            # 这里的 ::-1 是 Numpy 的标准翻转操作，适用于任意通道数
            img1 = img1[:, ::-1, :].copy()
            img2 = img2[:, ::-1, :].copy()
            if label is not None:
                label = label[:, ::-1].copy() if label.ndim == 2 else label[:, ::-1, :].copy()

        return (img1, img2, label) if label is not None else (img1, img2)


class RandomVerticalFlipTemporal:
    """随机垂直翻转（严格同步）"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1, img2, label=None):
        if random.random() < self.p:
            img1 = img1[::-1, :, :].copy()
            img2 = img2[::-1, :, :].copy()
            if label is not None:
                label = label[::-1, :].copy() if label.ndim == 2 else label[::-1, :, :].copy()

        return (img1, img2, label) if label is not None else (img1, img2)


class RandomRotate90Temporal:
    """随机90度旋转（严格同步）"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1, img2, label=None):
        if random.random() < self.p:
            k = random.choice([1, 2, 3]) # 旋转次数 (90, 180, 270)
            img1 = np.rot90(img1, k).copy()
            img2 = np.rot90(img2, k).copy()
            if label is not None:
                label = np.rot90(label, k).copy()

        return (img1, img2, label) if label is not None else (img1, img2)


class RandomRotateTemporal:
    """随机任意角度旋转（严格同步，带填充）"""
    def __init__(self, degrees=30, p=0.5, fill_mode='reflect'):
        self.degrees = degrees
        self.p = p
        self.fill_mode = fill_mode

    def _rotate_cv2(self, img, angle, interpolation, border_mode):
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 计算旋转后的新边界框，防止图像被裁切（可选，这里保持原图大小或自适应）
        # 这里为了保持输入输出尺寸一致，通常不改变画布大小，或者采用先放大后裁剪的策略
        # 下面逻辑保持原图尺寸，边缘填充

        ret = cv2.warpAffine(
            img, M, (w, h),
            flags=interpolation,
            borderMode=border_mode
        )
        return ret

    def __call__(self, img1, img2, label=None):
        if random.random() < self.p:
            # 1. 生成唯一的随机角度，确保三张图转的一样！
            angle = random.uniform(-self.degrees, self.degrees)

            # 设定参数
            border = cv2.BORDER_REFLECT if self.fill_mode == 'reflect' else cv2.BORDER_CONSTANT

            # img 使用双线性插值
            img1 = self._rotate_cv2(img1, angle, cv2.INTER_LINEAR, border)
            img2 = self._rotate_cv2(img2, angle, cv2.INTER_LINEAR, border)

            # label 必须使用最近邻插值 (INTER_NEAREST)
            if label is not None:
                label = self._rotate_cv2(label, angle, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)

        return (img1, img2, label) if label is not None else (img1, img2)


class RandomResizedCropTemporal:
    """随机缩放裁剪（严格同步）"""
    def __init__(self, size, scale=(0.8, 1.0), ratio=(0.75, 1.33)):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.scale = scale
        self.ratio = ratio

    def get_params(self, img):
        """一次性计算出裁剪坐标"""
        height, width = img.shape[0], img.shape[1]
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback
        w = min(width, height)
        i = (height - w) // 2
        j = (width - w) // 2
        return i, j, w, w

    def __call__(self, img1, img2, label=None):
        # 1. 获取唯一的裁剪参数
        i, j, h, w = self.get_params(img1)

        # 2. 执行裁剪
        img1 = img1[i:i+h, j:j+w, :]
        img2 = img2[i:i+h, j:j+w, :]

        # 3. 执行 Resize
        img1 = cv2.resize(img1, (self.size[1], self.size[0]), interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, (self.size[1], self.size[0]), interpolation=cv2.INTER_LINEAR)

        if label is not None:
            # 裁剪 label
            if label.ndim == 3:
                label = label[i:i+h, j:j+w, :]
            else:
                label = label[i:i+h, j:j+w]
            # Resize label (最近邻!)
            label = cv2.resize(label, (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)

        return (img1, img2, label) if label is not None else (img1, img2)


# ==================== 光学变换 (保持独立性) ====================
# 注意：对于变化检测，两张图的光照、颜色应该"独立"变化，以模拟不同时相的大气差异。

class ColorJitterTemporal:
    """色彩抖动（img1 和 img2 独立抖动，模拟不同季节/光照）"""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5):
        self.p = p
        # 复用 torchvision 的变换逻辑，但用于 Numpy
        self.transform = torch.nn.Sequential(
            torch.jit.script(torchvision.transforms.ColorJitter(
                brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
            ))
        )

    def _apply_jitter(self, img_np):
        """Numpy [H,W,C] -> PIL -> Jitter -> Numpy"""
        # 只有3通道图像才能做标准的 ColorJitter，多波段需要特殊处理
        # 这里假设如果是多波段，只处理前3个波段，或者跳过
        if img_np.shape[2] != 3:
            return img_np

        pil_img = Image.fromarray(img_np.astype(np.uint8))
        jit_img = self.transform(pil_img)
        return np.array(jit_img)

    def __call__(self, img1, img2, label=None):
        # 必须独立判断概率，还是共同判断？
        # 通常：只要触发增强，两张图都做，但做的参数不一样。
        import torchvision.transforms as T

        # 为了速度，我们手动构建 color jitter transform，每次 call 生成不同的参数
        jitter = T.ColorJitter(0.2, 0.2, 0.2, 0.1) # 示例值

        if random.random() < self.p:
            # 这里的关键是：对 img1 和 img2 分别调用 jitter，
            # 它们内部会产生不同的随机亮度/对比度因子。
            # 这正是 CD 任务需要的（时相差异）。
            img1 = self._apply_jitter(img1)
            img2 = self._apply_jitter(img2)

        return (img1, img2, label) if label is not None else (img1, img2)


class GaussianBlurTemporal:
    """高斯模糊"""
    def __init__(self, sigma_range=(0.1, 2.0), p=0.3):
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, img1, img2, label=None):
        if random.random() < self.p:
            sigma = random.uniform(*self.sigma_range)
            # 可以在这里让两张图 sigma 略有不同，或者保持一致，视需求而定
            # 这里选择保持一致，模拟同等的大气条件，或者为了简单
            img1 = gaussian_filter(img1, sigma=(sigma, sigma, 0))
            img2 = gaussian_filter(img2, sigma=(sigma, sigma, 0))

        return (img1, img2, label) if label is not None else (img1, img2)


# ==================== 格式化与归一化 ====================

class NormalizeTemporal:
    """
    归一化：(x - mean) / std
    支持用户自定义 mean/std，对于多波段遥感影像非常重要。
    """
    def __init__(self, mean=None, std=None):
        # 如果不传，默认不做处理或者设为 0-1 缩放
        self.mean = np.array(mean).reshape(1, 1, -1) if mean is not None else None
        self.std = np.array(std).reshape(1, 1, -1) if std is not None else None

    def __call__(self, img1, img2, label=None):
        # 确保是 float 类型
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)

        # 简单的 0-255 -> 0-1
        if self.mean is None:
            img1 /= 255.0
            img2 /= 255.0
        else:
            img1 = (img1 / 255.0 - self.mean) / self.std
            img2 = (img2 / 255.0 - self.mean) / self.std

        return (img1, img2, label) if label is not None else (img1, img2)


class ToTensorTemporal:
    """
    管道的终点：Numpy [H, W, C] -> Tensor [C, H, W]
    """
    def __call__(self, img1, img2, label=None):
        # Image: HWC -> CHW
        img1 = torch.from_numpy(img1).float().permute(2, 0, 1)
        img2 = torch.from_numpy(img2).float().permute(2, 0, 1)

        if label is not None:
            # Label: HWC -> HW (如果是单通道) 或 HWC -> CHW (如果是多类别OneHot)
            # 通常 CD 的 Label 是 [H, W] 的 int64 (0, 1)
            # 如果进来的是 [H, W]，直接转
            label = torch.from_numpy(label).long()

        return (img1, img2, label) if label is not None else (img1, img2)


class ComposeTemporal:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img1, img2, label=None):
        for t in self.transforms:
            if label is not None:
                img1, img2, label = t(img1, img2, label)
            else:
                img1, img2 = t(img1, img2)
        return (img1, img2, label) if label is not None else (img1, img2)

# ==================== 快捷入口 ====================

def get_train_transforms(img_size=256):
    return ComposeTemporal([
        RandomHorizontalFlipTemporal(0.5),
        RandomVerticalFlipTemporal(0.5),
        RandomRotate90Temporal(0.5),
        # RandomRotateTemporal(degrees=20, p=0.3),
        # RandomResizedCropTemporal(img_size, scale=(0.8, 1.0)),
        # ColorJitterTemporal(p=0.5), # 视波段情况开启
        NormalizeTemporal(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorTemporal()
    ])

def get_val_transforms(img_size=256):
    return ComposeTemporal([
        # 验证集通常不做 Resize Crop，除非图片尺寸不统一
        # 这里假设做简单的归一化和转换
        NormalizeTemporal(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorTemporal()
    ])