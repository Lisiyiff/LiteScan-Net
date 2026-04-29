"""
耕地变化检测专用损失函数库 (Enhanced)
集成多种主流语义分割与变化检测损失函数，支持自由组合与自适应加权。
"""
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any


# ==============================================================================
# Part 1: 基础损失函数原子库
# ==============================================================================

class DiceLoss(nn.Module):
    """
    Dice Loss: 关注区域重叠度，对正负样本不平衡非常有效。
    """

    def __init__(self, smooth=1.0, reduction='mean', from_logits=True):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.from_logits = from_logits

    def forward(self, pred, target):
        if self.from_logits:
            pred = torch.sigmoid(pred)

        # [B, C, H, W] -> [B, -1]
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(1)
        union = pred_flat.sum(1) + target_flat.sum(1)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice

        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()


class IoULoss(nn.Module):
    """
    IoU Loss (Jaccard Loss): 类似于 Dice，但直接优化 IoU 指标。
    """

    def __init__(self, smooth=1.0, reduction='mean', from_logits=True):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.from_logits = from_logits

    def forward(self, pred, target):
        if self.from_logits:
            pred = torch.sigmoid(pred)

        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(1)
        total = pred_flat.sum(1) + target_flat.sum(1)
        union = total - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)
        loss = 1 - iou

        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()


class BinaryFocalLoss(nn.Module):
    """
    Focal Loss: 降低简单样本权重，专注于难分样本（Hard Mining）。
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', from_logits=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.from_logits = from_logits

    def forward(self, pred, target):
        if self.from_logits:
            bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
            pred_prob = torch.sigmoid(pred)
        else:
            bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
            pred_prob = pred

        pt = torch.exp(-bce_loss)
        # alpha用于平衡正负样本，gamma用于平衡难易样本
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()


class TverskyLoss(nn.Module):
    """
    Tversky Loss: Dice 的推广版。
    通过调整 alpha 和 beta，可以控制对 False Positive (FP) 和 False Negative (FN) 的惩罚力度。
    适合极度不平衡的小目标检测（耕地变化通常较小）。
    alpha=0.5, beta=0.5 时等价于 Dice Loss。
    alpha=0.7, beta=0.3 时更关注 Recall (减少漏检)。
    """

    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0, from_logits=True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, pred, target):
        if self.from_logits:
            pred = torch.sigmoid(pred)

        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)

        # TP (True Positive)
        tp = (pred * target).sum(1)
        # FP (False Positive): 预测为1，真实为0
        fp = (pred * (1 - target)).sum(1)
        # FN (False Negative): 预测为0，真实为1
        fn = ((1 - pred) * target).sum(1)

        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1 - tversky_index).mean()


class BCELoss(nn.Module):
    """
    标准的二值交叉熵损失，作为基准。
    """

    def __init__(self, weight=None, pos_weight=None, from_logits=True):
        super().__init__()
        self.from_logits = from_logits
        # pos_weight 可以用来给正样本加权 (e.g. pos_weight=torch.tensor([3.0]))
        self.pos_weight = pos_weight
        self.weight = weight

    def forward(self, pred, target):
        if self.from_logits:
            return F.binary_cross_entropy_with_logits(
                pred, target, weight=self.weight, pos_weight=self.pos_weight
            )
        else:
            return F.binary_cross_entropy(
                pred, target, weight=self.weight
            )


class EdgeLoss(nn.Module):
    """
    (可选) 边界感知损失，利用 Laplace 算子。
    """

    def __init__(self, edge_weight=1.0, from_logits=True):
        super().__init__()
        self.edge_weight = edge_weight
        self.from_logits = from_logits
        self.kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32).view(1, 1, 3, 3)

    def forward(self, pred, target):
        if self.from_logits:
            pred = torch.sigmoid(pred)

        if self.kernel.device != pred.device:
            self.kernel = self.kernel.to(pred.device)

        pred_edge = torch.tanh(F.conv2d(pred, self.kernel, padding=1))
        target_edge = torch.tanh(F.conv2d(target, self.kernel, padding=1))

        return F.l1_loss(pred_edge, target_edge) * self.edge_weight


# ==============================================================================
# Part 2: 组合与自适应封装
# ==============================================================================

class CombinedLoss(nn.Module):
    """
    万能组合损失器
    可以通过配置字典随意组合上述 6 种损失函数。
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Args:
            config: 格式示例
            {
                'bce': {'weight': 0.5},
                'dice': {'weight': 0.5, 'kwargs': {'smooth': 1.0}}
            }
        """
        super().__init__()
        self.loss_map = {
            'dice': DiceLoss,
            'iou': IoULoss,
            'focal': BinaryFocalLoss,
            'tversky': TverskyLoss,
            'bce': BCELoss,
            'edge': EdgeLoss
        }

        # 默认使用 Dice + BCE (非常经典且稳健的组合)
        if config is None:
            config = {
                'bce': {'weight': 0.5},
                'dice': {'weight': 0.5}
            }

        self.losses = nn.ModuleDict()
        self.weights = {}

        for name, settings in config.items():
            if name not in self.loss_map:
                print(f"⚠️ Warning: Loss '{name}' not found, skipping.")
                continue

            # 提取权重和参数
            w = settings.get('weight', 1.0)
            kwargs = settings.get('kwargs', {})

            self.losses[name] = self.loss_map[name](**kwargs)
            self.weights[name] = w

    def forward(self, pred, target, return_components=False):
        # 统一 target 维度 [B, H, W] -> [B, 1, H, W]
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()

        total_loss = 0.0
        components = {}

        for name, loss_fn in self.losses.items():
            w = self.weights[name]
            l = loss_fn(pred, target)
            total_loss += l * w
            components[name] = l.detach()

        components['total'] = total_loss.detach()

        if return_components:
            return components
        return total_loss


class AdaptiveWeightedLoss(nn.Module):
    """
    自适应权重损失 (Learnable Weights)
    Loss = 1/(2*sigma^2) * L_i + log(sigma)
    """

    def __init__(self, loss_names: List[str] = ['dice', 'bce']):
        super().__init__()
        self.loss_map = {
            'dice': DiceLoss(),
            'iou': IoULoss(),
            'focal': BinaryFocalLoss(),
            'tversky': TverskyLoss(),
            'bce': BCELoss(),
            'edge': EdgeLoss()
        }

        self.losses = nn.ModuleDict()
        for name in loss_names:
            if name in self.loss_map:
                self.losses[name] = self.loss_map[name]

        # 初始化可学习参数 log_var (sigma^2 的对数)
        # 初始化为0，即 sigma=1
        self.log_vars = nn.Parameter(torch.zeros(len(self.losses)))

    def forward(self, pred, target):
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()

        total_loss = 0.0

        for i, (name, loss_fn) in enumerate(self.losses.items()):
            loss_val = loss_fn(pred, target)

            # Kendall Alex et al. 自适应公式
            precision = torch.exp(-self.log_vars[i])
            loss = precision * loss_val + self.log_vars[i]

            total_loss += loss

        return total_loss


# ==============================================================================
# Part 3: 常用预设配置 (Design Patterns)
# ==============================================================================

def build_loss(mode='standard'):
    """
    提供几种常用的设计模式供你直接选择
    """
    if mode == 'standard':
        # 方案A: 标准组合 (最常用)
        # 优势: BCE 保证像素分类准确，Dice 保证整体形状重叠
        return CombinedLoss({
            'bce': {'weight': 0.5},
            'dice': {'weight': 0.5}
        })

    elif mode == 'imbalanced':
        # 方案B: 针对极度不平衡 (变化区域很小)
        # 优势: Focal 挖掘难样本，Tversky 提高召回率(Recall)
        return CombinedLoss({
            'focal': {'weight': 0.5, 'kwargs': {'gamma': 2.0}},
            'tversky': {'weight': 0.5, 'kwargs': {'alpha': 0.7, 'beta': 0.3}}  # beta小，惩罚漏检
        })

    elif mode == 'boundary':
        # 方案C: 关注边界 (如果你想试试 Edge Loss)
        return CombinedLoss({
            'dice': {'weight': 0.4},
            'bce': {'weight': 0.4},
            'edge': {'weight': 0.2}
        })

    elif mode == 'adaptive':
        # 方案D: 让模型自己学权重
        return AdaptiveWeightedLoss(['dice', 'focal', 'edge'])

    else:
        raise ValueError("Unknown mode")

# ... (保留你之前的 Part 1, Part 2, Part 3 代码不变)

# ==============================================================================
# Part 4: 官方代码复现专用适配器 (新增部分)
# ==============================================================================

class LossAdapterForOfficial(nn.Module):
    """
    专门适配 Daudt et al. 官方代码输出格式的损失函数封装。

    官方模型输出特点:
    1. Shape: [B, 2, H, W] (Multiclass)
    2. Activation: LogSoftmax (已经在模型内部做过了)

    适配逻辑:
    - 主Loss: 使用 NLLLoss (Negative Log Likelihood)，这是 LogSoftmax 的标准配对。
    - 辅Loss (可选): DiceLoss。需要将 2通道LogSoftmax 转换回 1通道概率，以便复现论文通常关注的变化类F1。
    """

    def __init__(self, weights={'nll': 1.0, 'dice': 0.0}):
        super().__init__()
        self.weights = weights
        # NLLLoss 期望输入是 LogSoftmax，不需要再做 LogSoftmax
        # 论文提到: "The two classes... were assigned weights inversely proportional"
        # 这里的 weight 参数可以在初始化时传入，或者默认不传
        self.nll_loss = nn.NLLLoss(ignore_index=255)

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 2, H, W], 来自模型的 LogSoftmax 输出
            target: [B, H, W], 0/1 标签
        """
        loss_total = 0.0

        # --- 1. 计算 NLL Loss (等价于 CrossEntropy) ---
        if self.weights.get('nll', 0) > 0:
            # NLLLoss 需要 target 为 Long 类型
            # 如果 target 有 1 channel 维度，先 squeeze
            if target.ndim == 4:
                target_idx = target.squeeze(1).long()
            else:
                target_idx = target.long()

            l_nll = self.nll_loss(pred, target_idx)
            loss_total += self.weights['nll'] * l_nll

        # --- 2. 计算 Dice Loss (针对变化类 Index=1) ---
        if self.weights.get('dice', 0) > 0:
            # 提取变化类的概率
            # pred 是 log_prob, 所以 prob = exp(pred)
            # 取 index=1 代表 "Change" 类
            pred_change_log_prob = pred[:, 1, :, :]
            pred_change_prob = torch.exp(pred_change_log_prob)  # [B, H, W] Range 0-1

            # 使用简单的 Dice 计算逻辑 (无需 Sigmoid，因为已经是 prob 了)
            smooth = 1.0

            pred_flat = pred_change_prob.reshape(pred.size(0), -1)
            target_flat = target.reshape(target.size(0), -1).float()

            intersection = (pred_flat * target_flat).sum(1)
            union = pred_flat.sum(1) + target_flat.sum(1)

            dice_score = (2. * intersection + smooth) / (union + smooth)
            l_dice = 1 - dice_score.mean()

            loss_total += self.weights['dice'] * l_dice

        return loss_total


def build_loss_for_official():
    """复现专用构建函数"""
    # 默认只用 NLL (对应论文的 Weighted Cross Entropy)，也可以加 Dice 辅助
    return LossAdapterForOfficial(weights={'nll': 1.0, 'dice': 1.0})


# [追加到 loss.py 末尾]

class CrossEntropyLossAdapter(nn.Module):
    """
    适配 BIT 等直接输出 Logits [B, 2, H, W] 的模型。
    """

    def __init__(self, weight=None):
        super().__init__()
        # 这里的 weight 可以处理类别不平衡，例如 torch.tensor([1.0, 3.0])
        self.loss_fn = nn.CrossEntropyLoss(weight=weight, ignore_index=255)

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 2, H, W] (Logits)
            target: [B, H, W] (Long, 0/1)
        """
        # 确保 target 是 long 类型 (CrossEntropy 要求)
        if target.ndim == 4:
            target = target.squeeze(1)
        target = target.long()

        return self.loss_fn(pred, target)


def build_loss_for_bit():
    """BIT 专用 Loss 构建器"""
    # BIT 官方通常只用 CrossEntropy，不加 Dice
    return CrossEntropyLossAdapter()


# ==============================================================================
# Part 5: 特殊模型专用损失 (SNUNet, STANet)
# ==============================================================================

class DeepSupervisionLoss(nn.Module):
    """
    适用于 SNUNet 等输出多尺度预测列表的模型。
    逻辑：对 List 中的每一个输出都计算 Loss，然后加权求和。
    """

    def __init__(self, base_loss_fn, weights: List[float] = None):
        super().__init__()
        self.base_loss = base_loss_fn
        # 默认权重：如果未指定，默认均分或随层级递增
        # SNUNet 官方通常对 4 个输出给予 [1, 1, 1, 1] 或 [0.25, 0.25, 0.25, 0.25] 的权重
        self.weights = weights if weights else [0.25, 0.25, 0.25, 0.25]

    def forward(self, preds_list, target):
        """
        Args:
            preds_list: List of Tensors. SNUNet 输出4个尺度的预测 [out1, out2, out3, out4]
            target: [B, H, W]
        """
        # 容错：如果模型只返回了一个 Tensor 而不是 List，直接计算基础 Loss
        if not isinstance(preds_list, (list, tuple)):
            return self.base_loss(preds_list, target)

        loss_total = 0.0

        # 确保权重列表长度足够，不够则循环或截断
        num_preds = len(preds_list)
        if len(self.weights) < num_preds:
            # 权重不足，默认补 1.0
            current_weights = self.weights + [1.0] * (num_preds - len(self.weights))
        else:
            current_weights = self.weights

        for i, pred in enumerate(preds_list):
            # 对每个尺度的输出计算基础 Loss
            # 注意：SNUNet 输出通常也是 [B, 2, H, W] Logits
            # 这里的 target 会自动广播或适配
            loss_total += current_weights[i] * self.base_loss(pred, target)

        return loss_total


class SNUNetHybridLoss(nn.Module):
    """
    SNUNet 专用基础损失 (复现官方的 Hybrid Loss)。
    输入是 Logits [B, 2, H, W]，计算 CrossEntropy + Dice。
    """

    def __init__(self, weight_ce=1.0, weight_dice=1.0):
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, logits, target):
        """
        Args:
            logits: [B, 2, H, W] 未经过 Softmax
            target: [B, H, W] 0/1 标签
        """
        loss = 0.0

        # 1. Cross Entropy Part
        if self.weight_ce > 0:
            if target.ndim == 4: target = target.squeeze(1)
            loss += self.weight_ce * self.ce_loss(logits, target.long())

        # 2. Dice Part (针对变化类 Index=1)
        if self.weight_dice > 0:
            # Logits -> Softmax -> Prob
            probs = torch.softmax(logits, dim=1)
            prob_change = probs[:, 1, :, :]  # 取出变化类的概率图 [B, H, W]

            # 计算 Dice
            pred_flat = prob_change.contiguous().view(prob_change.size(0), -1)
            target_flat = target.contiguous().view(target.size(0), -1).float()

            intersection = (pred_flat * target_flat).sum(1)
            union = pred_flat.sum(1) + target_flat.sum(1)

            dice = (2. * intersection + 1.0) / (union + 1.0)
            loss += self.weight_dice * (1 - dice.mean())

        return loss


class BatchBalancedContrastiveLoss(nn.Module):
    """
    适用于 STANet (Spatial-Temporal Attention Network).
    STANet 不直接输出类别概率，而是输出两张特征图的欧氏距离 (Distance Map)。
    需要使用对比损失 (Contrastive Loss) 拉近不变像素的距离，推远变化像素的距离。
    """

    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, distance_map, target):
        """
        Args:
            distance_map: [B, H, W] 模型输出的距离图 (>=0)
            target: [B, H, W] 标签 (0:不变, 1:变化)
        """
        # 展平
        dist = distance_map.view(-1)
        label = target.view(-1).float()

        # 1. 对于不变像素 (label=0): 希望 distance 趋近于 0
        # Loss_unchanged = 0.5 * distance^2
        loss_unchanged = 0.5 * torch.pow(dist, 2)

        # 2. 对于变化像素 (label=1): 希望 distance > margin
        # Loss_changed = 0.5 * max(0, margin - distance)^2
        loss_changed = 0.5 * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)

        # 3. 组合并计算均值 (Batch-Balanced 策略：分别计算正负样本的均值，防止样本不平衡)
        num_unchanged = (label == 0).sum()
        num_changed = (label == 1).sum()

        loss_u = loss_unchanged[label == 0].mean() if num_unchanged > 0 else 0.0
        loss_c = loss_changed[label == 1].mean() if num_changed > 0 else 0.0

        return loss_u + loss_c


# ==============================================================================
# Part 6: 专用构建函数
# ==============================================================================

def build_loss_for_snunet():
    """SNUNet 专用: Deep Supervision + Hybrid Loss"""
    # 基础 Loss: 针对 2通道 Logits 的 CE + Dice
    base_loss = SNUNetHybridLoss(weight_ce=1.0, weight_dice=1.0)
    # 包装为深监督 Loss (SNUNet 输出 4 层)
    return DeepSupervisionLoss(base_loss_fn=base_loss, weights=[0.25, 0.25, 0.25, 0.25])


def build_loss_for_stanet():
    """STANet 专用: 对比损失"""
    return BatchBalancedContrastiveLoss(margin=2.0)


def build_loss_for_mscanet():
    """
    MSCANet 专用构建函数
    输出: (x16, x8, x4) 三个尺度的预测结果
    策略: 深监督 (Deep Supervision)
    """
    # 1. 基础 Loss: BCE + Dice
    base_loss = CombinedLoss(config={
        'bce': {'weight': 1.0},
        'dice': {'weight': 1.0}
    })

    # 2. 深监督权重
    # MSCANet 返回顺序通常是 [Final_Result, Aux_Result_1, Aux_Result_2]
    # 我们给最终结果最高权重 1.0，辅助结果权重 0.5 (参考常见深监督策略)
    return DeepSupervisionLoss(base_loss_fn=base_loss, weights=[1.0, 0.5, 0.5])


# [追加到 utils/loss.py 末尾]

# ==============================================================================
# Part 6: SRCNet 专用组件
# ==============================================================================

class SRCNet_EdgeLoss(nn.Module):
    """SRCNet 附带的边缘损失"""

    def __init__(self, KSIZE=7):
        super(SRCNet_EdgeLoss, self).__init__()
        self.KSIZE = KSIZE
        self.avg = nn.AvgPool2d(kernel_size=KSIZE, stride=1, padding=KSIZE // 2)

    def edgeLoss(self, prediction, target):
        # 简化版实现，直接复用 SRCNet 源码逻辑
        # prediction: [B, 2, H, W]
        # target: [B, H, W]
        targetAve = self.avg(target.float().unsqueeze(1))
        at = torch.abs(target.float().unsqueeze(1) - targetAve).squeeze(1)
        # 简单处理：使用 CrossEntropy 作为基础，加权 Edge
        # 这里为了不引入太复杂的依赖，我们使用标准的 CrossEntropy
        criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=255)
        loss = criterion(prediction, target.long())
        return (loss * at).mean()


class SRCNet_CombineLoss(nn.Module):
    """
    SRCNet 专用组合损失 (BCE + Dice + Edge)，带自适应权重 sigma
    """

    def __init__(self):
        super(SRCNet_CombineLoss, self).__init__()
        self.EL = SRCNet_EdgeLoss(KSIZE=7)
        self.ce = nn.CrossEntropyLoss(ignore_index=255)

    def dice_loss(self, prediction, target):
        # 简单的 Dice (针对 Multiclass)
        probs = torch.softmax(prediction, dim=1)
        pred_1 = probs[:, 1, :, :]
        intersection = (pred_1 * target).sum()
        union = pred_1.sum() + target.sum()
        return 1 - (2. * intersection + 1) / (union + 1)

    def forward(self, predictions, target, Diss, diff, sigma):
        # 1. 预处理 Target
        target_long = target.long()
        if target_long.ndim == 4: target_long = target_long.squeeze(1)

        # 2. 预处理 Sigma
        sigmas = sigma * sigma + 1e-6

        # 3. 处理 Predictions (Tensor vs Tuple)
        total_loss = 0

        # 如果是单个 Tensor，把它包装成 List，统一逻辑
        if isinstance(predictions, torch.Tensor):
            preds_list = [predictions]
        else:
            preds_list = predictions

        # 4. 循环计算 Loss (支持深监督)
        for pred in preds_list:
            loss_ce = self.ce(pred, target_long)
            loss_dice = self.dice_loss(pred, target)
            loss_edge = self.EL.edgeLoss(pred, target_long)

            current_loss = loss_ce / sigmas[0] + loss_dice / sigmas[1] + loss_edge / sigmas[2]
            total_loss += current_loss

        return total_loss


def build_loss_for_srcnet():
    return SRCNet_CombineLoss()