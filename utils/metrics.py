"""
耕地变化检测评估指标模块 (Universal Version)
兼容单通道 (BCE/Sigmoid) 和 双通道 (CrossEntropy/Argmax) 模型输出。
"""

import torch
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt, label
from typing import Dict, Optional, List

class MetricEvaluator:
    """
    高效的指标计算器 (GPU Accelerated)
    支持两套指标体系：
    1. New (Standard): 只关注变化类 (Index 1)，符合变化检测学术标准。
    2. Old (Legacy): 计算宏平均 (Macro Average)，即 (背景指标 + 变化指标)/2。
    """
    def __init__(self, class_num=2, device='cuda'):
        self.class_num = class_num
        self.device = device
        self.reset()

    def reset(self):
        """重置累计器"""
        # total_inter/union: [0]存背景, [1]存变化
        self.total_inter = torch.zeros(self.class_num, device=self.device)
        self.total_union = torch.zeros(self.class_num, device=self.device)

        # 记录全局准确度
        self.total_correct = torch.tensor(0.0, device=self.device)
        self.total_label = torch.tensor(0.0, device=self.device)

        # 记录 Class 1 (变化) 的 TP, FP, FN
        self.tp = torch.tensor(0.0, device=self.device)
        self.fp = torch.tensor(0.0, device=self.device)
        self.fn = torch.tensor(0.0, device=self.device)

        self.batch_count = 0

    def update(self, pred_logits, target):
        """
        更新状态 (Batch Level) - 自动适配单/双通道
        Args:
            pred_logits:
                - [B, 1, H, W] (自定义模型, Sigmoid模式)
                - [B, 2, H, W] (官方复现模型, Argmax模式)
            target: [B, H, W] 或 [B, 1, H, W], 0/1 整数
        """
        # 1. 自动适配模型输出格式 (Pred -> 0/1 Mask)
        if pred_logits.ndim == 4:
            if pred_logits.size(1) == 1:
                # Case A: 单通道 (BCE/Sigmoid模式)
                # logits > 0 等价于 sigmoid(logits) > 0.5
                pred_mask = (pred_logits > 0).long().squeeze(1)
            elif pred_logits.size(1) == 2:
                # Case B: 双通道 (CrossEntropy/LogSoftmax模式)
                # 适用于官方 FC-EF, FC-Siam 等模型
                pred_mask = torch.argmax(pred_logits, dim=1).long()
            else:
                raise ValueError(f"不支持的预测通道数: {pred_logits.shape}")
        else:
            # 如果输入已经是 [B, H, W] 的掩码
             pred_mask = (pred_logits > 0).long()

        # 2. 处理标签格式 (Target -> 0/1 Mask)
        if target.ndim == 4:
            target = target.squeeze(1)
        target_mask = target.long()

        # 确保在同一设备
        if pred_mask.device != self.device: pred_mask = pred_mask.to(self.device)
        if target_mask.device != self.device: target_mask = target_mask.to(self.device)

        # 3. 核心计算 (全 GPU 向量化操作)

        # --- Class 1 (变化) 统计 ---
        # TP: 预测1，真实1
        self.tp += (pred_mask * target_mask).sum()
        # FP: 预测1，真实0
        self.fp += (pred_mask * (1 - target_mask)).sum()
        # FN: 预测0，真实1
        self.fn += ((1 - pred_mask) * target_mask).sum()

        # --- IoU 统计 (同时计算 Class 0 和 Class 1) ---
        # Class 1: 变化
        inter_1 = (pred_mask & target_mask).sum()
        union_1 = (pred_mask | target_mask).sum()
        self.total_inter[1] += inter_1
        self.total_union[1] += union_1

        # Class 0: 背景 (为了计算 old_mIoU 必须统计)
        # 逻辑取反：0变1，1变0
        pred_0 = 1 - pred_mask
        target_0 = 1 - target_mask
        inter_0 = (pred_0 & target_0).sum()
        union_0 = (pred_0 | target_0).sum()
        self.total_inter[0] += inter_0
        self.total_union[0] += union_0

        # --- OA 统计 ---
        correct = (pred_mask == target_mask)
        self.total_correct += correct.sum()
        self.total_label += target_mask.numel()

        self.batch_count += 1

    def get_results(self):
        """
        获取结果，全方位对比 New vs Old
        """
        epsilon = 1e-6

        # =========================================================
        #  1. NEW Metrics (只关注 Class 1 - 变化) -> 学术界推荐
        # =========================================================
        iou_1 = self.total_inter[1] / (self.total_union[1] + epsilon)

        precision_1 = self.tp / (self.tp + self.fp + epsilon)
        recall_1 = self.tp / (self.tp + self.fn + epsilon)
        f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1 + epsilon)

        oa = self.total_correct / (self.total_label + epsilon)

        # =========================================================
        #  2. OLD Metrics (宏平均 Macro Average) -> 旧代码逻辑
        # =========================================================

        # --- Class 0 (背景) ---
        tp_0 = self.total_inter[0]
        fp_0 = self.fn # 背景的FP = 变化的FN
        fn_0 = self.fp # 背景的FN = 变化的FP

        precision_0 = tp_0 / (tp_0 + fp_0 + epsilon)
        recall_0 = tp_0 / (tp_0 + fn_0 + epsilon)
        f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0 + epsilon)
        iou_0 = self.total_inter[0] / (self.total_union[0] + epsilon)

        # --- 宏平均计算 (Macro Average) ---
        old_mIoU = (iou_0 + iou_1) / 2.0
        old_mF1 = (f1_0 + f1_1) / 2.0
        old_mPre = (precision_0 + precision_1) / 2.0
        old_mRec = (recall_0 + recall_1) / 2.0
        old_acc = oa

        return {
            # --- 真实指标 (只看变化) ---
            "IoU": iou_1.item(),
            "F1": f1_1.item(),
            "Precision": precision_1.item(),
            "Recall": recall_1.item(),
            "OA": oa.item(),

            # --- 旧逻辑指标 (平均背景，通常虚高) ---
            "old_mIoU": old_mIoU.item(),
            "old_mF1": old_mF1.item(),
            "old_mPre": old_mPre.item(),
            "old_mRec": old_mRec.item(),
            "old_acc": old_acc.item()
        }

# =================================================================
#  高级指标计算器 (CPU密集型，建议仅在验证/测试阶段使用)
# =================================================================

class AdvancedEvaluator:
    """
    计算边界IoU和小目标召回率 (基于 Numpy/Scipy)
    这部分保持不变，用于更深入的模型分析
    """
    def __init__(self, boundary_width=3, small_obj_limit=100):
        self.boundary_width = boundary_width
        self.small_obj_limit = small_obj_limit
        self.reset()

    def reset(self):
        self.boundary_iou_sum = 0
        self.small_obj_detected = 0
        self.small_obj_total = 0
        self.count = 0

    def update(self, pred_logits, target):
        """
        更新 (CPU Level) - 自动适配单/双通道
        """
        # 1. 生成 Binary Prediction (numpy uint8)
        if pred_logits.ndim == 4 and pred_logits.size(1) == 2:
            # 双通道 -> Argmax
            pred = torch.argmax(pred_logits, dim=1).cpu().numpy().astype(np.uint8)
        else:
            # 单通道 -> Sigmoid threshold
            pred = (torch.sigmoid(pred_logits) > 0.5).cpu().numpy().astype(np.uint8)
            if pred.ndim == 4: pred = pred.squeeze(1)

        # 2. 处理 Target
        target = target.cpu().numpy().astype(np.uint8)
        if target.ndim == 4: target = target.squeeze(1)

        for i in range(pred.shape[0]):
            self._update_single(pred[i], target[i])
            self.count += 1

    def _update_single(self, pred_mask, target_mask):
        # 1. 计算 Boundary IoU
        self.boundary_iou_sum += self._calc_boundary_iou(pred_mask, target_mask)

        # 2. 计算小目标召回
        detected, total = self._calc_small_obj_recall(pred_mask, target_mask)
        self.small_obj_detected += detected
        self.small_obj_total += total

    def _calc_boundary_iou(self, pred, target):
        """计算单张图的 Boundary IoU"""
        pred_boundary = self._mask_to_boundary(pred)
        target_boundary = self._mask_to_boundary(target)

        intersection = np.logical_and(pred_boundary, target_boundary).sum()
        union = np.logical_or(pred_boundary, target_boundary).sum()

        if union == 0: return 1.0
        return intersection / union

    def _mask_to_boundary(self, mask):
        # h, w = mask.shape
        mask_binary = mask > 0
        dist = distance_transform_edt(mask_binary) + distance_transform_edt(~mask_binary)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.boundary_width, self.boundary_width))
        dilated = cv2.dilate(mask, kernel)
        eroded = cv2.erode(mask, kernel)
        return dilated - eroded

    def _calc_small_obj_recall(self, pred, target):
        """计算小目标召回情况"""
        labeled_target, num_features = label(target)
        detected_count = 0
        small_total_count = 0

        for idx in range(1, num_features + 1):
            obj_mask = (labeled_target == idx)
            area = obj_mask.sum()

            if area <= self.small_obj_limit:
                small_total_count += 1
                overlap = np.logical_and(pred, obj_mask).sum()
                if overlap > 0:
                    detected_count += 1

        return detected_count, small_total_count

    def get_results(self):
        avg_boundary_iou = self.boundary_iou_sum / (self.count + 1e-6)
        small_recall = self.small_obj_detected / (self.small_obj_total + 1e-6)
        return {
            "Boundary_IoU": avg_boundary_iou,
            "Small_Obj_Recall": small_recall
        }

# =================================================================
#  测试模块 (Optional)
# =================================================================
if __name__ == "__main__":
    print("🚀 Testing Universal Metrics Module...")
    evaluator = MetricEvaluator(device='cuda')

    # 1. 测试单通道输入 (你的模型)
    print("\n[Test 1] Single Channel Input (Shape: 4, 1, 256, 256)")
    pred_1ch = torch.randn(4, 1, 256, 256).cuda() # Logits
    target = torch.randint(0, 2, (4, 256, 256)).cuda()
    evaluator.update(pred_1ch, target)
    print("  -> Update Success")

    # 2. 测试双通道输入 (官方复现模型)
    print("\n[Test 2] Double Channel Input (Shape: 4, 2, 256, 256)")
    pred_2ch = torch.randn(4, 2, 256, 256).cuda() # Logits/LogSoftmax
    evaluator.update(pred_2ch, target)
    print("  -> Update Success")

    res = evaluator.get_results()
    print(f"\n✅ Result F1: {res['F1']:.4f}")