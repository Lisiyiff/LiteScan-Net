import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import seaborn as sns
from pathlib import Path

# 设置绘图风格，适合论文
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['font.family'] = 'serif'  # 使用衬线字体，类似 LaTeX


class Visualizer:
    def __init__(self, save_dir='./vis_results'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # 1. 训练曲线绘制 (针对 metrics.csv)
    # =========================================================
    def plot_training_curves(self, csv_path, dataset_name='Dataset'):
        """
        读取 metrics.csv 并绘制专业的论文折线图
        包含: Loss对比, F1对比, IoU对比
        """
        if not os.path.exists(csv_path):
            print(f"Error: CSV file not found at {csv_path}")
            return

        df = pd.read_csv(csv_path)
        epochs = df['Epoch']

        # 定义要画的指标组
        metrics_groups = [
            {'title': 'Loss Curves', 'y_label': 'Loss',
             'keys': [('Train_Loss', 'Train'), ('Val_Loss', 'Val')]},

            {'title': f'{dataset_name} - F1 Score', 'y_label': 'F1 Score',
             'keys': [('Train_F1', 'Train'), ('Val_F1', 'Val')]},

            {'title': f'{dataset_name} - IoU', 'y_label': 'IoU',
             'keys': [('Train_IoU', 'Train'), ('Val_IoU', 'Val')]},

            # 如果你有高级指标，也可以画
            {'title': 'Boundary IoU & Recall', 'y_label': 'Score',
             'keys': [('Val_Boundary_IoU', 'Boundary IoU'), ('Val_Small_Obj_Recall', 'Small Obj Rec')]}
        ]

        fig, axes = plt.subplots(1, 4, figsize=(24, 5))

        for ax, group in zip(axes, metrics_groups):
            for col_name, legend_label in group['keys']:
                if col_name in df.columns:
                    # 转换数据类型，防止读取为字符串
                    data = pd.to_numeric(df[col_name], errors='coerce')
                    ax.plot(epochs, data, label=legend_label, linewidth=2)

            ax.set_title(group['title'], fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(group['y_label'], fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.6)

        save_path = self.save_dir / 'training_curves.png'
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"📈 训练曲线已保存: {save_path}")
        plt.close()

    # =========================================================
    # 2. 核心功能: 生成 TP/FP/FN 误差图
    # =========================================================
    def generate_error_map(self, pred_mask, label_mask):
        """
        生成彩色误差图 (Paper Killer Feature!)
        绿色: TP (正确检测的变化)
        红色: FP (误报/虚警)
        蓝色: FN (漏报/漏检)
        黑色/透明: TN (背景)

        Args:
            pred_mask: (H, W) 0/1 numpy array
            label_mask: (H, W) 0/1 numpy array
        """
        h, w = pred_mask.shape
        # 创建空白 RGB 图像
        error_map = np.zeros((h, w, 3), dtype=np.uint8)

        # 逻辑运算
        TP = (pred_mask == 1) & (label_mask == 1)
        FP = (pred_mask == 1) & (label_mask == 0)
        FN = (pred_mask == 0) & (label_mask == 1)
        # TN 不需要上色，保持黑色

        # 赋值颜色 (BGR 格式供 OpenCV 使用, 或者 RGB 供 Matplotlib)
        # 这里使用 RGB:
        # TP -> 绿色 [0, 255, 0]
        error_map[TP] = [0, 255, 0]
        # FP -> 红色 [255, 0, 0]
        error_map[FP] = [255, 0, 0]
        # FN -> 蓝色 [0, 0, 255] (或者青色 [0, 255, 255] 更显眼)
        error_map[FN] = [0, 255, 255]

        return error_map

    # =========================================================
    # 3. 组合可视化: T1, T2, GT, Pred, ErrorMap
    # =========================================================
    def save_paper_visualization(self, img1, img2, label, pred, name):
        """
        生成一行五张图的对比条，这是论文里最常用的 Figure 形式
        Args:
            img1, img2: RGB numpy array (H, W, 3)
            label, pred: Binary numpy array (H, W)
            name: 文件名
        """
        # 1. 生成误差图
        err_map = self.generate_error_map(pred, label)

        # 2. 将 label 和 pred 转为 RGB 以便拼接
        # Label: 白色表示变化
        label_vis = np.stack([label * 255] * 3, axis=-1).astype(np.uint8)
        # Pred: 白色表示预测变化
        pred_vis = np.stack([pred * 255] * 3, axis=-1).astype(np.uint8)

        # 3. 统一大小 (以防万一)
        h, w, _ = img1.shape

        # 4. 横向拼接
        # 顺序: 时相1 | 时相2 | 真实标签 | 预测结果 | 误差分析
        combined = np.concatenate([img1, img2, label_vis, pred_vis, err_map], axis=1)

        # 5. 保存
        save_path = self.save_dir / f"{name}_vis.png"

        # OpenCV 保存默认为 BGR，所以如果输入是 RGB，需要转一下，或者用 plt 保存
        # 这里假设输入 img1 是 RGB，我们用 cv2 保存前转 BGR
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path), combined_bgr)

    # =========================================================
    # 4. 辅助: 绘制雷达图 (Radar Chart)
    # =========================================================
    def plot_radar_chart(self, models_metrics):
        """
        绘制多模型对比雷达图
        Args:
            models_metrics: dict, e.g.
            {
                'MEMF-Net': {'IoU': 0.85, 'F1': 0.90, 'Recall': 0.88, 'OA': 0.98},
                'BIT':      {'IoU': 0.80, 'F1': 0.85, 'Recall': 0.82, 'OA': 0.97}
            }
        """
        # 这是一个稍微复杂点的 matplotlib 操作，简略版实现
        categories = list(next(iter(models_metrics.values())).keys())
        N = len(categories)

        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # 闭合

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        for model_name, metrics in models_metrics.items():
            values = list(metrics.values())
            values += values[:1]  # 闭合
            ax.plot(angles, values, linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        save_path = self.save_dir / 'radar_comparison.png'
        plt.savefig(save_path, dpi=300)
        print(f"🕸️ 雷达图已保存: {save_path}")


# =========================================================
# 测试代码 (你可以直接运行这个文件看看效果)
# =========================================================
if __name__ == "__main__":
    # 1. 模拟数据测试误差图
    vis = Visualizer('./test_vis')

    # 造一点假数据 (256x256)
    fake_gt = np.zeros((256, 256), dtype=np.uint8)
    fake_gt[50:150, 50:150] = 1  # 中间有个正方形是变化

    fake_pred = np.zeros((256, 256), dtype=np.uint8)
    fake_pred[60:160, 60:160] = 1  # 预测偏了一点点 -> 会产生 FP 和 FN

    fake_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    print("生成测试误差图...")
    vis.save_paper_visualization(fake_img, fake_img, fake_gt, fake_pred, 'test_sample')

    # 2. 模拟 CSV 测试曲线
    # 如果你有真实的 metrics.csv，把下面路径改成你的
    # vis.plot_training_curves('./checkpoints/PX_CLCD/MSCANet/metrics.csv')
    print("Done.")