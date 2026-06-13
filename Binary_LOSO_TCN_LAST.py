#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
复现三个 LOSO TCN 模型的 t-SNE 图：
1) subj_MCI_06/best.pth (测试集: MCI_06)
2) subj_HC_06/best.pth (测试集: HC_06)  
3) subj_ALL_06/best.pth (测试集: MCI_06+HC_06)

仅绘制指定测试集的t-SNE图
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import euclidean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------
# TCN 模型（来自你提供的原代码，保持一致）
# ------------------------------------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super().__init__(); self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, in_channels, num_classes=2,
                 channels=[64, 64, 128, 128], kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        prev = in_channels
        for i, c in enumerate(channels):
            layers.append(TemporalBlock(prev, c, kernel_size, stride=1,
                                        dilation=2**i, dropout=dropout))
            prev = c
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(prev, num_classes)

    def forward(self, x):
        z = self.tcn(x)
        g = self.pool(z).squeeze(-1)
        return self.head(g)

    @torch.no_grad()
    def extract_feat(self, x):
        z = self.tcn(x)
        g = self.pool(z).squeeze(-1)
        return 

# ------------------------------------------------------
# 固定列 & 标签
# ------------------------------------------------------
REQUIRED_COLUMNS = [
    'leftEye_gaze_X','leftEye_gaze_Y','leftEye_gaze_Z',
    'leftEye_openness','leftEye_pupil_position_X',
    'leftEye_pupil_position_Y','leftEye_pupil_dilation',
    'rightEye_gaze_X','rightEye_gaze_Y','rightEye_gaze_Z',
    'rightEye_openness','rightEye_pupil_position_X',
    'rightEye_pupil_position_Y','rightEye_pupil_dilation',
    'combinedEye_gaze_X','combinedEye_gaze_Y','combinedEye_gaze_Z'
]

STAGE_LABEL = [('1', 0), ('2', 0), ('3', 1), ('4', 1)]

# ------------------------------------------------------
# 窗口化（保持一致）
# ------------------------------------------------------
def windowize(arr, label, window_size=240):
    C,T = arr.shape
    if T < window_size: return [], []
    X, Y = [], []
    step = window_size
    n_seg = (T - window_size) // step + 1
    for i in range(n_seg):
        seg = arr[:, i*step:i*step+window_size].astype("float32")
        X.append(torch.from_numpy(seg.reshape(1,-1)))
        Y.append(label)
    return X, Y

# ------------------------------------------------------
# 提取指定测试集的数据
# ------------------------------------------------------
def load_test_data(data_root, test_subjects):
    """
    加载指定测试受试者的数据
    
    Args:
        data_root: 数据根目录
        test_subjects: 测试受试者列表，格式为 [('MCI', '06'), ('HC', '07'), ...]
    """
    feats, labels = [], []
    
    for pop, subj_id in test_subjects:
        for stage, lab in STAGE_LABEL:
            stage_path = os.path.join(data_root, pop, stage)
            if not os.path.isdir(stage_path): continue
            
            # 查找该受试者的CSV文件
            for f in os.listdir(stage_path):
                if not f.endswith(".csv"): continue
                # 假设文件名包含受试者ID，例如：subj_06_session_1.csv
                # 这里需要根据实际文件名调整
                if f"subj_{subj_id}" in f or f"subject_{subj_id}" in f or f"{subj_id}_" in f:
                    csv_path = os.path.join(stage_path, f)
                    
                    df = pd.read_csv(csv_path)
                    cols = [c for c in REQUIRED_COLUMNS if c in df.columns]
                    if len(cols)==0: continue
                    
                    arr = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
                    if arr.shape[0] > arr.shape[1]:
                        arr = arr.T  # (C,T)
                    
                    Xw, Yw = windowize(arr, lab)
                    feats.extend(Xw)
                    labels.extend(Yw)
                    print(f"  加载: {pop}/{stage}/{f} -> {len(Xw)}个窗口")
    
    return feats, labels

# ------------------------------------------------------
# t-SNE 绘制
# ------------------------------------------------------
def run_tsne(feats, labels, save_path, title):
    if len(feats) == 0:
        print(f"警告: {title} 没有数据!")
        return
    
    feats = torch.stack(feats).numpy()  # (N,D)
    labels = np.array(labels)

    print(f"  特征形状: {feats.shape}, 标签分布: LCL={sum(labels==0)}, HCL={sum(labels==1)}")
    
    # PCA → t-SNE
    n_components = min(50, feats.shape[1])
    Z = PCA(n_components=n_components).fit_transform(feats)
    emb = TSNE(n_components=2, init='pca', learning_rate=200, perplexity=30,
               n_iter=1500, metric='cosine').fit_transform(Z)

    # 质量指标
    if len(np.unique(labels)) > 1:
        sc = silhouette_score(emb, labels)
        dbi = davies_bouldin_score(emb, labels)
    else:
        sc = dbi = 0.0
    
    # 类中心距
    if sum(labels==0) > 0 and sum(labels==1) > 0:
        cen0 = emb[labels==0].mean(axis=0)
        cen1 = emb[labels==1].mean(axis=0)
        dist = euclidean(cen0, cen1)
    else:
        dist = 0.0

    # 绘图
    plt.figure(figsize=(6.5,5))
    
    if sum(labels==0) > 0:
        plt.scatter(emb[labels==0,0], emb[labels==0,1], s=8, c="#1f77b4", alpha=0.7, label="LCL")
    if sum(labels==1) > 0:
        plt.scatter(emb[labels==1,0], emb[labels==1,1], s=8, c="#d62728", alpha=0.7, label="HCL")
    
    plt.title(title)
    plt.legend()

    ax = plt.gca()
    txt = f"SC = {sc:.3f}\nDBI = {dbi:.3f}\ndist(LCL-HCL) = {dist:.3f}"
    ax.text(1.02, 0.98, txt, transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ------------------------------------------------------
# 主流程
# ------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True,
                        help="原眼动 CSV 根目录")
    parser.add_argument("--base", type=str, default="Binary_LOSO_68/FINAL_TCN/tcn_2/tcn_MCI")
    args = parser.parse_args()

    # 三个模型及其对应的测试集
    model_configs = [
        {
            "name": "MCI_06",
            "model_path": os.path.join(args.base, "subj_MCI_06/best.pth"),
            "test_subjects": [("MCI", "06")]  # 只测试MCI_06
        },
        {
            "name": "HC_06", 
            "model_path": os.path.join(args.base, "subj_HC_06/best.pth"),
            "test_subjects": [("HC", "06")]  # 只测试HC_06
        },
        {
            "name": "ALL_06",
            "model_path": os.path.join(args.base, "subj_ALL_06/best.pth"),
            "test_subjects": [("MCI", "06"), ("HC", "06")]  # 测试MCI_06和HC_06
        }
    ]

    # 输入通道 = 17
    model = TCN(in_channels=17)

    for config in model_configs:
        name = config["name"]
        model_path = config["model_path"]
        test_subjects = config["test_subjects"]
        
        print(f"\n===== 处理 {name} =====")
        print(f"模型: {model_path}")
        print(f"测试集: {test_subjects}")
        
        # 加载模型
        sd = torch.load(model_path, map_location="cpu")
        model.load_state_dict(sd)
        model.eval()
        
        # 加载测试数据
        print("加载测试数据...")
        feats_test, labels_test = load_test_data(args.data_root, test_subjects)
        print(f"测试集窗口数: {len(feats_test)}")
        
        if len(feats_test) == 0:
            print(f"错误: 没有找到测试数据! 请检查文件命名格式。")
            continue
        
        # 提取特征
        feats, labels = [], []
        for x, y in zip(feats_test, labels_test):
            x = x.view(1, 17, -1)  # (B,C,T)
            f = model.extract_feat(x).squeeze(0)
            feats.append(f)
            labels.append(y)
        
        # 绘制t-SNE
        save_path = f"tsne_results/{name}_testset_tsne.png"
        title = f"t-SNE — {name} (测试集)"
        run_tsne(feats, labels, save_path, title)
        print(f"[OK] 保存：{save_path}")

if __name__ == "__main__":
    main()