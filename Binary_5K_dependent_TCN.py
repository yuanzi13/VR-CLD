#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最好结果的指令：
CUDA_VISIBLE_DEVICES=3 python Binary_5K_dependent_68/TCN.py   --data-root data_rml   --result-dir Binary_5K_dependent_68/TCN_search_parameter  --datasets MCI,HC,ALL  --window-size 240 --overlap 0.0  --epochs 250 --batch-size 256  --lr 1e-3 --weight-decay 1e-4  --patience 80 --min-delta 0.0 --val-ratio 0.1  --kernel-size 3 --dropout 0.2 --channels 64,64,128,128  --gpu-id 0 
tcn.py：TCN 时序卷积模型（PyTorch）二分类 + 非独立 5 折 + t-SNE 可视化
输出目录结构：
  Binary_5K_dependent_68/tcn_2/tcn_<DATASET>/fold_xx/...
  - confusion.png / roc.png / curve_loss_acc.png / tsne.png
  - best.pth / train_val_history.csv
Overall:
  - confusion_overall.png / roc_overall.png / accuracy_across_folds.png / tsne_overall.png

新增：
1) overall 合并 ROC（MCI, HC, ALL）保存在 --result-dir 下： roc_MCI_HC_ALL_combined.png
2) t-SNE 图中计算并显示类间质心距离（右侧文本框）
3) t-SNE 图例中标签 0 -> LCL, 1 -> HCL
"""
import os, argparse, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import itertools, math
rcParams['font.family'] = ['WenQuanYi Micro Hei']
from sklearn.metrics import silhouette_score, davies_bouldin_score
"""
feats: (N, D)  模型提取向量（CPU）
labels: (N,)   0: LCL, 1: HCL
新增：计算并显示 SC 与 DBI，所有文字置于轴外右侧
"""
import os, itertools, numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
# ------------------------ 必要特征列 ------------------------
REQUIRED_COLUMNS = [
    'leftEye_gaze_X','leftEye_gaze_Y','leftEye_gaze_Z',
    'leftEye_openness','leftEye_pupil_position_X',
    'leftEye_pupil_position_Y','leftEye_pupil_dilation',
    'rightEye_gaze_X','rightEye_gaze_Y','rightEye_gaze_Z',
    'rightEye_openness','rightEye_pupil_position_X',
    'rightEye_pupil_position_Y','rightEye_pupil_dilation',
    'combinedEye_gaze_X','combinedEye_gaze_Y','combinedEye_gaze_Z'
]

# ------------------------ 标签映射（二分类） ------------------------
STAGE_LABEL = [('1', 0), ('2', 0), ('3', 1), ('4', 1)]

# ------------------------ 数据窗口化 ------------------------
def windowize_from_array(arr_ch_t: np.ndarray, label: int, window_size: int = 240, overlap: float = 0.0):
    """
    arr_ch_t: (C, T)
    返回：list[torch.Tensor(1,-1)] , list[int]
    """
    if arr_ch_t.ndim != 2:
        return [], []
    C, T = arr_ch_t.shape
    step = max(1, window_size - int(window_size * overlap))
    if T < window_size:
        return [], []
    X, Y = [], []
    n_seg = (T - window_size) // step + 1
    for i in range(n_seg):
        seg = arr_ch_t[:, i*step:i*step+window_size].astype('float32')
        seg = np.nan_to_num(seg)
        X.append(torch.from_numpy(seg.reshape(1, -1)))  # 1 × (C*T)
        Y.append(label)
    return X, Y

def split_windows_to_5folds(windows, labels, n_folds=5):
    folds = [{'X': [], 'Y': []} for _ in range(n_folds)]
    n = len(windows)
    if n == 0: return folds
    base, rem = n // n_folds, n % n_folds
    start = 0
    for k in range(n_folds):
        size = base + (1 if k < rem else 0)
        end = start + size
        if size > 0:
            folds[k]['X'].extend(windows[start:end])
            folds[k]['Y'].extend(labels[start:end])
        start = end
    return folds

def build_5fold_windows(data_root, subjects, window_size, overlap, n_folds=5):
    folds = [{'X': [], 'Y': []} for _ in range(n_folds)]
    for (pop, num) in subjects:
        for stage, lab in STAGE_LABEL:
            csv_path = os.path.join(data_root, pop, stage, f'{num}.csv')
            if not os.path.isfile(csv_path): continue
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f'读取失败: {csv_path} | {e}'); continue
            cols = [c for c in REQUIRED_COLUMNS if c in df.columns]
            if not cols:
                print(f'⚠️ 无 REQ 列：{csv_path}'); continue
            arr = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype('float32')
            if arr.shape[0] > arr.shape[1]:
                arr = arr.T  # (C,T)
            if arr.shape[1] == 0: continue
            Xw, Yw = windowize_from_array(arr, lab, window_size, overlap)
            perfile = split_windows_to_5folds(Xw, Yw, n_folds=n_folds)
            for k in range(n_folds):
                folds[k]['X'].extend(perfile[k]['X'])
                folds[k]['Y'].extend(perfile[k]['Y'])
    return folds

def get_subject_list(dataset_type):
    if dataset_type == 'MCI':
        return [('MCI', i) for i in range(1, 26+1)]
    if dataset_type == 'HC':
        return [('HC', i) for i in range(1, 42+1)]
    if dataset_type == 'ALL':
        return [('MCI', i) for i in range(1, 26+1)] + [('HC', i) for i in range(1, 42+1)]
    raise ValueError(dataset_type)

# ------------------------ 可视化：混淆矩阵 / ROC ------------------------
def plot_confusion_matrix(conf_mat, acc, total, side_txt, save_path):
    n = conf_mat.shape[0]
    M = np.zeros((n + 1, n + 1), dtype=float)
    M[:n, :n] = conf_mat
    for i in range(n):
        tp = conf_mat[i, i]
        cs = conf_mat[:, i].sum()
        rs = conf_mat[i, :].sum()
        M[n, i] = tp / cs if cs else 0
        M[i, n] = tp / rs if rs else 0
    M[n, n] = acc

    plt.figure(figsize=(8, 6))
    plt.imshow(M, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    ticks = np.arange(n + 1)
    plt.xticks(ticks, [f'P{i}' for i in range(n)] + ['Precision'], rotation=45, fontsize=12)
    plt.yticks(ticks, [f'T{i}' for i in range(n)] + ['Recall'], fontsize=12)

    thresh = M.max() / 2
    for i, j in np.ndindex(M.shape):
        v = M[i, j]
        if i < n and j < n:
            p = v / total if total else 0
            s = f'{int(v)}\n({p:.2%})'
        else:
            s = f'{v*100:.2f}%'
        plt.text(j, i, s, ha='center', va='center',
                 color='white' if v > thresh else 'black', fontsize=22)

    ax = plt.gca()
    ax.text(1.05, 0.05, side_txt, transform=ax.transAxes,
            va='top', ha='left', linespacing=1.3, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    plt.ylabel('真实标签', fontsize=14)
    plt.xlabel('预测标签', fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300); plt.close()

def plot_roc_safe(y_true, y_score, title, save_path):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
    except Exception as e:
        print(f"⚠️ {title} 无法绘制 ROC（原因：{e}），跳过。"); return
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2, label=f'AUC={auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(title, fontsize=14); plt.legend(loc='lower right')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300); plt.close()

def plot_acc_curve(accs, labels, save_path):
    plt.figure(figsize=(8, 4))
    plt.bar(labels, accs); plt.ylim(0, 1)
    plt.xlabel('Fold'); plt.ylabel('Accuracy'); plt.title('各折准确率')
    for i, v in enumerate(accs):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
    plt.tight_layout(); os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300); plt.close()

# ------------------------ 训练可视化曲线 & 早停 ------------------------
class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0, maximize=True):
        self.patience = patience
        self.min_delta = min_delta
        self.maximize = maximize
        self.best = -float('inf') if maximize else float('inf')
        self.num_bad = 0
    def step(self, metric):
        improved = (metric > self.best + self.min_delta) if self.maximize else (metric < self.best - self.min_delta)
        if improved:
            self.best = metric
            self.num_bad = 0
            return True
        else:
            self.num_bad += 1
            return False
    def should_stop(self):
        return self.num_bad >= self.patience

def plot_train_val_curves(history, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    epochs = history['epoch']
    ax1.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'],   label='Val Loss',   linewidth=2, linestyle='--')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(epochs, history['train_acc'], label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'],   label='Val Acc',   linewidth=2, linestyle='--')
    ax2.set_ylabel('Accuracy')
    lines, labels = [], []
    for ax in (ax1, ax2):
        L = ax.get_lines()
        lines += L
        labels += [l.get_label() for l in L]
    fig.legend(lines, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    plt.savefig(save_path, dpi=300); plt.close(fig)

# ------------------------ 设备选择 ------------------------
def select_device(gpu_id):
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU."); return torch.device('cpu')
    vis = torch.cuda.device_count()
    idx = 0 if gpu_id is None else int(gpu_id)
    if idx < 0 or idx >= vis:
        print(f"⚠️ requested gpu-id={idx} 无效（可见GPU数: {vis}），回退到 0。"); idx = 0
    name = torch.cuda.get_device_name(idx)
    print(f"Using device: cuda:{idx} ({name}) | visible_gpus={vis}")
    return torch.device(f'cuda:{idx}')

# ------------------------ TCN 模型（含特征提取） ------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super().__init__(); self.chomp_size = chomp_size
    def forward(self, x):  # x: (B, C, T_pad)
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=pad, dilation=dilation),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=pad, dilation=dilation),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout)
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, in_channels, num_classes=2,
                 channels=[64, 64, 128, 128], kernel_size=3, dropout=0.2,  dilation_base=2):
        super().__init__()
        layers = []
        prev = in_channels
        for i, c in enumerate(channels):
            layers.append(TemporalBlock(prev, c, kernel_size, stride=1,
                                        dilation=dilation_base**i, dropout=dropout))
            prev = c
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(prev, num_classes)
    def forward(self, x):  # x: (B, C, T)
        z = self.tcn(x)
        g = self.pool(z).squeeze(-1)   # (B, C')
        return self.fc(g)
    def extract_feat(self, x):         # 用于 t-SNE 的特征
        with torch.no_grad():
            z = self.tcn(x)
            g = self.pool(z).squeeze(-1)  # (B, C')
        return g

# ------------------------ t-SNE + SC & DBI ------------------------
def plot_tsne(feats, labels, save_path, title=None, dpi=300,
              per_class=1500, random_state=42, point_size=10, alpha=0.6):
    feats = np.asarray(feats); labels = np.asarray(labels)
    feats = np.nan_to_num(feats, copy=False)
    n = feats.shape[0]
    if n < 3:
        plt.figure(figsize=(6, 4)); plt.axis('off')
        msg = f"{title or 't-SNE'}\n(n={n} < 3, skipped)"
        plt.text(0.5, 0.5, msg, ha='center', va='center', fontsize=12)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi); plt.close(); return

    # 1. 等量抽样
    rng = np.random.RandomState(random_state)
    idx = []
    for c in np.unique(labels):
        I = np.where(labels == c)[0]
        take = min(per_class, len(I))
        idx.extend(rng.choice(I, take, replace=False))
    idx = np.array(idx)
    feats, labels = feats[idx], labels[idx]

    # 2. PCA→TSNE
    d = min(10, feats.shape[1], feats.shape[0]-1)
    Z = PCA(n_components=d, random_state=random_state).fit_transform(feats)
    emb = TSNE(n_components=2, init='pca', learning_rate=200,
               perplexity=40, n_iter=1500, early_exaggeration=12,
               metric='cosine', random_state=random_state).fit_transform(Z)

    # 3. 计算 SC & DBI
    sc = silhouette_score(emb, labels)
    dbi = davies_bouldin_score(emb, labels)

    # 4. 画图
    colors = {0: '#1f77b4', 1: '#d62728'}
    name_map = {0: 'LCL', 1: 'HCL'}
    plt.figure(figsize=(6.8, 5.2))
    for c in np.unique(labels):
        m = labels == c
        plt.scatter(emb[m, 0], emb[m, 1], s=point_size, alpha=alpha,
                    c=colors.get(int(c), '#7f7f7f'), edgecolors='none',
                    label=f'{name_map.get(int(c), str(int(c)))} (n={m.sum()})')

    # 5. 质心距离
    unique_classes = np.unique(labels)
    centroids = {c: emb[labels == c].mean(axis=0) for c in unique_classes}
    dist_lines = []
    for a, b in itertools.combinations(unique_classes, 2):
        d_ab = float(np.linalg.norm(centroids[a] - centroids[b]))
        dist_lines.append(f"{name_map.get(int(a), a)} - {name_map.get(int(b), b)}: {d_ab:.3f}")

    # 6. 右侧文字块（全部轴外）
    ax = plt.gca()
    info_txt = (
        f"Silhouette Score: {sc:.3f}\n"
        f"Davies-Bouldin Index: {dbi:.3f}\n\n"
        "Centroid distances:\n" + "\n".join(dist_lines)
    )
    ax.text(1.02, 0.05, info_txt, transform=ax.transAxes, va='bottom', ha='left',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    if title:
        plt.title(title)
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.legend(loc='best', framealpha=0.9, fontsize=9, title='Class')
    plt.tight_layout(rect=[0, 0, 0.90, 1])   # 给右侧文字留空
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
# ------------------------ 训练-评估 单折（含特征收集） ------------------------
def run_fold(model, device, train_loader, val_loader, test_loader,
             epochs, lr, weight_decay, out_dir,
             patience=10, min_delta=0.0, save_best=True, save_csv=True):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    early = EarlyStopper(patience=patience, min_delta=min_delta, maximize=True)

    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_path = os.path.join(out_dir, 'best.pth')

    for ep in range(1, epochs + 1):
        # —— Train —— #
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = crit(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * yb.size(0)
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
        train_loss = loss_sum / max(1, total)
        train_acc  = correct / max(1, total)

        # —— Validate —— #
        model.eval()
        v_total, v_correct, v_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = crit(logits, yb)
                v_loss_sum += loss.item() * yb.size(0)
                pred = logits.argmax(1)
                v_correct += (pred == yb).sum().item()
                v_total += yb.size(0)
        val_loss = v_loss_sum / max(1, v_total)
        val_acc  = v_correct / max(1, v_total)

        history['epoch'].append(ep)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if ep == 1 or ep % max(1, epochs // 10) == 0 or ep == epochs:
            print(f"  Epoch {ep}/{epochs} | train_loss={train_loss:.4f} acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} acc={val_acc:.4f}")

        improved = early.step(val_acc)
        if improved and save_best:
            torch.save({'epoch': ep, 'state_dict': model.state_dict()}, best_path)
        if early.should_stop():
            print(f"  ▲ Early stopping at epoch {ep} (best val acc={early.best:.4f}).")
            break

    plot_train_val_curves(history, os.path.join(out_dir, 'curve_loss_acc.png'))

    if save_csv:
        import csv
        with open(os.path.join(out_dir, 'train_val_history.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['epoch','train_loss','train_acc','val_loss','val_acc'])
            for i in range(len(history['epoch'])):
                w.writerow([history['epoch'][i], history['train_loss'][i], history['train_acc'][i],
                            history['val_loss'][i], history['val_acc'][i]])

    if save_best and os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['state_dict'])

    # —— Test（并收集特征做 t-SNE） —— #
    model.eval()
    y_true, y_pred, y_prob1 = [], [], []
    feat_list, lab_list = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            prob = torch.softmax(logits, dim=1)[:, 1]
            y_true.extend(yb.numpy().tolist())
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
            y_prob1.extend(prob.cpu().numpy().tolist())

            # 提取特征
            feats = model.extract_feat(xb).cpu().numpy()  # (B, D)
            feat_list.append(feats)
            lab_list.extend(yb.numpy().tolist())

    feats_arr = np.vstack(feat_list) if feat_list else None
    labels_arr = np.array(lab_list) if lab_list else None

    # 每折 t-SNE
    if feats_arr is not None and labels_arr.size > 0:
        plot_tsne(feats_arr, labels_arr, os.path.join(out_dir, "tsne.png"),
                  title="t-SNE (Test, best model)", dpi=300)

    return np.array(y_true), np.array(y_pred), np.array(y_prob1), feats_arr, labels_arr

# ------------------------ 主流程 ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default=os.path.join('DeepLearning','data_rml'))
    parser.add_argument('--result-dir', type=str, default='Binary_5K_dependent_68')
    parser.add_argument('--datasets', default='MCI,HC,ALL', help="MCI,HC,ALL（逗号分隔）")
    parser.add_argument('--window-size', type=int, default=240)
    parser.add_argument('--overlap', type=float, default=0.0)
    # 训练超参
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    # 验证 & 早停
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min-delta', type=float, default=0.0)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    # TCN 架构
    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--channels', type=str, default='64,64,128,128', help='逗号分隔的通道数列表')
    parser.add_argument('--dilation-base', type=int, default=2, help='dilation = dilation-base^i')
    # 设备
    parser.add_argument('--gpu-id', type=int, default=None)
    args = parser.parse_args()

    # 兼容 DeepLearning/data_rml
    if not os.path.isdir(args.data_root) and os.path.isdir(os.path.join('DeepLearning', 'data_rml')):
        args.data_root = os.path.join('DeepLearning', 'data_rml')

    device = select_device(args.gpu_id)
    datasets_req = [d.strip().upper() for d in args.datasets.split(',')]
    all_datasets = ['MCI','HC','ALL']
    for d in datasets_req:
        if d not in all_datasets:
            raise ValueError(f'不支持的数据集: {d}（可选: {",".join(all_datasets)}）')

    C = len(REQUIRED_COLUMNS); W = args.window_size
    chan_list = [int(x) for x in args.channels.split(',') if x.strip()]

    # 用于合并绘制 MCI/HC/ALL overall ROC
    combined_overall = {}

    for dtype in datasets_req:
        subjects = get_subject_list(dtype)
        folds = build_5fold_windows(args.data_root, subjects, args.window_size, args.overlap, n_folds=5)

        # 调试：打印每折标签分布
        for k in range(5):
            yk = np.array(folds[k]['Y'])
            print(f"[DEBUG][{dtype}] Fold {k+1}: 标签计数 {dict(zip(*np.unique(yk, return_counts=True))) if yk.size>0 else '空'}")

        res_dir = os.path.join(args.result_dir, 'tcn_2', f"tcn_{dtype}")
        os.makedirs(res_dir, exist_ok=True)

        total_conf = np.zeros((2, 2), int)
        y_t_all, y_p_all, y_pr_all = [], [], []
        accs, fold_labels = [], []

        # overall t-SNE 累计
        overall_feats, overall_labels = [], []

        for k in range(5):
            fold_dir = os.path.join(res_dir, f"fold_{k+1:02d}")
            os.makedirs(fold_dir, exist_ok=True)

            Xte_list, Yte_list = folds[k]['X'], folds[k]['Y']
            Xtr_list, Ytr_list = [], []
            for j in range(5):
                if j == k: continue
                Xtr_list += folds[j]['X']
                Ytr_list += folds[j]['Y']

            print(f"[TCN][{dtype}] Fold {k+1}/5 | train={len(Ytr_list)} windows, test={len(Yte_list)} windows")
            if not Xtr_list or not Xte_list:
                print(f"⚠️ Fold {k+1} 数据不足，跳过"); continue

            # 扁平 → 标准化 → 还原成 (B, C, W) 喂 TCN
            Xtr_flat = np.vstack([x.numpy() for x in Xtr_list])   # (N, C*W)
            Xte_flat = np.vstack([x.numpy() for x in Xte_list])
            Ytr = np.array(Ytr_list, dtype=np.int64); Yte = np.array(Yte_list, dtype=np.int64)

            scaler = StandardScaler().fit(Xtr_flat)
            Xtr_s = scaler.transform(Xtr_flat).reshape(-1, C, W)
            Xte_s = scaler.transform(Xte_flat).reshape(-1, C, W)

            Xtr_tensor = torch.tensor(Xtr_s, dtype=torch.float32)
            Xte_tensor = torch.tensor(Xte_s, dtype=torch.float32)
            Ytr_tensor = torch.tensor(Ytr, dtype=torch.long)
            Yte_tensor = torch.tensor(Yte, dtype=torch.long)

            # —— 划分训练/验证 —— #
            full_ds = TensorDataset(Xtr_tensor, Ytr_tensor)
            n_val = max(1, int(args.val_ratio * len(full_ds)))
            n_trn = max(1, len(full_ds) - n_val)
            trn_ds, val_ds = random_split(full_ds, [n_trn, n_val], generator=torch.Generator().manual_seed(42))

            train_loader = DataLoader(trn_ds, batch_size=args.batch_size, shuffle=True)
            val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
            test_loader  = DataLoader(TensorDataset(Xte_tensor, Yte_tensor), batch_size=args.batch_size, shuffle=False)

            model = TCN(in_channels=C, num_classes=2,
            channels=chan_list, kernel_size=args.kernel_size,
            dropout=args.dropout, dilation_base=args.dilation_base)

            # —— 训练（带验证/早停/最优模型保存） —— #
            y_true, y_pred, y_prob1, feats_fold, labels_fold = run_fold(
                model, device, train_loader, val_loader, test_loader,
                epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                out_dir=fold_dir, patience=args.patience, min_delta=args.min_delta, save_best=True
            )

            cm  = confusion_matrix(y_true, y_pred, labels=[0,1])
            acc = accuracy_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred, zero_division=0)
            pre = precision_score(y_true, y_pred, zero_division=0)
            f1s = f1_score(y_true, y_pred, zero_division=0)
            try:
                auc = roc_auc_score(y_true, y_prob1)
            except Exception:
                auc = 0.0

            side_txt = (f"{dtype} TCN Fold {k+1}\n"
                        f"Acc={acc:.4f}  Rec={rec:.4f}\n"
                        f"Pre={pre:.4f}  F1={f1s:.4f}\n"
                        f"AUC={auc:.4f}")

            plot_confusion_matrix(cm, acc, len(y_true), side_txt, os.path.join(fold_dir, "confusion.png"))
            plot_roc_safe(y_true, y_prob1, f"{dtype} TCN Fold {k+1} ROC", os.path.join(fold_dir, "roc.png"))

            total_conf += cm
            y_t_all += y_true.tolist()
            y_p_all += y_pred.tolist()
            y_pr_all += y_prob1.tolist()
            accs.append(acc); fold_labels.append(str(k+1))

            # 累计 overall t-SNE
            if feats_fold is not None and labels_fold is not None and len(labels_fold) > 0:
                overall_feats.append(feats_fold)
                overall_labels.extend(labels_fold.tolist())

        if accs:
            plot_acc_curve(accs, fold_labels, os.path.join(res_dir, "accuracy_across_folds.png"))

        if y_t_all:
            oa = accuracy_score(y_t_all, y_p_all)
            orc = recall_score(y_t_all, y_p_all, zero_division=0)
            opc = precision_score(y_t_all, y_p_all, zero_division=0)
            of1 = f1_score(y_t_all, y_p_all, zero_division=0)
            try:
                oauc = roc_auc_score(y_t_all, y_pr_all)
            except Exception:
                oauc = 0.0

            otxt = (f"{dtype} TCN Overall\n"
                    f"Acc={oa:.4f}  Rec={orc:.4f}\n"
                    f"Pre={opc:.4f}  F1={of1:.4f}\n"
                    f"AUC={oauc:.4f}")
            plot_confusion_matrix(total_conf, oa, len(y_t_all), otxt, os.path.join(res_dir, "confusion_overall.png"))
            plot_roc_safe(y_t_all, y_pr_all, f"{dtype} TCN Overall ROC", os.path.join(res_dir, "roc_overall.png"))

            # 保存以供 later 合并绘制 MCI/HC/ALL ROC
            try:
                combined_overall[dtype] = {
                    'y_true': np.array(y_t_all),
                    'y_score': np.array(y_pr_all),
                    'result_dir': res_dir
                }
            except Exception:
                pass

        # —— Overall t-SNE —— #
        if len(overall_labels) > 0:
            feats_all = np.vstack(overall_feats)
            feats_all = np.nan_to_num(feats_all, copy=False)
            plot_tsne(feats_all, np.array(overall_labels),
                      save_path=os.path.join(res_dir, "tsne_overall.png"),
                      title=f"{dtype} TCN Overall t-SNE", dpi=300)

    # ===== 合并绘制 MCI / HC / ALL Overall ROC（如果有） =====
    if combined_overall:
        plt.figure(figsize=(7,6))
        has_any = False
        for k, name in enumerate(['MCI','HC','ALL']):
            info = combined_overall.get(name)
            if info is None: continue
            y_t = info['y_true']; y_s = info['y_score']
            try:
                fpr, tpr, _ = roc_curve(y_t, y_s)
                auc_v = roc_auc_score(y_t, y_s)
                plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC={auc_v:.4f})')
                has_any = True
            except Exception as e:
                print(f"⚠️ 合并 ROC 绘制：{name} 失败（{e}）")
        if has_any:
            plt.plot([0,1],[0,1], linestyle='--', color='k', lw=1)
            plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
            plt.title('Overall ROC Comparison: MCI / HC / ALL')
            plt.legend(loc='lower right', fontsize=10)
            outp = os.path.join(args.result_dir, "roc_MCI_HC_ALL_combined.png")
            os.makedirs(os.path.dirname(outp), exist_ok=True)
            plt.tight_layout(); plt.savefig(outp, dpi=300); plt.close()
            print(f"[INFO] Saved combined ROC: {outp}")

    print("All done.")

if __name__ == '__main__':
    main()
