#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCN 三分类 + LOSO + t-SNE 可视化（含：比较 ROC（仅 LCL vs HCL） & t-SNE 类间距离）
输出目录：triple_LOSO_68/TCN3/<MCI|HC|ALL>/
每折：loss/acc曲线、best.pth、混淆矩阵/ROC、t-SNE
Overall：混淆矩阵/ROC、t-SNE
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import euclidean
from sklearn.metrics import silhouette_score, davies_bouldin_score
import os, argparse
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import datetime
from pathlib import Path
import json
rcParams['font.family'] = ['WenQuanYi Micro Hei']
rcParams['figure.dpi'] = 120

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

# ------------------------ 标签映射（三分类） ------------------------
# 1 → 0，2 → 1，3/4 → 2
STAGE_LABEL = [('1', 0), ('2', 1), ('3', 2), ('4', 2)]
# 类别名称映射（用于 t-SNE 图例与说明）
CLASS_NAMES = {0: 'LCL', 1: 'HCL', 2: 'MCL'}  # note: user requested 0=LCL;1=HCL in t-SNE labeling

# ------------------------ 工具函数 ------------------------
def windowize_from_array(arr_ch_t: np.ndarray, label: int, window_size: int = 240, overlap: float = 0.0):
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
def get_subject_list(dataset_type):

    # 你指定要保留的 HC 编号（26个）
    HC_KEEP = [3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,31,32,35,36]
    # MCI 全部 26 个保留
    MCI_ALL = list(range(1, 26+1))
    
    # if dataset_type == 'MCI':
    #     return [('MCI', i) for i in MCI_ALL]
    # if dataset_type == 'HC':
    #     return [('HC', i) for i in HC_KEEP]
    # if dataset_type == 'ALL':
    #     # ALL = 26 MCI + 26 HC（你指定的）
    #     return [('MCI', i) for i in MCI_ALL] + [('HC', i) for i in HC_KEEP]
    # raise ValueError(dataset_type)
    if dataset_type == 'MCI':
        return [('MCI', i) for i in range(1, 26+1)]
    if dataset_type == 'HC':
        return [('HC', i) for i in HC_KEEP]
    if dataset_type == 'ALL':
        return [('MCI', i) for i in range(1, 26+1)] + [('HC', i) for i in HC_KEEP]
    raise ValueError(dataset_type)
    
def build_loso_subject_windows(data_root, subjects, window_size, overlap):
    subj_data = {}
    for (pop, num) in subjects:
        X_subj, Y_subj = [], []
        any_file = False
        for stage, lab in STAGE_LABEL:
            csv_path = os.path.join(data_root, pop, stage, f'{num}.csv')
            if not os.path.isfile(csv_path):
                continue
            any_file = True
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f'读取失败: {csv_path} | {e}')
                continue
            cols = [c for c in REQUIRED_COLUMNS if c in df.columns]
            if not cols:
                print(f'⚠️ 无 REQ 列：{csv_path}')
                continue
            arr = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype('float32')
            if arr.shape[0] > arr.shape[1]:  # (T,C) -> (C,T)
                arr = arr.T
            if arr.shape[1] == 0:
                continue
            Xw, Yw = windowize_from_array(arr, lab, window_size, overlap)
            X_subj.extend(Xw); Y_subj.extend(Yw)
        if any_file and len(Y_subj) > 0:
            subj_data[(pop, num)] = {'X': X_subj, 'Y': Y_subj}

    # 调试统计
    for key, v in subj_data.items():
        pop, num = key
        yk = np.array(v['Y'])
        cnt = dict(zip(*np.unique(yk, return_counts=True))) if yk.size > 0 else {}
        print(f"[DEBUG] Subject {pop}-{num:02d}: 窗口={len(v['Y'])} | 标签计数={cnt}")
    return subj_data

def plot_confusion_matrix(conf_mat, acc, total, side_txt, save_path):
    n = conf_mat.shape[0]
    M = np.zeros((n + 1, n + 1), dtype=float)
    M[:n, :n] = conf_mat
    for i in range(n):
        tp = conf_mat[i, i]
        cs = conf_mat[:, i].sum()
        rs = conf_mat[i, :].sum()
        M[n, i] = tp / cs if cs else 0  # Precision_i
        M[i, n] = tp / rs if rs else 0  # Recall_i
    M[n, n] = acc

    plt.figure(figsize=(8, 6))
    plt.imshow(M, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    ticks = np.arange(n + 1)
    plt.xticks(ticks, [f'P{i}' for i in range(n)] + ['Precision'], rotation=45, fontsize=12)
    plt.yticks(ticks, [f'T{i}' for i in range(n)] + ['Recall'], fontsize=12)

    thresh = M.max() / 2 if M.size > 0 else 0.5
    for i, j in np.ndindex(M.shape):
        v = M[i, j]
        if i < n and j < n:
            p = v / total if total else 0
            s = f'{int(v)}\n({p:.2%})'
        else:
            s = f'{v*100:.2f}%'
        plt.text(j, i, s, ha='center', va='center',
                 color='white' if v > thresh else 'black', fontsize=12)

    ax = plt.gca()
    ax.text(1.05, 0.05, side_txt, transform=ax.transAxes,
            va='top', ha='left', linespacing=1.3, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    plt.ylabel('真实标签', fontsize=14)
    plt.xlabel('预测标签', fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300); plt.close()

def plot_multiclass_roc(y_true, y_prob, n_classes, title, save_path, include_micro=False):
    """
    多分类 ROC：每个类别的一对多(OVR)曲线 + 宏平均(macro)曲线（不画 micro，除非 include_micro=True）
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # one-hot
    Y = np.zeros((len(y_true), n_classes), dtype=int)
    Y[np.arange(len(y_true)), y_true] = 1

    fpr, tpr, aucs = {}, {}, {}

    # 每类 OVR
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y[:, i], y_prob[:, i])
        pos = Y[:, i].sum()
        aucs[i] = roc_auc_score(Y[:, i], y_prob[:, i]) if (pos not in (0, len(Y))) else np.nan

    # macro：在所有唯一 fpr 上插值求平均
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    aucs["macro"] = roc_auc_score(Y, y_prob, average='macro', multi_class='ovr')

    # （可选）micro
    if include_micro:
        fpr["micro"], tpr["micro"], _ = roc_curve(Y.ravel(), y_prob.ravel())
        aucs["micro"] = roc_auc_score(Y, y_prob, average='micro', multi_class='ovr')

    # 绘图
    plt.figure(figsize=(6.4, 6.0))
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    for i in range(n_classes):
        if not np.isnan(aucs[i]):
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC={aucs[i]:.3f})', color=colors[i % len(colors)])
    # 必画 macro
    plt.plot(fpr["macro"], tpr["macro"], linestyle='--', label=f'macro (AUC={aucs["macro"]:.3f})', color='k')
    # 需要 micro 再画
    if include_micro:
        plt.plot(fpr["micro"], tpr["micro"], linestyle='--', label=f'micro (AUC={aucs["micro"]:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0, 1]); plt.ylim([0, 1])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(title); plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300); plt.close()

def plot_curve(xs, tr, va, ylabel, save_path):
    plt.figure(figsize=(7,4))
    plt.plot(xs, tr, label='Train')
    plt.plot(xs, va, label='Val')
    plt.xlabel('Epoch'); plt.ylabel(ylabel); plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300); plt.close()
# ---------- t-SNE：先 PCA 再 t-SNE；计算类心间距、SC、DBI，所有文字放图外 ----------
def plot_tsne(feats, labels, save_path, title=None, dpi=300,
              per_class=1500, random_state=42, point_size=40, alpha=0.6):

    feats = np.asarray(feats)
    labels = np.asarray(labels)
    feats = np.nan_to_num(feats, copy=False)
    n = feats.shape[0]
    if n < 3:
        plt.figure(figsize=(6, 4))
        plt.axis('off')
        msg = f"{title or 't-SNE'}\n(n={n} < 3, skipped)"
        plt.text(0.5, 0.5, msg, ha='center', va='center', fontsize=12)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi)
        plt.close()
        return

    # 各类等量抽样
    rng = np.random.RandomState(random_state)
    idx = []
    for c in np.unique(labels):
        I = np.where(labels == c)[0]
        take = min(per_class, len(I))
        if take > 0:
            idx.extend(rng.choice(I, take, replace=False))
    idx = np.array(idx)
    if len(idx) < 3:
        plt.figure(figsize=(6, 4))
        plt.axis('off')
        msg = f"{title or 't-SNE'}\n(samples={len(idx)} < 3, skipped)"
        plt.text(0.5, 0.5, msg, ha='center', va='center', fontsize=12)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi)
        plt.close()
        return

    feats_sub, labels_sub = feats[idx], labels[idx]

    # PCA -> t-SNE
    d = min(50, feats_sub.shape[1])
    Z = PCA(n_components=d, random_state=random_state).fit_transform(feats_sub)
    tsne = TSNE(n_components=2, init='pca', learning_rate=200,
                perplexity=min(40, max(5, len(labels_sub) // 50)),
                n_iter=1500, early_exaggeration=12, metric='cosine', random_state=random_state)
    emb = tsne.fit_transform(Z)

    # 颜色与名称
    colors = {0: '#1f77b4', 1: '#2ca02c', 2: '#d62728'}
    name_map = {0: 'LCL', 1: 'HCL', 2: 'MCL'}
    plt.figure(figsize=(6.8, 5.2))
    ax = plt.gca()
    centroids = {}
    for c in np.unique(labels_sub):
        m = labels_sub == c
        ax.scatter(emb[m, 0], emb[m, 1], s=point_size, alpha=alpha,
                   c=colors.get(int(c), '#7f7f7f'), edgecolors='none',
                   label=f'{name_map.get(int(c), str(int(c)))} (n={m.sum()})')
        cx, cy = emb[m, 0].mean(), emb[m, 1].mean()
        centroids[int(c)] = (cx, cy)

    # 类间距离
    dist_lines = []
    classes_present = sorted(list(centroids.keys()))
    for i in range(len(classes_present)):
        for j in range(i + 1, len(classes_present)):
            a, b = classes_present[i], classes_present[j]
            dval = euclidean(centroids[a], centroids[b])
            dist_lines.append(f"d({name_map[a]}-{name_map[b]}) = {dval:.3f}")

    # SC & DBI
    sc = silhouette_score(emb, labels_sub)
    dbi = davies_bouldin_score(emb, labels_sub)

    # 图外右侧文本框
    info_box = (f"{title or 't-SNE'}\n"
                f"SC  = {sc:.3f}\n"
                f"DBI = {dbi:.3f}\n" +
                "\n".join(dist_lines))
    ax.text(1.02, 0.98, info_box, transform=ax.transAxes,
            ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    if title:
        plt.title(title)
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.legend(loc='best', framealpha=0.9, fontsize=9, title='Class')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    
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

# ------------------------ 模型 ------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super().__init__(); self.chomp_size = chomp_size
    def forward(self, x):  # x: (B, C, T_pad)
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout):
        super().__init__();
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
    def __init__(self, in_channels, num_classes=3,
                 channels=[64, 64, 128, 128], kernel_size=3, dropout=0.2):
        super().__init__()
        layers, prev = [], in_channels
        for i, c in enumerate(channels):
            layers.append(TemporalBlock(prev, c, kernel_size, stride=1,
                                        dilation=2**i, dropout=dropout))
            prev = c
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(prev, num_classes))
    def forward(self, x):  # x: (B, C, T)
        return self.head(self.tcn(x))
    # 取特征（用于 t-SNE）：全局平均池化前后的向量
    def extract_feat(self, x):
        z = self.tcn(x)                              # (B, C', T)
        z = nn.functional.adaptive_avg_pool1d(z, 1)  # (B, C', 1)
        return z.squeeze(-1)                         # (B, C')

# ------------------------ 提前停止 ------------------------
class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.count = 0
    def step(self, val_loss):
        if self.best is None or (self.best - val_loss) > self.min_delta:
            self.best = val_loss
            self.count = 0
            return False
        else:
            self.count += 1
            return self.count >= self.patience

# ------------------------ 单折训练/验证/测试 ------------------------
def run_fold(model, device, train_loader, val_loader, test_loader, epochs, lr, weight_decay, fold_dir):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # crit = nn.CrossEntropyLoss()
        # 权重：LCL(0) 和 HCL(2) 乘 2，MCL(1) 保持 1
    weight = torch.tensor([1.0, 1.0, 0.5], device=device)
    crit = nn.CrossEntropyLoss(weight=weight)

    stopper = EarlyStopper(patience=early_stop_cfg['patience'], min_delta=early_stop_cfg['min_delta'])

    tr_losses, va_losses, tr_accs, va_accs, epochs_list = [], [], [], [], []
    best_state = None
    best_val_loss = float('inf')

    for ep in range(1, epochs + 1):
        # ===== Train =====
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
        tr_loss = loss_sum / max(1, total)
        tr_acc = correct / max(1, total)

        # ===== Validate =====
        model.eval()
        v_total, v_correct, v_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = crit(logits, yb)
                v_loss_sum += loss.item() * yb.size(0)
                v_correct += (logits.argmax(1) == yb).sum().item()
                v_total += yb.size(0)
        va_loss = v_loss_sum / max(1, v_total)
        va_acc = v_correct / max(1, v_total)

        # log
        epochs_list.append(ep)
        tr_losses.append(tr_loss); va_losses.append(va_loss)
        tr_accs.append(tr_acc);   va_accs.append(va_acc)

        print(f"  Epoch {ep}/{epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}")

        # save best by val_loss
        if va_loss < best_val_loss - early_stop_cfg['min_delta']:
            best_val_loss = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(fold_dir, "best.pth"))

        # early stop
        if stopper.step(va_loss):
            print(f"  Early stopped at epoch {ep} (best val_loss={best_val_loss:.4f}).")
            break

    # 曲线 & csv
    log_df = pd.DataFrame({
        'epoch': epochs_list,
        'train_loss': tr_losses, 'val_loss': va_losses,
        'train_acc': tr_accs,   'val_acc': va_accs
    })
    log_df.to_csv(os.path.join(fold_dir, "train_log.csv"), index=False)
    plot_curve(epochs_list, tr_losses, va_losses, 'Loss', os.path.join(fold_dir, "loss_curve.png"))
    plot_curve(epochs_list, tr_accs,  va_accs,  'Accuracy', os.path.join(fold_dir, "acc_curve.png"))

    # load best for testing
    if best_state is not None:
        model.load_state_dict(best_state)

    # ===== Test =====
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    feats_collect, labels_collect = [], []  # for t-SNE
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            prob = torch.softmax(logits, dim=1).cpu().numpy()  # (B, C)
            y_prob.extend(prob.tolist())
            y_true.extend(yb.numpy().tolist())
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())

            # ==== 提取特征用于 t-SNE ====
            f = model.extract_feat(xb)        # (B, C')
            feats_collect.append(f.cpu().numpy())
            labels_collect.extend(yb.numpy().tolist())

    feats_arr = np.vstack(feats_collect) if feats_collect else None
    labels_arr = np.array(labels_collect) if labels_collect else None

    return np.array(y_true), np.array(y_pred), np.array(y_prob), feats_arr, labels_arr

# ------------------------ 比较 ROC（binary LCL vs HCL） ------------------------
def plot_binary_roc_compare(entries, save_path, title="LCL vs HCL ROC Comparison"):
    """
    entries: list of tuples (name, y_true_multi, y_prob_multi)
    对每项会筛选 y_true != 1（剔除 MCL），并以 prob[:,2] 作为 HCL 的概率（正类）
    """
    plt.figure(figsize=(6.4,6.0))
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
    plotted = 0
    for i, (name, y_true_all, y_prob_all) in enumerate(entries):
        y_true_all = np.asarray(y_true_all)
        y_prob_all = np.asarray(y_prob_all)
        # 仅保留 LCL(0) 与 HCL(2)
        mask = (y_true_all == 0) | (y_true_all == 2)
        if mask.sum() == 0:
            print(f"  ⚠️ {name} 在比较 ROC 时没有 LCL/HCL 样本，跳过")
            continue
        y_true_bin = (y_true_all[mask] == 2).astype(int)  # 1 表示 HCL
        # 取 HCL 的概率（原 prob 对应 class index 2）
        if y_prob_all.ndim == 2 and y_prob_all.shape[1] >= 3:
            y_score = y_prob_all[mask, 2]
        else:
            # 如果没有多类概率，则跳过
            print(f"  ⚠️ {name} 没有多类概率列，跳过")
            continue
        try:
            fpr, tpr, _ = roc_curve(y_true_bin, y_score)
            aucv = roc_auc_score(y_true_bin, y_score)
            plt.plot(fpr, tpr, lw=2, color=colors[plotted % len(colors)], label=f"{name} (AUC={aucv:.3f})")
            plotted += 1
        except Exception as e:
            print(f"  ⚠️ {name} ROC 绘制失败: {e}")
    if plotted == 0:
        print("  ⚠️ 没有任何组满足 LCL/HCL 比较条件，未绘制比较 ROC。")
        return
    plt.plot([0,1],[0,1],'k--', lw=1)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right', fontsize=9)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300); plt.close()

# ------------------------ 主流程 ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default=os.path.join('DeepLearning','data_rml'))
    parser.add_argument('--result-dir', type=str, default='triple_LOSO_68/TCN3')
    parser.add_argument('--datasets', default='MCI,HC,ALL', help="MCI,HC,ALL（逗号分隔）")
    parser.add_argument('--window-size', type=int, default=240)
    parser.add_argument('--overlap', type=float, default=0.0)

    # 训练超参
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)

    # 早停
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min-delta', type=float, default=0.0)

    # TCN 架构
    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--channels', type=str, default='64,64,128,128', help='逗号分隔的通道数列表')

    # 设备
    parser.add_argument('--gpu-id', type=int, default=None)
    args = parser.parse_args()
        
    # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 日志文件放在 result-dir 内
    log_dir = Path(args.result_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"run_log_{timestamp}.txt"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("本次运行关键设置与改动：\n")
        f.write(f"- 采用 LOSO（Leave-One-Subject-Out）评估方案\n")
        f.write(f"- 完整序列 → 窗口化(window={args.window_size}, overlap={args.overlap})\n")
        f.write(f"- Early Stopping: patience={args.patience}, min_delta={args.min_delta}（监控验证损失或准确率）\n")
        f.write(f"- 优化器：Adam(lr={args.lr}, weight_decay={args.weight_decay})\n")
        f.write(f"- Epoch 上限：{args.epochs}\n")
        f.write(f"- Batch Size：{args.batch_size}\n")
        f.write(f"- TCN 架构：kernel={args.kernel_size}, dropout={args.dropout}, channels={args.channels}\n")
        f.write(f"- 数据集选择：{args.datasets}\n")
        f.write(f"- 数据根目录：{args.data_root}\n")
        f.write(f"- 输出根目录：{args.result_dir}\n")
        f.write(f"- GPU ID：{args.gpu_id}\n\n")

        f.write("关键命令行参数示例：\n")
        f.write(
            f"  --epochs {args.epochs} --batch-size {args.batch_size}\n"
            f"  --window-size {args.window_size} --overlap {args.overlap}\n"
            f"  --lr {args.lr} --weight-decay {args.weight_decay}\n"
            f"  --patience {args.patience} --min-delta {args.min_delta}\n"
            f"  --kernel-size {args.kernel_size} --dropout {args.dropout}\n"
            f"  --channels {args.channels}\n"
            f"  --datasets {args.datasets}\n"
            f"  --data-root {args.data_root}\n"
        )

        f.write("\n全部参数（自动导出以便复现）：\n")
        f.write(json.dumps(vars(args), indent=4, ensure_ascii=False))

    print(f"日志已写入：{log_path}")
    # ======================================================================
    # 兼容 DeepLearning/data_rml
    if not os.path.isdir(args.data_root) and os.path.isdir(os.path.join('DeepLearning', 'data_rml')):
        args.data_root = os.path.join('DeepLearning', 'data_rml')

    global early_stop_cfg
    early_stop_cfg = {'patience': args.patience, 'min_delta': args.min_delta}

    device = select_device(args.gpu_id)
    datasets_req = [d.strip().upper() for d in args.datasets.split(',')]
    all_datasets = ['MCI','HC','ALL']
    for d in datasets_req:
        if d not in all_datasets:
            raise ValueError(f'不支持的数据集: {d}（可选: {",".join(all_datasets)}）')

    C = len(REQUIRED_COLUMNS); W = args.window_size
    chan_list = [int(x) for x in args.channels.split(',') if x.strip()]
    num_classes = 3

    # 收集三组 overall 的 LCL/HCL 用于比较 ROC（entries 为 list of (name, y_true_all, y_prob_all)）
    compare_entries = []

    for dtype in datasets_req:
        subjects = get_subject_list(dtype)
        subj_data = build_loso_subject_windows(args.data_root, subjects, args.window_size, args.overlap)
        valid_keys = [k for k, v in subj_data.items() if len(v['Y']) > 0]
        if not valid_keys:
            print(f"⚠️ 数据集 {dtype} 无有效受试者，跳过"); continue

        res_dir = os.path.join(args.result_dir, f"tcn_{dtype}")
        os.makedirs(res_dir, exist_ok=True)

        total_conf = np.zeros((num_classes, num_classes), int)
        y_t_all, y_p_all, y_pr_all = [], [], []
        accs, fold_labels = [], []

        # for overall t-SNE
        overall_feats, overall_labels = [], []

        print(f"[TCN-LOSO][{dtype}] 受试者数（有效）：{len(valid_keys)}")

        for key in valid_keys:
            pop, sid = key
            fold_dir = os.path.join(res_dir, f"subj_{pop}_{sid:02d}")
            os.makedirs(fold_dir, exist_ok=True)

            # 构造训练/验证/测试
            Xte_list, Yte_list = subj_data[key]['X'], subj_data[key]['Y']
            Xtr_list, Ytr_list = [], []
            for k2 in valid_keys:
                if k2 == key: continue
                Xtr_list += subj_data[k2]['X']
                Ytr_list += subj_data[k2]['Y']

            print(f"[TCN][{dtype}] LOSO {pop}-{sid:02d} | train={len(Ytr_list)} windows, test={len(Yte_list)} windows")
            if not Xtr_list or not Xte_list:
                print(f"⚠️ Subject {pop}-{sid:02d} 数据不足，跳过"); continue

            Xtr_flat = np.vstack([x.numpy() for x in Xtr_list])
            Xte_flat = np.vstack([x.numpy() for x in Xte_list])
            Ytr = np.array(Ytr_list, dtype=np.int64); Yte = np.array(Yte_list, dtype=np.int64)

            scaler = StandardScaler().fit(Xtr_flat)
            Xtr_s = scaler.transform(Xtr_flat).reshape(-1, C, W)
            Xte_s = scaler.transform(Xte_flat).reshape(-1, C, W)

            Xtr_tensor = torch.tensor(Xtr_s, dtype=torch.float32)
            Xte_tensor = torch.tensor(Xte_s, dtype=torch.float32)
            Ytr_tensor = torch.tensor(Ytr, dtype=torch.long)
            Yte_tensor = torch.tensor(Yte, dtype=torch.long)

            # 10% 验证集
            full_ds = TensorDataset(Xtr_tensor, Ytr_tensor)
            n_val = max(1, int(0.1*len(full_ds)))
            n_trn = len(full_ds) - n_val
            trn_ds, val_ds = random_split(full_ds, [n_trn, n_val], generator=torch.Generator().manual_seed(42))
            train_loader = DataLoader(trn_ds, batch_size=args.batch_size, shuffle=True)
            val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
            test_loader  = DataLoader(TensorDataset(Xte_tensor, Yte_tensor), batch_size=args.batch_size, shuffle=False)

            model = TCN(in_channels=C, num_classes=num_classes,
                        channels=chan_list, kernel_size=args.kernel_size, dropout=args.dropout)

            y_true, y_pred, y_prob, feats_fold, labels_fold = run_fold(
                model, device, train_loader, val_loader, test_loader,
                epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                fold_dir=fold_dir
            )

            # ===== 评估与图 =====
            cm  = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
            acc = accuracy_score(y_true, y_pred)
            pre = precision_score(y_true, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1s = f1_score(y_true, y_pred, average='macro', zero_division=0)
            try:
                ovr_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            except Exception:
                ovr_auc = 0.0

            side_txt = (f"{dtype} TCN LOSO {pop}-{sid:02d}\n"
                        f"Acc={acc:.4f}\nPre(macro)={pre:.4f}\nRec(macro)={rec:.4f}\n"
                        f"F1(macro)={f1s:.4f}\nAUC(OVR)={ovr_auc:.4f}")
            plot_confusion_matrix(cm, acc, len(y_true), side_txt, os.path.join(fold_dir, "confusion.png"))

            # —— 正式多分类 ROC（画曲线） —— 
            plot_multiclass_roc(
                y_true, y_prob, n_classes=num_classes,
                title=f"{dtype} TCN LOSO {pop}-{sid:02d} ROC (OVR)",
                save_path=os.path.join(fold_dir, "roc.png"),
                include_micro=False
            )

            # ===== t-SNE：每折 =====
            if feats_fold is not None and len(labels_fold) > 0:
                plot_tsne(feats_fold, labels_fold,
                          save_path=os.path.join(fold_dir, "tsne.png"),
                          title=f"{dtype} TCN LOSO {pop}-{sid:02d} t-SNE", dpi=300)
                overall_feats.append(feats_fold)
                overall_labels.extend(labels_fold.tolist())

            total_conf += cm
            y_t_all += y_true.tolist()
            y_p_all += y_pred.tolist()
            y_pr_all += y_prob.tolist()
            accs.append(acc); fold_labels.append(f"{pop}-{sid:02d}")

        # 每受试者准确率柱状图
        if accs:
            plt.figure(figsize=(10,4))
            plt.bar(fold_labels, accs); plt.ylim(0, 1)
            plt.xlabel('Subject'); plt.ylabel('Accuracy'); plt.title('LOSO 每折准确率')
            for i, v in enumerate(accs): plt.text(i, v + 0.02, f'{v:.3f}', ha='center', rotation=90)
            plt.tight_layout(); plt.savefig(os.path.join(res_dir, "accuracy_across_subjects.png"), dpi=300); plt.close()

        # Overall 指标 & 混淆矩阵 & ROC
        if y_t_all:
            oa  = accuracy_score(y_t_all, y_p_all)
            op  = precision_score(y_t_all, y_p_all, average='macro', zero_division=0)
            orc = recall_score(y_t_all, y_p_all, average='macro', zero_division=0)
            of1 = f1_score(y_t_all, y_p_all, average='macro', zero_division=0)
            try:
                oauc = roc_auc_score(np.array(y_t_all), np.array(y_pr_all), multi_class='ovr', average='macro')
            except Exception:
                oauc = 0.0
            otxt = (f"{dtype} TCN LOSO Overall\n"
                    f"Acc={oa:.4f}\nPre(macro)={op:.4f}\nRec(macro)={orc:.4f}\n"
                    f"F1(macro)={of1:.4f}\nAUC(OVR)={oauc:.4f}")
            plot_confusion_matrix(total_conf, oa, len(y_t_all), otxt, os.path.join(res_dir, "confusion_overall.png"))

            # —— Overall 多分类 ROC（画曲线） —— 
            plot_multiclass_roc(
                np.array(y_t_all), np.array(y_pr_all), n_classes=num_classes,
                title=f"{dtype} TCN LOSO Overall ROC (OVR)",
                save_path=os.path.join(res_dir, "roc_overall.png"),
                include_micro=False
            )

            # 将本组 overall 的 multi-class truth/prob 保存到比较队列（用于 LCL vs HCL 比较）
            compare_entries.append((dtype, np.array(y_t_all), np.array(y_pr_all)))

        # ===== Overall t-SNE =====
        if len(overall_labels) > 0:
            feats_all = np.vstack(overall_feats)
            feats_all = np.nan_to_num(feats_all, copy=False)
            plot_tsne(feats_all, np.array(overall_labels),
                      save_path=os.path.join(res_dir, "tsne_overall.png"),
                      title=f"{dtype} TCN LOSO Overall t-SNE", dpi=300)

    # ===== 三组比较 ROC（仅 LCL vs HCL） =====
    # entries 格式 (name, y_true_multi, y_prob_multi)
    if compare_entries:
        # 输出到 result-dir 根下一个对比图
        comp_save = os.path.join(args.result_dir, "roc_compare_LCL_vs_HCL.png")
        plot_binary_roc_compare(compare_entries, comp_save, title="LCL vs HCL ROC: MCI / HC / ALL")
        print(f"Saved comparison ROC (LCL vs HCL) to: {comp_save}")
    else:
        print("No overall entries collected for ROC comparison across groups.")

    print("All done.")

if __name__ == '__main__':
    main()
