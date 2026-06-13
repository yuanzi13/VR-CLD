#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tcn.py：TCN 时序卷积（PyTorch）三分类 + 非独立 5 折 + t-SNE
- 标签映射：task_label = {1:0, 2:1, 3:2, 4:2}
- 指标与图形：混淆矩阵（含P/R）、多分类ROC（OvR）、各折Acc柱状图、训练/验证Loss&Acc曲线、t-SNE（每折+Overall）
- 早停：基于验证集 val_acc（可改为 val_loss），保存 fold 最优模型到 fold_xx/best.pth

目录结构：
  triple_5K_dependent_68/tcn/tcn_<DATASET>/fold_xx/...

示例：
  python tcn.py --data-root DeepLearning/data_rml --datasets MCI,HC,ALL --epochs 50 --gpu-id 0
"""
import datetime
import json
from pathlib import Path
import itertools
import os, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc, roc_auc_score
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = ['WenQuanYi Micro Hei']
import os, itertools, numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
# ------------------------ 配置 ------------------------
REQUIRED_COLUMNS = [
    'leftEye_gaze_X','leftEye_gaze_Y','leftEye_gaze_Z',
    'leftEye_openness','leftEye_pupil_position_X',
    'leftEye_pupil_position_Y','leftEye_pupil_dilation',
    'rightEye_gaze_X','rightEye_gaze_Y','rightEye_gaze_Z',
    'rightEye_openness','rightEye_pupil_position_X',
    'rightEye_pupil_position_Y','rightEye_pupil_dilation',
    'combinedEye_gaze_X','combinedEye_gaze_Y','combinedEye_gaze_Z'
]

# 三分类标签
STAGE_LABEL = [('1', 0), ('2', 1), ('3', 2), ('4', 2)]
N_CLASSES = 3

# ------------------------ 数据与可视化工具 ------------------------
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

def plot_confusion_matrix(conf_mat, acc, total, side_txt, save_path, auc_value=None):
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
    plt.xticks(ticks, [f'P{i}' for i in range(n)] + ['Precision'], rotation=45, fontsize=10)
    plt.yticks(ticks, [f'T{i}' for i in range(n)] + ['Recall'], fontsize=10)

    thresh = M.max() / 2 if M.size else 0.5
    for i, j in np.ndindex(M.shape):
        v = M[i, j]
        if i < n and j < n:
            p = v / total if total else 0
            s = f'{int(v)}\n({p:.2%})'
        else:
            s = f'{v*100:.2f}%'
        plt.text(j, i, s, ha='center', va='center',
                 color='white' if v > thresh else 'black', fontsize=16)

    ax = plt.gca()
    if auc_value is not None:
        ax.text(1.05, 0.05, f'AUC(macro-OVR): {auc_value:.4f}', transform=ax.transAxes,
                va='top', ha='left', linespacing=1.3, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    ax.text(1.05, 0.15, side_txt, transform=ax.transAxes,
            va='top', ha='left', linespacing=1.3, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    
    plt.ylabel('真实标签', fontsize=14)
    plt.xlabel('预测标签', fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300); plt.close()

def compute_macro_roc(y_true, y_proba, n_classes):
    """
    计算多类 macro-average ROC 的共同 fpr (all_fpr)、mean_tpr、AUC
    返回 (all_fpr, mean_tpr, macro_auc)
    """
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fpr, tpr = {}, {}
    for c in range(n_classes):
        try:
            fpr[c], tpr[c], _ = roc_curve(y_bin[:, c], y_proba[:, c])
        except Exception:
            fpr[c], tpr[c] = np.array([0.,1.]), np.array([0.,1.])
    all_fpr = np.unique(np.concatenate([fpr[c] for c in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for c in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[c], tpr[c])
    mean_tpr /= n_classes
    try:
        macro_auc = auc(all_fpr, mean_tpr)
    except Exception:
        macro_auc = float('nan')
    return all_fpr, mean_tpr, macro_auc

def plot_roc_multi(y_true, y_proba, n_classes, title, save_path):
    try:
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        fpr, tpr, roc_auc = dict(), dict(), dict()
        for c in range(n_classes):
            fpr[c], tpr[c], _ = roc_curve(y_bin[:, c], y_proba[:, c])
            roc_auc[c] = auc(fpr[c], tpr[c])
        # macro-average
        all_fpr = np.unique(np.concatenate([fpr[c] for c in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for c in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[c], tpr[c])
        mean_tpr /= n_classes
        macro_auc = auc(all_fpr, mean_tpr)

        plt.figure(figsize=(6, 6))
        for c in range(n_classes):
            plt.plot(fpr[c], tpr[c], lw=1.5, label=f'Class {c} AUC={roc_auc[c]:.4f}')
        plt.plot(all_fpr, mean_tpr, lw=2.5, linestyle='--', label=f'Macro AUC={macro_auc:.4f}')
        plt.plot([0, 1], [0, 1], linestyle=':', lw=1)
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title(title, fontsize=14); plt.legend(loc='lower right', fontsize=9)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300); plt.close()
        return macro_auc
    except Exception as e:
        print(f"⚠️ {title} 无法绘制多分类 ROC（原因：{e}），跳过。")
        return None

def plot_acc_curve(accs, labels, save_path):
    plt.figure(figsize=(8, 4))
    plt.bar(labels, accs); plt.ylim(0, 1)
    plt.xlabel('Fold'); plt.ylabel('Accuracy'); plt.title('各折准确率')
    for i, v in enumerate(accs):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
    plt.tight_layout(); os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300); plt.close()

def plot_train_val_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    epochs = np.arange(1, len(train_losses)+1)
    plt.figure(figsize=(8, 6))

    # Loss
    plt.subplot(2,1,1)
    plt.plot(epochs, train_losses, label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Train/Val Loss')
    plt.legend(); plt.grid(alpha=0.3)

    # Acc
    plt.subplot(2,1,2)
    plt.plot(epochs, train_accs, label='Train Acc', linewidth=2)
    plt.plot(epochs, val_accs, label='Val Acc', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Train/Val Accuracy')
    plt.legend(); plt.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300); plt.close()


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
    d = min(50, feats.shape[1])
    Z = PCA(n_components=d, random_state=random_state).fit_transform(feats)
    emb = TSNE(n_components=2, init='pca', learning_rate=200,
               perplexity=40, n_iter=1500, early_exaggeration=12,
               metric='cosine', random_state=random_state).fit_transform(Z)

    # 3. 计算 SC & DBI
    sc = silhouette_score(emb, labels)
    dbi = davies_bouldin_score(emb, labels)

    # 4. 画图
    colors = {0: '#1f77b4',   # LCL 蓝色
          1: '#2ca02c',   # HCL 绿色 → 改成你想要的颜色
          2: '#d62728'}   # MCL 红色
    name_map = {0: 'LCL', 1: 'MCL', 2: 'HCL'}
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
# ------------------------ 设备 ------------------------
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
    def __init__(self, in_channels, num_classes=N_CLASSES,
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

    def forward(self, x):  # x: (B, C, T)
        z = self.tcn(x)                  # (B, C', T)
        g = self.pool(z).squeeze(-1)     # (B, C')
        return self.head(g)              # (B, num_classes)

    @torch.no_grad()
    def extract_feat(self, x):
        """用于 t-SNE 的判别特征（全局池化后的向量）"""
        z = self.tcn(x)
        g = self.pool(z).squeeze(-1)
        return g                          # (B, C')

# ------------------------ 训练/验证/测试 ------------------------
@torch.no_grad()
def eval_on_loader(model, loader, device, crit):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = crit(logits, yb)
        loss_sum += loss.item() * yb.size(0)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return (loss_sum / max(1, total)), (correct / max(1, total))

def train_with_early_stop(model, device, train_loader, val_loader, epochs, lr, weight_decay,
                          patience, min_delta, monitor='val_acc', ckpt_path=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    best_score = -np.inf if monitor == 'val_acc' else np.inf
    best_state = None
    no_improve = 0

    train_losses, val_losses = [], []
    train_accs,  val_accs  = [], []

    for ep in range(1, epochs + 1):
        # --------- train ---------
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
        tr_acc  = correct / max(1, total)

        # --------- val ---------
        val_loss, val_acc = eval_on_loader(model, val_loader, device, crit)

        train_losses.append(tr_loss); val_losses.append(val_loss)
        train_accs.append(tr_acc);    val_accs.append(val_acc)

        # 监控指标
        if monitor == 'val_acc':
            score = val_acc
            improved = (score - best_score) > min_delta
            comp_better = score > best_score
        else:  # val_loss
            score = val_loss
            improved = (best_score - score) > min_delta
            comp_better = score < best_score

        if improved:
            best_score = score
            no_improve = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            if ckpt_path:
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save(best_state, ckpt_path)
        else:
            no_improve += 1

        if ep == 1 or ep % max(1, epochs//5) == 0 or ep == epochs:
            print(f"  Epoch {ep}/{epochs} | "
                  f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                  f"val_loss={val_loss:.4f} acc={val_acc:.4f} | "
                  f"no_improve={no_improve}/{patience}")

        if no_improve >= patience:
            print(f"  ⏹ 早停触发于 epoch {ep} | monitor={monitor} best={best_score:.4f}")
            break

    # 载入最优
    if best_state is not None:
        model.load_state_dict(best_state)
    elif ckpt_path and os.path.isfile(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

    return train_losses, val_losses, train_accs, val_accs

@torch.no_grad()
def test_predict(model, test_loader, device, collect_feat=False):
    model.eval()
    y_true, y_pred = [], []
    y_proba_rows = []
    feats_rows, labels_rows = [], []

    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        proba = torch.softmax(logits, dim=1).cpu().numpy()
        y_true.extend(yb.numpy().tolist())
        y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
        y_proba_rows.append(proba)

        if collect_feat:
            f = model.extract_feat(xb).cpu().numpy()  # (B, D)
            feats_rows.append(f)
            labels_rows.extend(yb.numpy().tolist())

    y_proba = np.vstack(y_proba_rows) if y_proba_rows else np.zeros((0, N_CLASSES), dtype=float)
    feats = np.vstack(feats_rows) if feats_rows else None
    labels = np.array(labels_rows) if labels_rows else None
    return np.array(y_true), np.array(y_pred), y_proba, feats, labels

# ------------------------ 主流程 ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default=os.path.join('DeepLearning','data_rml'))
    parser.add_argument('--result-dir', type=str, default='triple_5K_dependent_68')
    parser.add_argument('--datasets', default='MCI,HC,ALL', help="MCI,HC,ALL（逗号分隔）")
    parser.add_argument('--window-size', type=int, default=240)
    parser.add_argument('--overlap', type=float, default=0.0)

    # 训练超参
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)

    # 早停
    parser.add_argument('--patience', type=int, default=80)
    parser.add_argument('--min-delta', type=float, default=1e-4)
    parser.add_argument('--monitor', type=str, default='val_acc', choices=['val_acc','val_loss'])

    # TCN 架构
    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--channels', type=str, default='64,64,128,128', help='逗号分隔的通道数列表')

    # 设备
    parser.add_argument('--gpu-id', type=int, default=None)

    args = parser.parse_args()

        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.result_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"run_log_{timestamp}.txt"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("本次运行改动与关键超参：\n")
        f.write(f"- 先整段窗口化，再按时间顺序均分为 5 折（余数分配给前面）\n")
        f.write(f"- 训练集划分出验证集，用于早停；best.pt 在测试集评估\n")
        f.write(f"- 早停监控指标：{args.monitor}（patience={args.patience}）\n")
        f.write(f"- Adam 学习率：{args.lr}\n")
        f.write(f"- Epochs 上限：{args.epochs}（结合早停）\n")
        f.write(f"- 窗口大小：{args.window_size}，overlap：{args.overlap}\n")
        f.write(f"- 输出根目录：{args.result_dir}\n\n")

        f.write("关键命令行参数：\n")
        f.write(
            f"  --model TCN --epochs {args.epochs} --batch-size {args.batch_size}\n"
            f"  --window-size {args.window_size} --overlap {args.overlap}\n"
            f"  --lr {args.lr} --early-metric {args.monitor} --patience {args.patience}\n"
            f"  --data-root {args.data_root}\n\n"
        )

        # (可选) 把当前所有参数都写进去，便于复现实验
        f.write("全部参数（自动导出）：\n")
        f.write(json.dumps(vars(args), indent=4, ensure_ascii=False))
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

    # 用于合并绘制 MCI/HC/ALL overall macro ROC
    combined_overall = {}

    for dtype in datasets_req:
        subjects = get_subject_list(dtype)
        folds = build_5fold_windows(args.data_root, subjects, args.window_size, args.overlap, n_folds=5)

        for k in range(5):
            yk = np.array(folds[k]['Y'])
            print(f"[DEBUG][{dtype}] Fold {k+1}: 标签计数 {dict(zip(*np.unique(yk, return_counts=True))) if yk.size>0 else '空'}")

        res_dir = os.path.join(args.result_dir, 'tcn_3', f"tcn_{dtype}")
        os.makedirs(res_dir, exist_ok=True)

        total_conf = np.zeros((N_CLASSES, N_CLASSES), int)
        y_t_all, y_p_all = [], []
        y_pb_all = []
        accs, fold_labels = [], []

        # 用于 overall t-SNE 的累积
        overall_feats, overall_labels = [], []

        for k in range(5):
            Xte_list, Yte_list = folds[k]['X'], folds[k]['Y']
            Xtr_list, Ytr_list = [], []
            for j in range(5):
                if j == k: continue
                Xtr_list += folds[j]['X']
                Ytr_list += folds[j]['Y']

            print(f"[TCN][{dtype}] Fold {k+1}/5 | train={len(Ytr_list)} windows, test={len(Yte_list)} windows")
            if not Xtr_list or not Xte_list:
                print(f"⚠️ Fold {k+1} 数据不足，跳过"); continue

            # 扁平 → 标准化 → 还原成 (B, C, W)
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

            # 10% 验证
            full_ds = TensorDataset(Xtr_tensor, Ytr_tensor)
            n_val = max(1, int(0.1*len(full_ds)))
            n_trn = len(full_ds) - n_val
            trn_ds, val_ds = random_split(full_ds, [n_trn, n_val])
            train_loader = DataLoader(trn_ds, batch_size=args.batch_size, shuffle=True)
            val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
            test_loader  = DataLoader(TensorDataset(Xte_tensor, Yte_tensor), batch_size=args.batch_size, shuffle=False)

            model = TCN(in_channels=C, num_classes=N_CLASSES,
                        channels=chan_list, kernel_size=args.kernel_size, dropout=args.dropout).to(device)

            fold_dir = os.path.join(res_dir, f"fold_{k+1:02d}")
            os.makedirs(fold_dir, exist_ok=True)
            ckpt_path = os.path.join(fold_dir, "best.pth")

            # 训练 + 早停
            train_losses, val_losses, train_accs, val_accs = train_with_early_stop(
                model, device, train_loader, val_loader,
                epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                patience=args.patience, min_delta=args.min_delta,
                monitor=args.monitor, ckpt_path=ckpt_path
            )

            # 曲线图
            plot_train_val_curves(train_losses, val_losses, train_accs, val_accs,
                                  os.path.join(fold_dir, "loss_acc_curve.png"))

            # 测试评估（已加载最优） + 收集特征用于 t-SNE
            y_true, y_pred, y_proba, feats_fold, labels_fold = test_predict(
                model, test_loader, device, collect_feat=True
            )

            cm  = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
            acc = accuracy_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
            pre = precision_score(y_true, y_pred, average='macro', zero_division=0)
            f1s = f1_score(y_true, y_pred, average='macro', zero_division=0)

            macro_auc = None
            if len(y_proba):
                macro_auc = plot_roc_multi(y_true, y_proba, N_CLASSES,
                                           f"{dtype} TCN Fold {k+1} ROC (OvR)",
                                           os.path.join(fold_dir, "roc.png"))

            side_txt = (f"{dtype} TCN Fold {k+1}\n"
                        f"Acc={acc:.4f}\n"
                        f"Macro-Rec={rec:.4f}  Macro-Pre={pre:.4f}\n"
                        f"Macro-F1={f1s:.4f}")
            plot_confusion_matrix(cm, acc, len(y_true), side_txt,
                                  os.path.join(fold_dir, "confusion.png"),
                                  auc_value=macro_auc)

            # ===== t-SNE：每折 =====
            if feats_fold is not None and labels_fold is not None and len(labels_fold) > 0:
                plot_tsne(feats_fold, labels_fold,
                          save_path=os.path.join(fold_dir, "tsne.png"),
                          title=f"{dtype} TCN Fold {k+1} t-SNE", dpi=300)
                overall_feats.append(feats_fold)
                overall_labels.extend(labels_fold.tolist())

            total_conf += cm
            y_t_all += y_true.tolist()
            y_p_all += y_pred.tolist()
            if len(y_proba):
                y_pb_all.append(y_proba)
            accs.append(acc); fold_labels.append(str(k+1))

        # 各折准确率柱状图
        if accs:
            plot_acc_curve(accs, fold_labels, os.path.join(res_dir, "accuracy_across_folds.png"))

        # Overall 指标 + 图
        if y_t_all:
            y_t_all = np.array(y_t_all); y_p_all = np.array(y_p_all)
            oa  = accuracy_score(y_t_all, y_p_all)
            orc = recall_score(y_t_all, y_p_all, average='macro', zero_division=0)
            opc = precision_score(y_t_all, y_p_all, average='macro', zero_division=0)
            of1 = f1_score(y_t_all, y_p_all, average='macro', zero_division=0)

            oauc = None
            if y_pb_all:
                y_pb_all = np.vstack(y_pb_all)
                # 计算并画 overall 多分类 ROC（并得到 macro AUC）
                oauc = plot_roc_multi(y_t_all, y_pb_all, N_CLASSES,
                                      f"{dtype} TCN Overall ROC (OvR)",
                                      os.path.join(res_dir, "roc_overall.png"))

            otxt = (f"{dtype} TCN Overall\n"
                    f"Acc={oa:.4f}\n"
                    f"Macro-Rec={orc:.4f}  Macro-Pre={opc:.4f}\n"
                    f"Macro-F1={of1:.4f}")
            plot_confusion_matrix(total_conf, oa, len(y_t_all), otxt,
                                  os.path.join(res_dir, "confusion_overall.png"),
                                  auc_value=oauc)

            # 保存用于合并 ROC 的原始数据与 macro ROC 曲线信息
            try:
                if y_pb_all is not None and len(y_pb_all) > 0:
                    all_fpr, mean_tpr, macro_auc = compute_macro_roc(np.array(y_t_all), y_pb_all, N_CLASSES)
                    combined_overall[dtype] = {
                        'all_fpr': all_fpr,
                        'mean_tpr': mean_tpr,
                        'macro_auc': macro_auc
                    }
            except Exception as e:
                print(f"⚠️ 保存 {dtype} 用于合并 ROC 失败：{e}")

        # ===== Overall t-SNE =====
        if overall_feats and len(overall_labels) > 0:
            feats_all = np.vstack(overall_feats)
            feats_all = np.nan_to_num(feats_all, copy=False)
            plot_tsne(feats_all, np.array(overall_labels),
                      save_path=os.path.join(res_dir, "tsne_overall.png"),
                      title=f"{dtype} TCN Overall t-SNE", dpi=300)

    # ===== 合并绘制 MCI / HC / ALL macro ROC（如果有） =====
    if combined_overall:
        plt.figure(figsize=(7.2,6))
        plotted = False
        color_map = {'MCI':'#1f77b4','HC':'#2ca02c','ALL':'#d62728'}
        for name in ['MCI','HC','ALL']:
            info = combined_overall.get(name)
            if info is None: continue
            all_fpr = info['all_fpr']; mean_tpr = info['mean_tpr']; macro_auc = info['macro_auc']
            try:
                plt.plot(all_fpr, mean_tpr, lw=2, label=f'{name} (AUC={macro_auc:.4f})', color=color_map.get(name))
                plotted = True
            except Exception as e:
                print(f"⚠️ 合并 ROC 绘制 {name} 失败：{e}")
        if plotted:
            plt.plot([0,1],[0,1], linestyle='--', color='k', lw=1)
            plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
            plt.title('Overall macro-average ROC Comparison: MCI / HC / ALL')
            plt.legend(loc='lower right', fontsize=10)
            outp = os.path.join(args.result_dir, "roc_MCI_HC_ALL_combined.png")
            os.makedirs(os.path.dirname(outp), exist_ok=True)
            plt.tight_layout(); plt.savefig(outp, dpi=300); plt.close()
            print(f"[INFO] Saved combined ROC: {outp}")

    print("All done.")

if __name__ == '__main__':
    main()
