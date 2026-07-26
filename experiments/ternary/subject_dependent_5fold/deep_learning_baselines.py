#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MIX_2.py：四模型对比（MLP / LSTM / Transformer / CNN）非独立 5 折实验（三分类版）
-----------------------------------------------------------------------
· 标签规则（更新为三分类）：
  stage: 1 → label=0；2 → label=1；3/4 → label=2

· 其他保持：先窗口后五折；训练集再划10%验证；早停/达标即停；默认 epochs=5000
· 指标与可视化更新为 3 类（宏平均 precision/recall/f1；多类 ROC-AUC（ovr））
· 输出：
  Triple_5K_dependent_68/MIX_2/<MODEL>/<GROUP>/fold_k/
    - train_val_curves.png
    - confusion.png（每折 3×3 主体 + 末行/末列为 Recall/Precision，总体 Acc 在右下）
    - tsne.png
  以及 overall 的 confusion_overall.png / tsne_overall.png

  CUDA_VISIBLE_DEVICES=5,6,7 bash -c '
  CUDA_VISIBLE_DEVICES=5 python triple_5K_dependent_68/MIX_2.py --model MLP --epochs 1000 &
  CUDA_VISIBLE_DEVICES=6 python triple_5K_dependent_68/MIX_2.py --model LSTM --epochs 1000 &
  CUDA_VISIBLE_DEVICES=7 python triple_5K_dependent_68/MIX_2.py --model Transformer --epochs 1000 &
  CUDA_VISIBLE_DEVICES=5 python triple_5K_dependent_68/MIX_2.py --model CNN --epochs 1000 &
  wait
'
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib
matplotlib.use("Agg")  # 无显示环境
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = ['WenQuanYi Micro Hei']
rcParams['figure.dpi'] = 120

# ───────────────── 工具：安全选 GPU（兼容 CUDA_VISIBLE_DEVICES） ─────────────────
def select_device(requested_id, model_name):
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU.")
        return torch.device('cpu')
    visible = torch.cuda.device_count()
    idx = 0 if requested_id is None else int(requested_id)
    if idx < 0 or idx >= visible:
        print(f"⚠️ requested gpu-id={idx} 无效（本进程可见 GPU 数: {visible}），回退到 0。")
        idx = 0
    name = torch.cuda.get_device_name(idx)
    dev = torch.device(f'cuda:{idx}')
    print(f"Using device: {dev} ({name}) for model {model_name} | visible_gpus={visible}")
    return dev

# ───────────────── 窗口化 ─────────────────
def DataToXY_from_ndarray(data_ch_t: np.ndarray, label: int,
                          window_size: int = 240, overlap: float = 0.0):
    X, Y = [], []
    step = window_size - int(window_size * overlap)
    step = max(1, step)
    T = data_ch_t.shape[1]
    if T < window_size:
        return X, Y
    n_seg = (T - window_size) // step + 1
    for i in range(n_seg):
        seg = data_ch_t[:, i*step:i*step+window_size].astype(np.float32)
        X.append(seg); Y.append(label)
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

def build_5fold_windows(subjects, required_columns, data_root, window_size, overlap, n_folds=5):
    folds = [{'X': [], 'Y': []} for _ in range(n_folds)]
    # ★ 三分类映射：1→0，2→1，3/4→2
    stage_label = [('1', 0), ('2', 1), ('3', 2), ('4', 2)]
    for pop, num in subjects:
        for stage, lab in stage_label:
            fpath = os.path.join(data_root, pop, stage, f"{num}.csv")
            if not os.path.isfile(fpath): continue
            try:
                df = pd.read_csv(fpath)
            except Exception as e:
                print(f"⚠️ 读取失败: {fpath} ({e})"); continue
            cols = [c for c in required_columns if c in df.columns]
            if not cols:
                print(f"⚠️ 缺失所需列，跳过: {fpath}"); continue
            arr = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float32)
            if arr.shape[0] > arr.shape[1]:  # (T,C) -> (C,T)
                arr = arr.T
            if arr.shape[1] == 0: continue
            Xs, Ys = DataToXY_from_ndarray(arr, lab, window_size, overlap)
            perfile_folds = split_windows_to_5folds(Xs, Ys, n_folds=n_folds)
            for k in range(n_folds):
                folds[k]['X'].extend(perfile_folds[k]['X'])
                folds[k]['Y'].extend(perfile_folds[k]['Y'])
    return folds

# ───────────────── 模型定义（num_classes 改为 3） ─────────────────
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256], num_classes=3):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.5)]
            prev = h
        layers += [nn.Linear(prev, num_classes)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):  # x: (B, seq, feat)
        B = x.size(0)
        x = x.view(B, -1)
        return self.net(x)
    def extract(self, x):
        B = x.size(0)
        x = x.view(B, -1)
        for layer in self.net[:-1]:
            x = layer(x)
        return x

class LSTMClassifier(nn.Module):
    def __init__(self, feat_dim, hidden_dim=128, num_layers=1, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        return self.fc(h)
    def extract(self, x):
        out, _ = self.lstm(x)
        return out[:, -1, :]

class TransformerClassifier(nn.Module):
    def __init__(self, feat_dim, seq_len,
                 model_dim=32, nhead=4, dim_feedforward=128, num_layers=2, num_classes=3):
        super().__init__()
        self.proj = nn.Linear(feat_dim, model_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, model_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nhead,
            dim_feedforward=dim_feedforward, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, num_classes)
    def forward(self, x):
        x = self.proj(x) + self.pos_embed
        out = self.encoder(x)
        h = out.mean(dim=1)
        return self.fc(h)
    def extract(self, x):
        x = self.proj(x) + self.pos_embed
        out = self.encoder(x)
        return out.mean(dim=1)

class CNNClassifier(nn.Module):
    def __init__(self, in_ch, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, 8, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),   nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),  nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(32, num_classes)
    def forward(self, x):
        x = x.permute(0, 2, 1)     # (B, seq, feat) -> (B, feat, seq)
        out = self.features(x).squeeze(-1)
        return self.fc(out)
    def extract(self, x):
        x = x.permute(0, 2, 1)
        out = self.features(x).squeeze(-1)
        return out

# ───────────────── 绘图工具（通用 N 类混淆矩阵） ─────────────────
def safe_savefig(path, **kw):
    try:
        plt.savefig(path, **kw)
    except OSError as e:
        print(f"[WARN] savefig failed: {e}. Try SVG fallback...")
        alt = path.rsplit('.', 1)[0] + ".svg"
        try:
            plt.savefig(alt)
        except Exception as e2:
            print(f"[WARN] fallback save failed: {e2}")
    finally:
        plt.close()

def plot_train_val_curves(train_losses, val_losses, train_accs, val_accs, save_path, dpi=300):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # 左：Loss
    axes[0].plot(train_losses, label='Train')
    axes[0].plot(val_losses, label='Val')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].legend()
    axes[0].set_title('Loss (Train vs Val)')
    # 右：Acc
    axes[1].plot(train_accs, label='Train')
    axes[1].plot(val_accs, label='Val')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy'); axes[1].legend()
    axes[1].set_title('Accuracy (Train vs Val)')
    fig.tight_layout()
    safe_savefig(save_path, dpi=dpi)

def plot_confusion_matrix_general(cm, metrics, total, title, save_path):
    """
    通用 N 类混淆矩阵可视化：
      - 主体 N×N 为计数（及占总体比例）
      - 末行为 per-class Recall，末列为 per-class Precision
      - 右下角为整体 Accuracy
    """
    cm = np.asarray(cm, dtype=float)
    n = cm.shape[0]
    M = np.zeros((n+1, n+1), dtype=float)
    M[:n, :n] = cm

    # per-class precision / recall
    col_sum = cm.sum(axis=0)  # predicted
    row_sum = cm.sum(axis=1)  # true
    precision = np.divide(np.diag(cm), col_sum, out=np.zeros_like(col_sum), where=col_sum>0)
    recall    = np.divide(np.diag(cm), row_sum, out=np.zeros_like(row_sum), where=row_sum>0)

    # 填入末列（Precision）与末行（Recall）
    M[:n,  n] = precision
    M[ n, :n] = recall
    M[n, n]   = metrics['acc']  # overall acc

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    im = ax.imshow(M, cmap='Blues', interpolation='nearest')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 轴刻度
    ticks = list(range(n)) + [n]
    xticklabels = [f'P{i}' for i in range(n)] + ['Precision']
    yticklabels = [f'T{i}' for i in range(n)] + ['Recall']
    ax.set_xticks(ticks); ax.set_xticklabels(xticklabels, fontsize=12, rotation=45)
    ax.set_yticks(ticks); ax.set_yticklabels(yticklabels, fontsize=12)

    # 文本
    thresh = M[:n, :n].max() / 2 if n>0 else 0.5
    for i in range(n+1):
        for j in range(n+1):
            v = M[i, j]
            if i < n and j < n:
                perc = (v/total) if total else 0.0
                s = f"{int(v)}\n({perc:.2%})"
                color = 'white' if v > thresh else 'black'
                ax.text(j, i, s, ha='center', va='center', color=color, fontsize=18)
            elif i == n and j == n:
                ax.text(j, i, f"{v*100:.2f}%", ha='center', va='center', fontsize=12, fontweight='bold')
            else:
                ax.text(j, i, f"{v*100:.2f}%", ha='center', va='center', fontsize=12)

    ax.set_ylabel('真实标签'); ax.set_xlabel('预测标签')
    ax.set_title(title.split('\n', 1)[0], fontsize=14)

    # 右侧汇总文本（宏平均）
    txt = (f"ACC={metrics['acc']:.4f}\n"
           f"REC(macro)={metrics['rec']:.4f}\n"
           f"PRE(macro)={metrics['pre']:.4f}\n"
           f"F1 (macro)={metrics['f1']:.4f}\n"
           f"AUC(ovr)={metrics['auc']:.4f}")
    fig.text(0.80, 0.50, txt, va='center', ha='left',
             fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    fig.tight_layout(rect=[0.0, 0.0, 0.75, 1.0])
    safe_savefig(save_path, dpi=300)

def plot_tsne(feats, labels, save_path, title=None, dpi=300,
              max_points=8000, random_state=42,
              point_size=30, alpha=0.65,
              draw_centroid=True, draw_hull=False):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    feats = np.asarray(feats); labels = np.asarray(labels)
    feats = np.nan_to_num(feats, copy=False)
    n = feats.shape[0]
    if n == 0: return

    if max_points and n > max_points:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n, max_points, replace=False)
        feats = feats[idx]; labels = labels[idx]; n = feats.shape[0]

    if n < 3:
        plt.figure(figsize=(6,4)); plt.axis('off')
        msg = f"t-SNE skipped (n={n} < 3)"
        if title: msg = f"{title}\n{msg}"
        plt.text(0.5, 0.5, msg, ha='center', va='center', fontsize=12)
        safe_savefig(save_path, dpi=dpi); return

    if n <= 10: perpl = max(2, n//2)
    else:       perpl = max(5, min(30, n//3))
    perpl = min(perpl, n - 1 - 1e-6)

    try:
        tsne2d = TSNE(n_components=2, init='random', learning_rate='auto',
                      perplexity=perpl, random_state=random_state).fit_transform(feats)
        emb = tsne2d; used_fallback = False
    except Exception:
        from sklearn.decomposition import PCA
        emb = PCA(n_components=2).fit_transform(feats)
        used_fallback = True

    classes = np.unique(labels)
    color_map = {0: '#1f77b4', 1: '#2ca02c', 2: '#d62728'}
    handles = []
    plt.figure(figsize=(7,5))
    ax = plt.gca()
    for c in classes:
        mask = labels == c
        sc = plt.scatter(emb[mask,0], emb[mask,1],
                         s=point_size, alpha=alpha,
                         c=color_map.get(int(c), '#7f7f7f'),
                         edgecolors='none', label=f"{int(c)} (n={mask.sum()})")
        handles.append(sc)
        if draw_centroid and mask.sum() > 0:
            cx, cy = emb[mask,0].mean(), emb[mask,1].mean()
            plt.scatter([cx],[cy], s=80, marker='x', linewidths=1.5, c='k', zorder=3)
        if draw_hull and mask.sum() >= 3:
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(emb[mask])
                poly = Polygon(emb[mask][hull.vertices], closed=True,
                               fill=False, linewidth=1.0,
                               edgecolor=color_map.get(int(c), '#7f7f7f'), alpha=0.8)
                ax.add_patch(poly)
            except Exception:
                pass

    if title:
        plt.title(title + (" (PCA fallback)" if used_fallback else ""))
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.legend(handles=handles, title="Class", loc='best', framealpha=0.9)
    plt.tight_layout()
    safe_savefig(save_path, dpi=dpi)

# ───────────────── 主流程 ─────────────────
def main():
    print("Running file:", os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        choices=['MLP', 'LSTM', 'Transformer', 'CNN'], required=True)
    parser.add_argument('--gpu-id', type=int, default=None,
        help='逻辑 GPU 号（受 CUDA_VISIBLE_DEVICES 影响）；不填默认 0')
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--window-size', type=int, default=240)
    parser.add_argument('--overlap', type=float, default=0.0)
    parser.add_argument('--result-dir', type=str, default='triple_5K_dependent_68/MIX_2')
    parser.add_argument('--data-root', type=str, default='data_rml',
                        help="数据根目录（默认 data_rml；也可填绝对路径或 DeepLearning/data_rml）")
    parser.add_argument('--patience', type=int, default=80, help='EarlyStopping patience')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--early-metric', type=str, default='val_loss',
                        choices=['val_loss', 'val_acc'],
                        help='早停监控指标：val_loss(越小越好) 或 val_acc(越大越好)')
    parser.add_argument('--target-acc', type=float, default=None,
                        help='达到该准确率时提前停止；例如 1.0 表示100%准确率。默认关闭')
    parser.add_argument('--acc-scope', type=str, choices=['val', 'train'], default='val',
                        help='--target-acc 监控的准确率来源：val 或 train（默认 val）')
    parser.add_argument('--min-epochs', type=int, default=5,
                        help='达标即停前至少训练的 epoch 数，默认 5')
    args = parser.parse_args()

    # data_root 兼容 DeepLearning/data_rml
    if not os.path.isdir(args.data_root) and os.path.isdir(os.path.join('DeepLearning', 'data_rml')):
        args.data_root = os.path.join('DeepLearning', 'data_rml')

    device = select_device(args.gpu_id, args.model)

    required_columns = [
        'leftEye_gaze_X', 'leftEye_gaze_Y', 'leftEye_gaze_Z',
        'leftEye_openness', 'leftEye_pupil_position_X', 'leftEye_pupil_position_Y', 'leftEye_pupil_dilation',
        'rightEye_gaze_X', 'rightEye_gaze_Y', 'rightEye_gaze_Z',
        'rightEye_openness', 'rightEye_pupil_position_X', 'rightEye_pupil_position_Y', 'rightEye_pupil_dilation',
        'combinedEye_gaze_X', 'combinedEye_gaze_Y', 'combinedEye_gaze_Z'
    ]

    # 受试者列表（MCI=1..26；HC=1..42）
    MCI_subjects = [('MCI', i) for i in range(1, 26+1)]
    HC_subjects  = [('HC',  i) for i in range(1, 42+1)]
    dataset_configs = {
        'MCI': (MCI_subjects, 'MCI'),
        'HC':  (HC_subjects,  'HC'),
        'ALL': (MCI_subjects + HC_subjects, 'ALL')
    }

    # 生成一次运行说明 TXT（写在 <RESULT_ROOT>/<MODEL>/）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_root = os.path.join(args.result_dir, args.model)
    os.makedirs(model_root, exist_ok=True)
    changelog_path = os.path.join(model_root, f'run_{timestamp}.txt')
    with open(changelog_path, 'w', encoding='utf-8') as f:
        f.write(
            "本次运行改动与关键超参：\n"
            "- 先整段窗口化，再把窗口序列按时间均分为5折（余数分配给前面的折）\n"
            "- 训练集再划10%验证，仅用于早停与选择best epoch；用best.pt在测试集评估\n"
            f"- 早停监控指标：{args.early_metric}（patience={args.patience}）\n"
            f"- Adam 学习率：{args.lr}\n"
            f"- Epochs 上限：{args.epochs}（配合早停/达标即停）\n"
            f"- 绝对准确率停止：target_acc={args.target_acc} (scope={args.acc_scope}, min_epochs={args.min_epochs})\n"
            "- ★ 标签映射（三分类）：stage 1→0，2→1，3/4→2；指标使用宏平均与多类 AUC(ovr)\n"
            f"- 输出根目录：{args.result_dir}/{args.model}/<GROUP>/fold_k/\n\n"
            "关键命令行参数：\n"
            f"  --model {args.model} --epochs {args.epochs} --batch-size {args.batch_size}\n"
            f"  --window-size {args.window_size} --overlap {args.overlap}\n"
            f"  --lr {args.lr} --early-metric {args.early_metric} --patience {args.patience}\n"
            f"  --target-acc {args.target_acc} --acc-scope {args.acc_scope} --min-epochs {args.min_epochs}\n"
            f"  --data-root {args.data_root}\n"
        )

    def build_model(feat_dim, seq_len):
        if args.model == 'MLP':         return MLPClassifier(feat_dim*seq_len).to(device)
        if args.model == 'LSTM':        return LSTMClassifier(feat_dim).to(device)
        if args.model == 'Transformer': return TransformerClassifier(feat_dim, seq_len).to(device)
        if args.model == 'CNN':         return CNNClassifier(feat_dim).to(device)

    # ──────────────── 5 折循环 ────────────────
    for _, (subjects, subname) in dataset_configs.items():
        out_dir = os.path.join(args.result_dir, args.model, subname)
        os.makedirs(out_dir, exist_ok=True)

        # 预先把所有文件先窗口化，再切成 5 份并聚合
        folds = build_5fold_windows(
            subjects, required_columns, args.data_root,
            args.window_size, args.overlap, n_folds=5
        )

        total_cm = np.zeros((3, 3), int)
        all_y_true, all_y_pred = [], []
        all_y_proba = []  # (n, 3)
        overall_feats_list, overall_labels_list = [], []

        for k in range(5):
            print(f"[{args.model}][{subname}] Fold {k+1}/5")

            # 测试集：第 k 份；训练集：其余 4 份并起来
            Xte_list = folds[k]['X']; Yte_list = folds[k]['Y']
            Xtr_list = []; Ytr_list = []
            for j in range(5):
                if j == k: continue
                Xtr_list += folds[j]['X']
                Ytr_list += folds[j]['Y']

            if len(Xtr_list) == 0 or len(Xte_list) == 0:
                print("数据不足，跳过此折"); continue

            # (B, seq, feat)
            Xtr = torch.tensor(np.stack(Xtr_list), dtype=torch.float32).permute(0, 2, 1)
            Xte = torch.tensor(np.stack(Xte_list), dtype=torch.float32).permute(0, 2, 1)
            Ytr = torch.tensor(Ytr_list, dtype=torch.long)
            Yte = torch.tensor(Yte_list, dtype=torch.long)

            seq_len, feat_dim = Xtr.shape[1], Xtr.shape[2]

            # 标准化（按训练集）
            scaler = StandardScaler()
            Xtr_flat = Xtr.reshape(len(Xtr), -1).numpy()
            Xte_flat = Xte.reshape(len(Xte), -1).numpy()
            scaler.fit(Xtr_flat)
            Xtr = torch.tensor(scaler.transform(Xtr_flat), dtype=torch.float32).view(-1, seq_len, feat_dim)
            Xte = torch.tensor(scaler.transform(Xte_flat), dtype=torch.float32).view(-1, seq_len, feat_dim)

            # 训练/验证划分（仅用于早停/挑最佳）
            dataset = TensorDataset(Xtr, Ytr)
            if len(dataset) <= 1:
                print("训练样本过少，跳过此折"); continue
            n_val = max(1, int(0.1 * len(dataset)))
            n_train = len(dataset) - n_val
            if n_train == 0: n_train, n_val = 1, len(dataset) - 1
            train_ds, val_ds = random_split(dataset, [n_train, n_val])
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
            val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
            test_loader  = DataLoader(TensorDataset(Xte, Yte), batch_size=args.batch_size, shuffle=False)

            # 模型/优化器/损失
            model = build_model(feat_dim, seq_len)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            criterion = nn.CrossEntropyLoss()

            # 早停相关
            best_score = float('inf') if args.early_metric == 'val_loss' else -float('inf')
            mode_min = (args.early_metric == 'val_loss')
            min_delta = 1e-6
            best_epoch = 0
            patience_cnt = 0

            fold_dir = os.path.join(out_dir, f"fold_{k+1}")
            os.makedirs(fold_dir, exist_ok=True)
            best_path = os.path.join(fold_dir, "best.pt")

            train_losses, val_losses, train_accs, val_accs = [], [], [], []

            # 训练 & 验证（带早停 + 达标即停）
            for ep in range(1, args.epochs + 1):
                model.train()
                total_loss, correct, total = 0.0, 0, 0
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    total_loss += loss.item() * yb.size(0)
                    pred = logits.argmax(dim=1)
                    correct += (pred == yb).sum().item()
                    total += yb.size(0)
                train_losses.append(total_loss/total if total>0 else 0.0)
                train_accs.append(correct/total if total>0 else 0.0)

                model.eval()
                v_loss, v_corr, v_tot = 0.0, 0, 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        logits = model(xb)
                        l = criterion(logits, yb)
                        v_loss += l.item()*yb.size(0)
                        p = logits.argmax(dim=1)
                        v_corr += (p == yb).sum().item()
                        v_tot  += yb.size(0)
                val_loss = (v_loss/v_tot if v_tot>0 else 0.0)
                val_acc  = (v_corr/v_tot if v_tot>0 else 0.0)
                val_losses.append(val_loss); val_accs.append(val_acc)

                # 达标即停
                if args.target_acc is not None:
                    acc_now = val_acc if args.acc_scope == 'val' else train_accs[-1]
                    if acc_now >= args.target_acc and ep >= args.min_epochs:
                        torch.save(model.state_dict(), best_path)
                        best_epoch = ep
                        print(f"Target {args.acc_scope} accuracy {acc_now:.4f} >= {args.target_acc} "
                              f"-> stop at epoch {ep} (min_epochs={args.min_epochs})")
                        break

                # Early Stopping
                current = val_loss if mode_min else val_acc
                improved = (current < best_score - min_delta) if mode_min else (current > best_score + min_delta)
                if improved:
                    best_score = current
                    best_epoch = ep
                    patience_cnt = 0
                    torch.save(model.state_dict(), best_path)
                else:
                    patience_cnt += 1
                    if patience_cnt >= args.patience:
                        print(f"Early stop at epoch {ep}, best epoch = {best_epoch} ({args.early_metric}={best_score:.6f})")
                        break

            # 曲线
            plot_train_val_curves(train_losses, val_losses, train_accs, val_accs,
                                  save_path=os.path.join(fold_dir, 'train_val_curves.png'), dpi=300)

            # 加载 best 并测试
            if os.path.isfile(best_path):
                try:
                    state_obj = torch.load(best_path, map_location=device, weights_only=True)
                except TypeError:
                    state_obj = torch.load(best_path, map_location=device)
                state_dict = state_obj['model'] if isinstance(state_obj, dict) and 'model' in state_obj else state_obj
                model.load_state_dict(state_dict)
            model.eval()

            y_true, y_pred = [], []
            y_proba_rows = []
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb = xb.to(device)
                    logits = model(xb)
                    prob = torch.softmax(logits, dim=1)  # (B,3)
                    y_true.extend(yb.tolist())
                    y_pred.extend(logits.argmax(dim=1).cpu().tolist())
                    y_proba_rows.append(prob.cpu().numpy())

            y_proba = np.vstack(y_proba_rows) if len(y_proba_rows)>0 else np.zeros((0,3), dtype=float)

            cm  = confusion_matrix(y_true, y_pred, labels=[0,1,2])
            acc = accuracy_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
            pre = precision_score(y_true, y_pred, average='macro', zero_division=0)
            f1  = f1_score(y_true, y_pred, average='macro', zero_division=0)
            try:
                auc = roc_auc_score(y_true, y_proba, multi_class='ovr')
            except Exception:
                auc = 0.0

            total_cm += cm
            all_y_true += y_true
            all_y_pred += y_pred
            all_y_proba.append(y_proba)

            met = {'acc': acc, 'rec': rec, 'pre': pre, 'f1': f1, 'auc': auc}
            plot_confusion_matrix_general(cm, met, len(y_true),
                                          f"{subname} {args.model} Fold {k+1}",
                                          os.path.join(fold_dir, 'confusion.png'))

            # t-SNE（fold）+ 累计 Overall
            feats, labs = [], []
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb = xb.to(device)
                    f = model.extract(xb)
                    feats.append(f.cpu().numpy()); labs.extend(yb.tolist())
            if len(labs) > 0:
                feats = np.vstack(feats)
                feats = np.nan_to_num(feats, copy=False)
                plot_tsne(feats, labs, os.path.join(fold_dir, 'tsne.png'),
                          title=f"{subname} {args.model} Fold {k+1} t-SNE", dpi=300)
                overall_feats_list.append(feats)
                overall_labels_list.extend(labs)

        # Overall 指标 & 混淆矩阵
        if len(all_y_true) > 0:
            oacc = accuracy_score(all_y_true, all_y_pred)
            orec = recall_score(all_y_true, all_y_pred, average='macro', zero_division=0)
            opre = precision_score(all_y_true, all_y_pred, average='macro', zero_division=0)
            of1  = f1_score(all_y_true, all_y_pred, average='macro', zero_division=0)
            try:
                y_proba_all = np.vstack(all_y_proba) if len(all_y_proba)>0 else np.zeros((0,3), dtype=float)
                oauc = roc_auc_score(all_y_true, y_proba_all, multi_class='ovr')
            except Exception:
                oauc = 0.0
            omet = {'acc': oacc, 'rec': orec, 'pre': opre, 'f1': of1, 'auc': oauc}
            plot_confusion_matrix_general(total_cm, omet, len(all_y_true),
                                          f"{subname} {args.model} Overall",
                                          os.path.join(out_dir, 'confusion_overall.png'))

        # Overall t-SNE
        if len(overall_labels_list) > 0:
            feats_all = np.vstack(overall_feats_list)
            feats_all = np.nan_to_num(feats_all, copy=False)
            plot_tsne(feats_all, overall_labels_list,
                      save_path=os.path.join(out_dir, 'tsne_overall.png'),
                      title=f"{subname} {args.model} Overall t-SNE", dpi=300)

    print("All done.")

if __name__ == '__main__':
    main()
