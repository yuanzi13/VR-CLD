#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MIX_2.py：四模型对比（MLP / LSTM / Transformer / CNN）三分类 + LOSO
-----------------------------------------------------------------------
· 数据结构：
  data_root/<Population>/<stage>/<number>.csv
  - Population: MCI 或 HC
  - stage: 1 -> label=0；2 -> label=1；3/4 -> label=2
  - number: MCI=1..26，HC=1..42

· 三组：MCI / HC / ALL
· 训练：每折将训练集再划 10% 验证；输出训练曲线
· 可视化：
  - 混淆矩阵（旁侧标注 Accuracy / Recall / Precision / F1 / AUC）
  - ROC（多分类 OVR：三条单类曲线 + micro + macro）
  - t-SNE（提取模型特征后降维）
· 输出目录：<result-dir>/<MODEL>/<GROUP>/fold_k/ 与 overall 文件
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve
)
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = ['WenQuanYi Micro Hei']
rcParams['figure.dpi'] = 120

# --------------------------- 基础 ---------------------------
NUM_CLASSES = 3
STAGE_LABEL = [('1', 0), ('2', 1), ('3', 2), ('4', 2)]
REQUIRED_COLUMNS = [
    'leftEye_gaze_X','leftEye_gaze_Y','leftEye_gaze_Z',
    'leftEye_openness','leftEye_pupil_position_X','leftEye_pupil_position_Y','leftEye_pupil_dilation',
    'rightEye_gaze_X','rightEye_gaze_Y','rightEye_gaze_Z',
    'rightEye_openness','rightEye_pupil_position_X','rightEye_pupil_position_Y','rightEye_pupil_dilation',
    'combinedEye_gaze_X','combinedEye_gaze_Y','combinedEye_gaze_Z'
]

def DataToXY(df: pd.DataFrame, label: int, window_size=240, overlap=0.0):
    data = df.values.astype(np.float32)
    if data.shape[0] > data.shape[1]:
        data = data.T                      # (C,T)
    step = max(1, window_size - int(window_size * overlap))
    if data.shape[1] < window_size:
        return [], []
    X, Y = [], []
    n_seg = (data.shape[1] - window_size) // step + 1
    for i in range(n_seg):
        seg = data[:, i*step:i*step+window_size]      # (C, W)
        X.append(seg)
        Y.append(label)
    return X, Y

def load_data_rml(test_subj, subjects, data_root, window_size, overlap):
    train_X, train_Y, test_X, test_Y = [], [], [], []
    for pop, num in subjects:
        for stage, lab in STAGE_LABEL:
            fpath = os.path.join(data_root, pop, stage, f"{num}.csv")
            if not os.path.isfile(fpath): 
                continue
            try:
                df = pd.read_csv(fpath)
            except Exception as e:
                print(f"⚠️ 读取失败: {fpath} ({e})"); 
                continue
            cols = [c for c in REQUIRED_COLUMNS if c in df.columns]
            if not cols:
                print(f"⚠️ 缺失所需列，跳过: {fpath}")
                continue
            df = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
            Xs, Ys = DataToXY(df, lab, window_size, overlap)
            if (pop, num) == test_subj:
                test_X += Xs; test_Y += Ys
            else:
                train_X += Xs; train_Y += Ys
    return train_X, train_Y, test_X, test_Y

# --------------------------- 模型 ---------------------------
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden=[512, 256], num_classes=NUM_CLASSES):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.5)]
            prev = h
        layers += [nn.Linear(prev, num_classes)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):                 # (B, seq, feat)
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
    def __init__(self, feat_dim, hidden_dim=128, num_layers=1, num_classes=NUM_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)            # (B, T, H)
        h = out[:, -1, :]
        return self.fc(h)
    def extract(self, x):
        out, _ = self.lstm(x)
        return out[:, -1, :]

class TransformerClassifier(nn.Module):
    def __init__(self, feat_dim, seq_len, model_dim=64, nhead=4, dim_feedforward=256, num_layers=2, num_classes=NUM_CLASSES):
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
        out = self.encoder(x)            # (B, T, D)
        h = out.mean(dim=1)              # mean-pool
        return self.fc(h)
    def extract(self, x):
        x = self.proj(x) + self.pos_embed
        out = self.encoder(x)
        return out.mean(dim=1)

class CNNClassifier(nn.Module):
    def __init__(self, in_ch, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),    nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),    nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, num_classes)
    def forward(self, x):
        x = x.permute(0, 2, 1)           # (B, T, C) -> (B, C, T)
        out = self.features(x).squeeze(-1)  # (B, 64)
        return self.fc(out)
    def extract(self, x):
        x = x.permute(0, 2, 1)
        out = self.features(x).squeeze(-1)
        return out

# --------------------------- 绘图 ---------------------------
def safe_savefig(path, dpi=300):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_train_val_curve(tr, va, ylabel, path):
    plt.figure(figsize=(6.2,4))
    plt.plot(tr, label='Train'); plt.plot(va, label='Val')
    plt.xlabel('Epoch'); plt.ylabel(ylabel); plt.legend(); plt.tight_layout()
    safe_savefig(path, dpi=300)

def plot_confusion_with_metrics(cm, metrics, title, path):
    """
    通用 (n 类) 混淆矩阵；右侧显示 Accuracy / Recall / Precision / F1 / AUC（均为 macro）
    """
    n = cm.shape[0]
    M = np.zeros((n+1, n+1), dtype=float)
    M[:n, :n] = cm
    # 每类 Precision/Recall（放到最后一行/列）
    for i in range(n):
        tp = cm[i,i]
        col = cm[:,i].sum(); row = cm[i,:].sum()
        M[n, i] = tp/col if col else 0.0     # Precision_i
        M[i, n] = tp/row if row else 0.0     # Recall_i
    M[n, n] = metrics['acc']

    plt.figure(figsize=(8.2,5.6))
    im = plt.imshow(M, cmap='Blues', interpolation='nearest')
    plt.colorbar(im, fraction=0.046, pad=0.04)

    ticks = np.arange(n+1)
    plt.xticks(ticks, [f'P{i}' for i in range(n)] + ['Prec.'], rotation=45)
    plt.yticks(ticks, [f'T{i}' for i in range(n)] + ['Rec.'])
    thresh = M.max()/2 if M.size else 0.5

    total = cm.sum()
    for i, j in np.ndindex(M.shape):
        v = M[i, j]
        if i < n and j < n:
            perc = v/total if total else 0.0
            s = f"{int(v)}\n({perc:.1%})"
        else:
            s = f"{v*100:.1f}%"
        plt.text(j, i, s, ha='center', va='center',
                 color='white' if v > thresh else 'black', fontsize=12)

    plt.title(title, fontsize=14)
    plt.xlabel('预测'); plt.ylabel('真实')

    side = (f"Accuracy = {metrics['acc']:.4f}\n"
            f"Recall (macro) = {metrics['rec']:.4f}\n"
            f"Precision (macro) = {metrics['pre']:.4f}\n"
            f"F1-score (macro) = {metrics['f1']:.4f}\n"
            f"AUC (macro-OVR) = {metrics['auc']:.4f}")
    ax = plt.gca()
    ax.text(1.05, 0.02, side, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    plt.tight_layout(rect=[0,0,0.8,1])
    safe_savefig(path, dpi=300)

def plot_multiclass_roc(y_true, y_prob, n_classes, title, path, include_micro=False):
    """
    y_true: (N,) int labels 0..n-1
    y_prob: (N, n) softmax prob
    画：每类 OVR ROC + macro（平均），默认不画 micro（可通过 include_micro=True 打开）
    """
    # 使用全局的 numpy as np（不要在函数里再 import）
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # one-hot
    Y = np.zeros((len(y_true), n_classes), dtype=int)
    Y[np.arange(len(y_true)), y_true] = 1

    fpr, tpr, roc_auc = {}, {}, {}

    # 每类 one-vs-rest
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y[:, i], y_prob[:, i])
        # 仅当该类既有正样本也有负样本时才计算 AUC；否则设为 NaN，避免异常
        pos = Y[:, i].sum()
        roc_auc[i] = roc_auc_score(Y[:, i], y_prob[:, i]) if (pos not in (0, len(Y))) else np.nan

    # macro：在所有唯一 fpr 上插值求平均
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = roc_auc_score(Y, y_prob, average='macro', multi_class='ovr')

    # （可选）micro
    if include_micro:
        fpr["micro"], tpr["micro"], _ = roc_curve(Y.ravel(), y_prob.ravel())
        roc_auc["micro"] = roc_auc_score(Y, y_prob, average='micro', multi_class='ovr')

    # 绘图
    plt.figure(figsize=(6.4, 6.0))
    for i in range(n_classes):
        if not np.isnan(roc_auc[i]):
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC={roc_auc[i]:.3f})')
    # 只画 macro（必画）
    plt.plot(fpr["macro"], tpr["macro"], linestyle='--', label=f'macro (AUC={roc_auc["macro"]:.3f})')
    # 如需 micro，再画
    if include_micro:
        plt.plot(fpr["micro"], tpr["micro"], linestyle='--', label=f'micro (AUC={roc_auc["micro"]:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0, 1]); plt.ylim([0, 1])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(title); plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches='tight'); plt.close()

def plot_tsne(feats, labels, path, title=None, max_points=6000, random_state=42):
    feats = np.asarray(feats); labels = np.asarray(labels)
    feats = np.nan_to_num(feats, copy=False)
    n = feats.shape[0]
    if n < 3:
        plt.figure(figsize=(6,4)); plt.axis('off')
        plt.text(0.5,0.5,f"{title or 't-SNE'}\n(n={n} < 3, skipped)",ha='center',va='center')
        safe_savefig(path, dpi=300); return
    # 限制点数
    if max_points and n > max_points:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n, max_points, replace=False)
        feats = feats[idx]; labels = labels[idx]
    # TSNE
    try:
        emb = TSNE(n_components=2, init='pca', learning_rate='auto',
                   perplexity=min(40, max(5, n//50)), random_state=random_state).fit_transform(feats)
    except Exception:
        from sklearn.decomposition import PCA
        emb = PCA(n_components=2).fit_transform(feats)

    plt.figure(figsize=(6.8,5.2))
    cmap = {0:'#1f77b4', 1:'#2ca02c', 2:'#d62728'}
    for c in np.unique(labels):
        m = labels == c
        plt.scatter(emb[m,0], emb[m,1], s=10, alpha=0.65, c=cmap.get(int(c),'#7f7f7f'), edgecolors='none', label=f'{int(c)} (n={m.sum()})')
    if title: plt.title(title)
    plt.grid(alpha=0.2, linestyle='--')
    plt.legend(loc='best', framealpha=0.9, fontsize=9, title='Class')
    plt.tight_layout()
    safe_savefig(path, dpi=300)

# --------------------------- 主流程 ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['MLP','LSTM','Transformer','CNN'], required=True)
    parser.add_argument('--datasets', default='MCI,HC,ALL')
    parser.add_argument('--data-root', type=str, default='data_rml')
    parser.add_argument('--result-dir', type=str, default='Ternary_LOSO_independent_68')
    parser.add_argument('--window-size', type=int, default=240)
    parser.add_argument('--overlap', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu-id', type=int, default=None)
    args = parser.parse_args()

    # 兼容 DeepLearning/data_rml
    if (not os.path.isdir(args.data_root)) and os.path.isdir(os.path.join('DeepLearning','data_rml')):
        args.data_root = os.path.join('DeepLearning','data_rml')

    # 设备
    device = torch.device(f'cuda:{args.gpu_id}' if (args.gpu_id is not None and torch.cuda.is_available()) else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")

    # 受试者列表
    MCI_subjects = [('MCI', i) for i in range(1, 26+1)]
    HC_subjects  = [('HC',  i) for i in range(1, 42+1)]
    dataset_configs = {
        'MCI': (MCI_subjects, 'MCI'),
        'HC':  (HC_subjects,  'HC'),
        'ALL': (MCI_subjects + HC_subjects, 'ALL')
    }

    def build_model(feat_dim, seq_len):
        if args.model == 'MLP':         return MLPClassifier(feat_dim*seq_len).to(device)
        if args.model == 'LSTM':        return LSTMClassifier(feat_dim).to(device)
        if args.model == 'Transformer': return TransformerClassifier(feat_dim, seq_len).to(device)
        if args.model == 'CNN':         return CNNClassifier(feat_dim).to(device)

    # 解析 datasets
    req_groups = [s.strip().upper() for s in args.datasets.split(',') if s.strip()]
    for grp in req_groups:
        if grp not in dataset_configs:
            raise ValueError(f"未知数据组: {grp} (可选: MCI,HC,ALL)")

    for grp in req_groups:
        subjects, subname = dataset_configs[grp]
        out_dir = os.path.join(args.result_dir, args.model, subname)
        os.makedirs(out_dir, exist_ok=True)

        total_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        all_true, all_pred = [], []
        all_prob_list = []

        # t-SNE overall 累计
        overall_feats, overall_labels = [], []

        for fold_idx, test_subj in enumerate(subjects, 1):
            print(f"[{args.model}][{subname}] LOSO Fold {fold_idx}/{len(subjects)}: {test_subj[0]}-{test_subj[1]:02d}")
            fold_dir = os.path.join(out_dir, f"fold_{fold_idx}")
            os.makedirs(fold_dir, exist_ok=True)

            # 数据
            Xtr, Ytr, Xte, Yte = load_data_rml(test_subj, subjects, args.data_root, args.window_size, args.overlap)
            if not Xtr or not Xte:
                print("  ⚠️ 数据不足，跳过此折")
                continue

            # (B, C, W) -> (B, T, C) 这里 T=W, C=feat_dim
            Xtr = torch.tensor(np.stack(Xtr), dtype=torch.float32).permute(0,2,1)  # (B, W, C)
            Xte = torch.tensor(np.stack(Xte), dtype=torch.float32).permute(0,2,1)
            Ytr = torch.tensor(Ytr, dtype=torch.long)
            Yte = torch.tensor(Yte, dtype=torch.long)

            seq_len, feat_dim = Xtr.shape[1], Xtr.shape[2]

            # 标准化
            scaler = StandardScaler()
            Xtr_flat = Xtr.reshape(len(Xtr), -1).numpy()
            Xte_flat = Xte.reshape(len(Xte), -1).numpy()
            scaler.fit(Xtr_flat)
            Xtr = torch.tensor(scaler.transform(Xtr_flat), dtype=torch.float32).view(-1, seq_len, feat_dim)
            Xte = torch.tensor(scaler.transform(Xte_flat), dtype=torch.float32).view(-1, seq_len, feat_dim)

            # 训练/验证
            full_ds = TensorDataset(Xtr, Ytr)
            n_val = max(1, int(0.1 * len(full_ds)))
            n_trn = len(full_ds) - n_val
            if n_trn<=0: n_trn, n_val = 1, len(full_ds)-1
            trn_ds, val_ds = random_split(full_ds, [n_trn, n_val], generator=torch.Generator().manual_seed(42))
            train_loader = DataLoader(trn_ds, batch_size=args.batch_size, shuffle=True)
            val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
            test_loader  = DataLoader(TensorDataset(Xte, Yte), batch_size=args.batch_size, shuffle=False)

            model = build_model(feat_dim, seq_len)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            criterion = nn.CrossEntropyLoss()

            tr_losses, va_losses, tr_accs, va_accs = [], [], [], []
            for ep in range(1, args.epochs+1):
                # train
                model.train()
                tot, corr, loss_sum = 0, 0, 0.0
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    loss_sum += loss.item() * yb.size(0)
                    corr += (logits.argmax(1)==yb).sum().item()
                    tot  += yb.size(0)
                tr_losses.append(loss_sum/max(1,tot))
                tr_accs.append(corr/max(1,tot))

                # val
                model.eval()
                vtot, vcorr, vloss_sum = 0, 0, 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        logits = model(xb)
                        vloss_sum += criterion(logits, yb).item() * yb.size(0)
                        vcorr += (logits.argmax(1)==yb).sum().item()
                        vtot  += yb.size(0)
                va_losses.append(vloss_sum/max(1,vtot))
                va_accs.append(vcorr/max(1,vtot))

            # 训练曲线
            plot_train_val_curve(tr_losses, va_losses, 'Loss', os.path.join(fold_dir, 'loss.png'))
            plot_train_val_curve(tr_accs,  va_accs,  'Accuracy', os.path.join(fold_dir, 'acc.png'))

            # 测试
            model.eval()
            y_true, y_pred, y_prob = [], [], []
            feats_fold, labs_fold = [], []
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb = xb.to(device)
                    logits = model(xb)                               # (B, 3)
                    prob = torch.softmax(logits, dim=1).cpu().numpy()
                    y_prob.extend(prob.tolist())
                    y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
                    y_true.extend(yb.numpy().tolist())
                    # 特征 for t-SNE
                    f = model.extract(xb).cpu().numpy()              # (B, D)
                    feats_fold.append(f); labs_fold.extend(yb.numpy().tolist())

            y_true = np.array(y_true); y_pred = np.array(y_pred); y_prob = np.array(y_prob)
            cm  = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
            acc = accuracy_score(y_true, y_pred)
            pre = precision_score(y_true, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1  = f1_score(y_true, y_pred, average='macro', zero_division=0)
            try:
                auc_macro = roc_auc_score(y_true, y_prob, average='macro', multi_class='ovr')
            except Exception:
                auc_macro = 0.0

            # 保存可视化
            plot_confusion_with_metrics(
                cm,
                {'acc':acc,'rec':rec,'pre':pre,'f1':f1,'auc':auc_macro},
                title=f"{subname} {args.model} LOSO Fold {fold_idx}",
                path=os.path.join(fold_dir, 'confusion.png')
            )
            plot_multiclass_roc(y_true, y_prob, NUM_CLASSES,
                                title=f"{subname} {args.model} LOSO Fold {fold_idx} ROC (OVR)",
                                path=os.path.join(fold_dir, 'roc.png'))

            # t-SNE（本折）
            if feats_fold:
                feats_fold = np.vstack(feats_fold)
                plot_tsne(feats_fold, np.array(labs_fold),
                          path=os.path.join(fold_dir, 'tsne.png'),
                          title=f"{subname} {args.model} LOSO Fold {fold_idx} t-SNE")
                overall_feats.append(feats_fold)
                overall_labels.extend(labs_fold)

            # 累计 overall
            total_cm += cm
            all_true.extend(y_true.tolist())
            all_pred.extend(y_pred.tolist())
            all_prob_list.append(y_prob)

        # Overall
        if all_true:
            all_true = np.array(all_true)
            all_pred = np.array(all_pred)
            all_prob = np.vstack(all_prob_list) if len(all_prob_list)>0 else None

            oa  = accuracy_score(all_true, all_pred)
            op  = precision_score(all_true, all_pred, average='macro', zero_division=0)
            orc = recall_score(all_true, all_pred, average='macro', zero_division=0)
            of1 = f1_score(all_true, all_pred, average='macro', zero_division=0)
            try:
                oauc = roc_auc_score(all_true, all_prob, average='macro', multi_class='ovr') if all_prob is not None else 0.0
            except Exception:
                oauc = 0.0

            plot_confusion_with_metrics(
                total_cm,
                {'acc':oa,'rec':orc,'pre':op,'f1':of1,'auc':oauc},
                title=f"{subname} {args.model} LOSO Overall",
                path=os.path.join(out_dir, 'confusion_overall.png')
            )
            if all_prob is not None:
                plot_multiclass_roc(all_true, all_prob, NUM_CLASSES,
                                    title=f"{subname} {args.model} LOSO Overall ROC (OVR)",
                                    path=os.path.join(out_dir, 'roc_overall.png'))

        # Overall t-SNE
        if overall_feats:
            feats_all = np.vstack(overall_feats)
            plot_tsne(feats_all, np.array(overall_labels),
                      path=os.path.join(out_dir, 'tsne_overall.png'),
                      title=f"{subname} {args.model} LOSO Overall t-SNE")

    print("All done.")

if __name__ == '__main__':
    main()
