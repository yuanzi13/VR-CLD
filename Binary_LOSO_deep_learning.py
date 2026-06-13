#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MIX_2.py：四模型对比（MLP / LSTM / Transformer / CNN）LOSO 留一交叉验证
-----------------------------------------------------------------------
· 新数据结构（替换旧 A/B/C/D）：
  data_root/<Population>/<stage>/<number>.csv
  - Population: MCI 或 HC
  - stage: 1/2 → label=0；3/4 → label=1
  - number: MCI=1..26，HC=1..42（各自计数）
· 四模型：MLP、LSTM、Transformer、CNN
· 三组：MCI / HC / ALL（移除 SCD）
· 每折：训练集再划 10% 验证，默认 1000 epochs
· 输出到：Binary_LOSO_independent_68/<MODEL>/<GROUP>/
  - loss.png / acc.png
  - confusion.png（每折） / confusion_overall.png（整体）
  - tsne.png（每折） / tsne_overall.png（整体）
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
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
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 无显示环境下保存图
matplotlib.use("Agg")
rcParams['font.family'] = ['WenQuanYi Micro Hei']

# ───────────────── 数据准备 ─────────────────
def DataToXY(df: pd.DataFrame, label: int,
             window_size: int = 240, overlap: float = 0.0):
    data = df.values.astype(np.float32)
    if data.shape[0] > data.shape[1]:
        data = data.T
    X, Y = [], []
    step = window_size - int(window_size * overlap)
    if step <= 0: step = 1
    if data.shape[1] < window_size:
        return X, Y
    n_seg = (data.shape[1] - window_size) // step + 1
    for i in range(max(0, n_seg)):
        seg = data[:, step*i: step*i+window_size]
        if seg.shape[1] != window_size: continue
        X.append(seg)   # (feat, seq)
        Y.append(label)
    return X, Y

def load_data_rml(test_subj, subjects, required_columns, data_root, window_size, overlap):
    """
    test_subj: (pop, num)
    subjects:  list of (pop, num)
    路径：data_root/<pop>/<stage>/<num>.csv
    stage∈{1,2}->0；{3,4}->1
    """
    train_X, train_Y, test_X, test_Y = [], [], [], []
    # stages 与标签
    stage_label = [('1', 0), ('2', 0), ('3', 1), ('4', 1)]

    for pop, num in subjects:
        for stage, lab in stage_label:
            fpath = os.path.join(data_root, pop, stage, f"{num}.csv")
            if not os.path.isfile(fpath):
                continue
            try:
                df = pd.read_csv(fpath)
            except Exception as e:
                print(f"⚠️ 读取失败: {fpath} ({e})")
                continue
            cols = [c for c in required_columns if c in df.columns]
            if len(cols) == 0:
                print(f"⚠️ 缺失全部所需列，跳过: {fpath}")
                continue
            df = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
            Xs, Ys = DataToXY(df, lab, window_size, overlap)
            if (pop, num) == test_subj:
                test_X.extend(Xs); test_Y.extend(Ys)
            else:
                train_X.extend(Xs); train_Y.extend(Ys)
    return train_X, train_Y, test_X, test_Y

# ───────────────── 模型定义 ─────────────────
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256], num_classes=2):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.5)]
            prev = h
        layers += [nn.Linear(prev, num_classes)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        B = x.size(0)              # (B, seq, feat)
        x = x.view(B, -1)
        return self.net(x)
    def extract(self, x):
        B = x.size(0)
        x = x.view(B, -1)
        for layer in self.net[:-1]:
            x = layer(x)
        return x

class LSTMClassifier(nn.Module):
    def __init__(self, feat_dim, hidden_dim=128, num_layers=1, num_classes=2):
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
                 model_dim=32, nhead=4, dim_feedforward=128, num_layers=2, num_classes=2):
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
    def __init__(self, in_ch, num_classes=2):
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

# ───────────────── 绘图工具 ─────────────────
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

def plot_train_val_curve(train_vals, val_vals, ylabel, save_path, dpi=300):
    plt.figure(figsize=(6, 4))
    plt.plot(train_vals, label='Train')
    plt.plot(val_vals, label='Val')
    plt.xlabel('Epoch'); plt.ylabel(ylabel)
    plt.legend(); plt.tight_layout()
    safe_savefig(save_path, dpi=dpi)

def plot_confusion_matrix(cm, metrics, total, title, save_path):
    """
    metrics: dict(acc/rec/pre/f1/auc) in [0,1]
    单元格字号固定 28；右侧显示 ACC/REC/PRE/F1/AUC。
    """
    M = np.zeros((3, 3))
    M[:2, :2] = cm
    for i in range(2):
        tp  = cm[i, i]
        col = cm[:, i].sum()
        row = cm[i, :].sum()
        M[2, i] = tp/col if col else 0
        M[i, 2] = tp/row if row else 0
    M[2, 2] = metrics['acc']

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    im = ax.imshow(M, cmap='Blues', interpolation='nearest')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = [0, 1, 2]
    ax.set_xticks(ticks, ['P0', 'P1', 'Precision'], fontsize=12, rotation=45)
    ax.set_yticks(ticks, ['T0', 'T1', 'Recall'],    fontsize=12)

    thresh = M.max() / 2
    for i, j in np.ndindex(M.shape):
        v = M[i, j]
        if i < 2 and j < 2:
            perc = (v/total) if total else 0.0
            s = f"{int(v)}\n({perc:.2%})"
        else:
            s = f"{v*100:.2f}%"
        ax.text(j, i, s, ha='center', va='center',
                color='white' if v > thresh else 'black',
                fontsize=14)  # 固定 14

    ax.set_ylabel('真实标签'); ax.set_xlabel('预测标签')
    ax.set_title(title.split('\n', 1)[0], fontsize=14)

    txt = (f"ACC={metrics['acc']:.4f}\n"
           f"REC={metrics['rec']:.4f}\n"
           f"PRE={metrics['pre']:.4f}\n"
           f"F1 ={metrics['f1']:.4f}\n"
           f"AUC={metrics['auc']:.4f}")
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
    color_map = {0: '#1f77b4', 1: '#d62728'}
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        choices=['MLP', 'LSTM', 'Transformer', 'CNN'], required=True)
    parser.add_argument('--gpu-id', type=int, default=None,
        help='物理卡号，如 2；不指定则用默认映射（MLP:0/LSTM:1/Transformer:2/CNN:3）')
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--window-size', type=int, default=240)
    parser.add_argument('--overlap', type=float, default=0.0)
    parser.add_argument('--result-dir', type=str, default='Binary_LOSO_independent_68')
    parser.add_argument('--data-root', type=str, default='data_rml',
                        help="数据根目录（默认相对路径 data_rml；也可填绝对路径或 DeepLearning/data_rml）")
    args = parser.parse_args()

    # 解析 data_root（兼容 DeepLearning/data_rml）
    if not os.path.isdir(args.data_root) and os.path.isdir(os.path.join('DeepLearning', 'data_rml')):
        args.data_root = os.path.join('DeepLearning', 'data_rml')

    # 模型→物理卡默认映射（可被 --gpu-id 覆盖）
    default_map = {'MLP': 0, 'LSTM': 1, 'Transformer': 2, 'CNN': 3}
    gpu = args.gpu_id if args.gpu_id is not None else default_map[args.model]

    # 设备
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print(f"Using device: {device} for model {args.model}")

    required_columns = [
        'leftEye_gaze_X', 'leftEye_gaze_Y', 'leftEye_gaze_Z',
        'leftEye_openness', 'leftEye_pupil_position_X', 'leftEye_pupil_position_Y', 'leftEye_pupil_dilation',
        'rightEye_gaze_X', 'rightEye_gaze_Y', 'rightEye_gaze_Z',
        'rightEye_openness', 'rightEye_pupil_position_X', 'rightEye_pupil_position_Y', 'rightEye_pupil_dilation',
        'combinedEye_gaze_X', 'combinedEye_gaze_Y', 'combinedEye_gaze_Z'
    ]

    # 受试者列表（按新方案：MCI=1..26；HC=1..42）
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

    # ──────────────── 开始 LOSO ────────────────
    for dtype, (subjects, subname) in dataset_configs.items():
        out_dir = os.path.join(args.result_dir, args.model, subname)
        os.makedirs(out_dir, exist_ok=True)

        total_cm = np.zeros((2, 2), int)
        all_y_true, all_y_pred, all_y_score = [], [], []
        overall_feats_list, overall_labels_list = [], []

        for idx, test_subj in enumerate(subjects, 1):
            print(f"[{args.model}][{subname}] Fold {idx}/{len(subjects)}: {test_subj[0]}-{test_subj[1]}")

            # 1) 数据（新路径+标签规则）
            Xtr, Ytr, Xte, Yte = load_data_rml(
                test_subj, subjects, required_columns,
                args.data_root, args.window_size, args.overlap
            )
            if len(Xtr) == 0 or len(Xte) == 0:
                print("数据不足，跳过此折")
                continue

            Xtr = torch.tensor(np.stack(Xtr), dtype=torch.float32)
            Ytr = torch.tensor(Ytr, dtype=torch.long)
            Xte = torch.tensor(np.stack(Xte), dtype=torch.float32)
            Yte = torch.tensor(Yte, dtype=torch.long)

            # 标准化：扁平 -> 标准化 -> 还原为 (B, seq, feat)
            feat_dim, seq_len = Xtr.shape[1], Xtr.shape[2]
            scaler = StandardScaler()
            Xtr_flat = Xtr.view(len(Xtr), -1).numpy()
            Xte_flat = Xte.view(len(Xte), -1).numpy()
            scaler.fit(Xtr_flat)
            Xtr_flat = scaler.transform(Xtr_flat)
            Xte_flat = scaler.transform(Xte_flat)
            Xtr = torch.tensor(Xtr_flat, dtype=torch.float32).view(-1, seq_len, feat_dim)
            Xte = torch.tensor(Xte_flat, dtype=torch.float32).view(-1, seq_len, feat_dim)

            # 2) 训练/验证划分
            dataset = TensorDataset(Xtr, Ytr)
            if len(dataset) <= 1:
                print("训练样本过少，跳过此折")
                continue
            n_val = max(1, int(0.1 * len(dataset)))
            n_train = len(dataset) - n_val
            if n_train == 0:
                n_train, n_val = 1, len(dataset) - 1
            train_ds, val_ds = random_split(dataset, [n_train, n_val])
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
            val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
            test_loader  = DataLoader(TensorDataset(Xte, Yte), batch_size=args.batch_size, shuffle=False)

            # 3) 模型/优化器/损失
            model = build_model(feat_dim, seq_len)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()

            train_losses, val_losses, train_accs, val_accs = [], [], [], []

            # 4) 训练 & 验证
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
                val_losses.append(v_loss/v_tot if v_tot>0 else 0.0)
                val_accs.append(v_corr/v_tot if v_tot>0 else 0.0)

            # 5) 曲线
            fold_dir = os.path.join(out_dir, f"fold_{idx}")
            os.makedirs(fold_dir, exist_ok=True)
            plot_train_val_curve(train_losses, val_losses, 'Loss',     os.path.join(fold_dir, 'loss.png'))
            plot_train_val_curve(train_accs,  val_accs,   'Accuracy', os.path.join(fold_dir, 'acc.png'))

            # 6) 测试 & 指标
            model.eval()
            y_true, y_pred, y_score = [], [], []
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb = xb.to(device)
                    logits = model(xb)
                    prob1 = torch.softmax(logits, dim=1)[:, 1]
                    y_true.extend(yb.tolist())
                    y_pred.extend(logits.argmax(dim=1).cpu().tolist())
                    y_score.extend(prob1.cpu().tolist())

            cm  = confusion_matrix(y_true, y_pred, labels=[0, 1])
            acc = accuracy_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred, zero_division=0)
            pre = precision_score(y_true, y_pred, zero_division=0)
            f1  = f1_score(y_true, y_pred, zero_division=0)
            try: auc = roc_auc_score(y_true, y_score)
            except Exception: auc = 0.0

            total_cm += cm
            all_y_true += y_true
            all_y_pred += y_pred
            all_y_score += y_score

            met = {'acc': acc, 'rec': rec, 'pre': pre, 'f1': f1, 'auc': auc}
            plot_confusion_matrix(cm, met, len(y_true),
                                  f"{subname} {args.model} Fold {idx}",
                                  os.path.join(fold_dir, 'confusion.png'))

            # 7) t-SNE（fold）+ 累计 Overall
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
                          title=f"{subname} {args.model} Fold {idx} t-SNE", dpi=300)
                overall_feats_list.append(feats)
                overall_labels_list.extend(labs)

        # ── Overall 指标 & 混淆矩阵 ──
        if len(all_y_true) > 0:
            oacc = accuracy_score(all_y_true, all_y_pred)
            orec = recall_score(all_y_true, all_y_pred, zero_division=0)
            opre = precision_score(all_y_true, all_y_pred, zero_division=0)
            of1  = f1_score(all_y_true, all_y_pred, zero_division=0)
            try: oauc = roc_auc_score(all_y_true, all_y_score)
            except Exception: oauc = 0.0
            omet = {'acc': oacc, 'rec': orec, 'pre': opre, 'f1': of1, 'auc': oauc}
            plot_confusion_matrix(total_cm, omet, len(all_y_true),
                                  f"{subname} {args.model} Overall",
                                  os.path.join(out_dir, 'confusion_overall.png'))

        # ── Overall t-SNE ──
        if len(overall_labels_list) > 0:
            feats_all = np.vstack(overall_feats_list)
            feats_all = np.nan_to_num(feats_all, copy=False)
            plot_tsne(feats_all, overall_labels_list,
                      save_path=os.path.join(out_dir, 'tsne_overall.png'),
                      title=f"{subname} {args.model} Overall t-SNE", dpi=300)

    print("All done.")

if __name__ == '__main__':
    main()
