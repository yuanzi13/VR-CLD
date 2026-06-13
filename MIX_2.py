#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mix.py：经典机器学习多模型（NB / SVM / KNN / DT / RF / AdaBoost）+ TCN（PyTorch）+ 非独立五折
—— 与 MIX_2.py 保持一致的【数据输入 / 标签规则 / 数据集划分方法（先窗口化→再均分5折）】

目录结构（与 MIX_2.py 一致）：
  data_root/<Population>/<stage>/<number>.csv
    - Population: MCI 或 HC
    - stage: '1'/'2'→label=0；'3'/'4'→label=1（本脚本二分类）
    - number: MCI=1..26，HC=1..42（各自独立编号）

输出：
  Binary_5K_dependent_68/<model>/<model_dataset>/fold_xx/...
    - confusion.png / confusion_overall.png
    - roc.png / roc_overall.png
    - accuracy_across_folds.png
    - （TCN 还会额外保存 best.pt 与 train_val_curves.png）

快速启动（把不同模型分配到 4/5/6/7 卡上，并跑所有数据集）：
  python mix.py --data-root DeepLearning/data_rml --launch4567
"""

import os, sys, argparse, subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve
)
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# ------------------------ 全局显示设置 ------------------------
rcParams['font.family'] = ['WenQuanYi Micro Hei']

# ------------------------ 必要特征列（与 MIX_2.py 保持一致） ------------------------
REQUIRED_COLUMNS = [
    'leftEye_gaze_X','leftEye_gaze_Y','leftEye_gaze_Z',
    'leftEye_openness','leftEye_pupil_position_X',
    'leftEye_pupil_position_Y','leftEye_pupil_dilation',
    'rightEye_gaze_X','rightEye_gaze_Y','rightEye_gaze_Z',
    'rightEye_openness','rightEye_pupil_position_X',
    'rightEye_pupil_position_Y','rightEye_pupil_dilation',
    'combinedEye_gaze_X','combinedEye_gaze_Y','combinedEye_gaze_Z'
]

# ------------------------ 窗口化（与 MIX_2.py 等价） ------------------------
def windowize_from_array(arr_ch_t: np.ndarray, label: int, window_size: int = 240, overlap: float = 0.0):
    """
    arr_ch_t: (C, T)
    返回：list[torch.Tensor(1,-1)] , list[int]
    说明：为了与传统 ML 兼容，这里把 (C,T) 拉平成 (1, C*T)。
         TCN 训练阶段会按 C,T 维度再 reshape 回去。
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
        seg = arr_ch_t[:, i*step:i*step+window_size].astype('float64')
        seg = np.nan_to_num(seg)
        X.append(torch.from_numpy(seg).reshape(1, -1))  # flatten
        Y.append(label)
    return X, Y

def split_windows_to_5folds(windows, labels, n_folds=5):
    """按窗口序列顺序等分到 5 折（余数给前面），与 MIX_2.py 的 split 规则一致。"""
    folds = [{'X': [], 'Y': []} for _ in range(n_folds)]
    n = len(windows)
    if n == 0:
        return folds
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

# ------------------------ 5 折构建（与 MIX_2.py 一致：先窗后切） ------------------------
def build_5fold_windows(data_root, subjects, required_columns, window_size, overlap, n_folds=5):
    """
    返回 folds 列表（长度 n_folds），每个元素 {'X': list[tensor(1,-1)], 'Y': list[int]}
    对每个 CSV：先窗口化，再把窗口“按数量”均分到 5 折（时间顺序）。
    """
    folds = [{'X': [], 'Y': []} for _ in range(n_folds)]
    # 二分类（与原 mix.py 保持）：'1','2' -> 0；'3','4' -> 1
    stage_label = [('1', 0), ('2', 0), ('3', 1), ('4', 1)]

    for (pop, num) in subjects:
        for stage, lab in stage_label:
            csv_path = os.path.join(data_root, pop, stage, f'{num}.csv')
            if not os.path.isfile(csv_path):
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f'读取失败: {csv_path} | {e}')
                continue

            cols = [c for c in required_columns if c in df.columns]
            if not cols:
                print(f'⚠️ 无 REQ 列：{csv_path}')
                continue
            if len(cols) < len(required_columns):
                miss = list(set(required_columns) - set(cols))
                print(f'⚠️ 缺失列 {miss} | {csv_path}')

            arr = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype('float32')
            if arr.shape[0] > arr.shape[1]:
                arr = arr.T  # (C,T)
            if arr.shape[1] == 0:
                continue

            Xw, Yw = windowize_from_array(arr, lab, window_size, overlap)   # 先窗口化
            perfile = split_windows_to_5folds(Xw, Yw, n_folds=n_folds)      # 再均分 5 折
            for k in range(n_folds):
                folds[k]['X'].extend(perfile[k]['X'])
                folds[k]['Y'].extend(perfile[k]['Y'])

    return folds

# ------------------------ 构建 (pop, number) 列表（与 MIX_2.py 一致） ------------------------
def get_subject_list(dataset_type):
    if dataset_type == 'MCI':
        return [('MCI', i) for i in range(1, 26+1)]
    if dataset_type == 'HC':
        return [('HC', i) for i in range(1, 42+1)]
    if dataset_type == 'ALL':
        return [('MCI', i) for i in range(1, 26+1)] + [('HC', i) for i in range(1, 42+1)]
    raise ValueError(f'Unknown dataset type: {dataset_type}')

# ------------------------ 可视化 ------------------------
def plot_confusion_matrix(conf_mat, acc, total, side_txt, save_path):
    n = conf_mat.shape[0]
    M = np.zeros((n + 1, n + 1))
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
                 color='white' if v > thresh else 'black', fontsize=24)

    plt.gca().text(1.05, 0.05, side_txt, transform=plt.gca().transAxes,
                   va='top', ha='left', linespacing=1.3, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    plt.ylabel('真实标签', fontsize=14)
    plt.xlabel('预测标签', fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_safe(y_true, y_score, title, save_path):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
    except Exception as e:
        print(f"⚠️ {title} 无法绘制 ROC（原因：{e}），跳过。")
        return
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2, label=f'AUC={auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title, fontsize=14)
    plt.legend(loc='lower right')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_acc_curve(accs, labels, save_path):
    plt.figure(figsize=(8, 4))
    plt.bar(labels, accs)
    plt.ylim(0, 1)
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('各折准确率')
    for i, v in enumerate(accs):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_train_val_curves(train_losses, val_losses, train_accs, val_accs, save_path, dpi=300):
    plt.figure(figsize=(10,4))
    # Loss
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train'); plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss'); plt.legend()
    # Acc
    plt.subplot(1,2,2)
    plt.plot(train_accs, label='Train'); plt.plot(val_accs, label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy'); plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi)
    plt.close()

# ------------------------ 经典模型构建器 ------------------------
def make_classifier(key, args):
    k = key.lower()
    if k == 'nb':
        return GaussianNB()
    if k == 'svm':
        return SVC(
            kernel='rbf',
            C=args.svm_C,
            gamma=args.svm_gamma,
            probability=True,
            class_weight='balanced',
            random_state=args.random_state
        )
    if k == 'knn':
        return 'KNN_CV'
    if k == 'dt':
        return DecisionTreeClassifier(random_state=args.random_state)
    if k == 'rf':
        return RandomForestClassifier(
            n_estimators=args.rf_estimators,
            class_weight='balanced',
            random_state=args.random_state,
            n_jobs=args.n_jobs
        )
    if k == 'ab':
        # 兼容老版 sklearn：使用 base_estimator
        return AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=1, random_state=args.random_state),
            n_estimators=args.ab_estimators,
            learning_rate=args.ab_lr,
            algorithm='SAMME.R',
            random_state=args.random_state
        )
    if k == 'tcn':
        return 'TCN'
    raise ValueError(f"Unknown model key: {key}")

# ------------------------ TCN 模型（PyTorch） ------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super().__init__(); self.chomp_size = chomp_size
    def forward(self, x): return x[..., :-self.chomp_size] if self.chomp_size>0 else x

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1  = nn.ReLU()
        self.drop1  = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2  = nn.ReLU()
        self.drop2  = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu       = nn.ReLU()

    def forward(self, x):
        out = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.relu2(self.chomp2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.25):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_ch = num_inputs if i==0 else num_channels[i-1]
            out_ch = num_channels[i]
            padding = (kernel_size-1) * dilation
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1,
                                        dilation=dilation, padding=padding, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):  # x: (B,C,T)
        return self.network(x)

class TCNClassifier(nn.Module):
    def __init__(self, in_channels, n_classes=2, tcn_channels=[32,64,64], kernel_size=3, dropout=0.25):
        super().__init__()
        self.tcn = TemporalConvNet(in_channels, tcn_channels, kernel_size=kernel_size, dropout=dropout)
        self.head = nn.Linear(tcn_channels[-1], n_classes)

    def forward(self, x):  # x: (B,C,T)
        h = self.tcn(x)                 # (B, C_out, T)
        h = h.mean(dim=-1)              # 全局平均池化 (B, C_out)
        return self.head(h)

# ------------------------ 单进程主流程 ------------------------
def run_once(args):
    # 解析模型与数据集
    all_model_keys = ['nb', 'svm', 'knn', 'dt', 'rf', 'ab', 'tcn']
    models_req = [m.strip().lower() for m in args.models.split(',')]
    if 'all' in models_req:
        models_req = all_model_keys
    for m in models_req:
        if m not in all_model_keys:
            raise ValueError(f'不支持的模型: {m}（可选: {",".join(all_model_keys)} 或 all）')

    datasets_req = [d.strip().upper() for d in args.datasets.split(',')]
    all_datasets = ['MCI', 'HC', 'ALL']
    for d in datasets_req:
        if d not in all_datasets:
            raise ValueError(f'不支持的数据集: {d}（可选: {",".join(all_datasets)}）')

    C = len(REQUIRED_COLUMNS)  # 通道数
    W = args.window_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 跑每个数据集 & 每个模型
    for dtype in datasets_req:
        subjects = get_subject_list(dtype)

        # 与 MIX_2.py 一致：先窗口化再均分 5 折
        folds = build_5fold_windows(
            args.data_root, subjects, REQUIRED_COLUMNS,
            args.window_size, args.overlap, n_folds=5
        )

        # 调试：打印每折标签分布，确认不单类
        for k in range(5):
            yk = np.array(folds[k]['Y'])
            if yk.size == 0:
                print(f"[DEBUG][{dtype}] Fold {k+1}: 空折")
            else:
                u, c = np.unique(yk, return_counts=True)
                print(f"[DEBUG][{dtype}] Fold {k+1}: 标签计数 {dict(zip(u, c))}")

        for key in models_req:
            # 输出：Binary_5K_dependent_68/<model>/<model_dataset>/...
            res_dir = os.path.join(args.result_dir, key, f"{key}_{dtype}")
            os.makedirs(res_dir, exist_ok=True)

            total_conf = np.zeros((2, 2), int)
            y_t_all, y_p_all, y_pr_all = [], [], []
            accs, fold_labels = [], []

            for k in range(5):
                Xte_list = folds[k]['X']; Yte_list = folds[k]['Y']
                Xtr_list, Ytr_list = [], []
                for j in range(5):
                    if j == k: continue
                    Xtr_list += folds[j]['X']
                    Ytr_list += folds[j]['Y']

                print(f"[{key.upper()}][{dtype}] Fold {k+1}/5 | train={len(Ytr_list)} windows, test={len(Yte_list)} windows")
                if not Xtr_list or not Xte_list:
                    print(f"⚠️ Fold {k+1} 数据不足，跳过")
                    continue

                # 经典模型与 TCN 共用同一标准化（对 flatten 后的特征），TCN 再 reshape 回 (C,T)
                Xtr = np.vstack([x.numpy() for x in Xtr_list])  # (N, C*T)
                Xte = np.vstack([x.numpy() for x in Xte_list])
                Ytr = np.array(Ytr_list); Yte = np.array(Yte_list)

                scaler = StandardScaler().fit(Xtr)
                Xtr_s = scaler.transform(Xtr)
                Xte_s = scaler.transform(Xte)

                if key.lower() != 'tcn':
                    # --------- 传统模型路径 ----------
                    clf = make_classifier(key, args)
                    if clf == 'KNN_CV':
                        grid = {'n_neighbors': list(range(1, args.knn_max_k+1))}
                        search = GridSearchCV(KNeighborsClassifier(), grid, cv=5, scoring='accuracy', n_jobs=args.n_jobs)
                        search.fit(Xtr_s, Ytr)
                        model = search.best_estimator_
                        best_k = search.best_params_['n_neighbors']
                        print(f"    → KNN 最佳 K={best_k}")
                    else:
                        model = clf.fit(Xtr_s, Ytr)

                    pred = model.predict(Xte_s)
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(Xte_s)[:, 1]
                    else:
                        if hasattr(model, 'decision_function'):
                            scores = model.decision_function(Xte_s)
                            mn, mx = scores.min(), scores.max()
                            prob = (scores - mn) / (mx - mn + 1e-12)
                        else:
                            prob = np.full(len(Yte), 0.5)

                else:
                    # --------- TCN 路径（PyTorch） ----------
                    # reshape -> (N, C, T)
                    try:
                        Xtr_seq = Xtr_s.reshape(-1, C, W)
                        Xte_seq = Xte_s.reshape(-1, C, W)
                    except Exception:
                        # 若 window_size 与实际不一致，尝试自动推断 T
                        T = Xtr_s.shape[1] // C
                        Xtr_seq = Xtr_s.reshape(-1, C, T)
                        Xte_seq = Xte_s.reshape(-1, C, T)

                    train_ds_full = TensorDataset(torch.tensor(Xtr_seq, dtype=torch.float32),
                                                  torch.tensor(Ytr, dtype=torch.long))
                    # 留 10% 做“早停/挑最佳”的验证
                    n_val = max(1, int(0.1 * len(train_ds_full)))
                    n_train = len(train_ds_full) - n_val
                    if n_train == 0: n_train, n_val = 1, len(train_ds_full) - 1
                    train_ds, val_ds = random_split(train_ds_full, [n_train, n_val])

                    test_ds = TensorDataset(torch.tensor(Xte_seq, dtype=torch.float32),
                                            torch.tensor(Yte, dtype=torch.long))

                    train_loader = DataLoader(train_ds, batch_size=args.tcn_batch, shuffle=True, num_workers=0)
                    val_loader   = DataLoader(val_ds, batch_size=args.tcn_batch, shuffle=False, num_workers=0)
                    test_loader  = DataLoader(test_ds, batch_size=args.tcn_batch, shuffle=False, num_workers=0)

                    model = TCNClassifier(in_channels=C, n_classes=2,
                                          tcn_channels=[32,64,64],
                                          kernel_size=args.tcn_kernel,
                                          dropout=args.tcn_dropout).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.tcn_lr)
                    criterion = nn.CrossEntropyLoss()

                    best_metric = float('inf')  # 监控 val_loss
                    patience = 0
                    train_losses, val_losses, train_accs, val_accs = [], [], [], []

                    fold_dir = os.path.join(res_dir, f"fold_{k+1:02d}")
                    os.makedirs(fold_dir, exist_ok=True)
                    best_path = os.path.join(fold_dir, "best.pt")

                    for ep in range(1, args.tcn_epochs + 1):
                        model.train()
                        tot_loss, corr, tot = 0.0, 0, 0
                        for xb, yb in train_loader:
                            xb, yb = xb.to(device), yb.to(device)
                            logits = model(xb)
                            loss = criterion(logits, yb)
                            optimizer.zero_grad(); loss.backward(); optimizer.step()
                            tot_loss += loss.item() * yb.size(0)
                            pred_b = logits.argmax(dim=1)
                            corr += (pred_b == yb).sum().item()
                            tot  += yb.size(0)
                        train_losses.append(tot_loss / tot if tot else 0.0)
                        train_accs.append(corr / tot if tot else 0.0)

                        model.eval()
                        v_loss, v_corr, v_tot = 0.0, 0, 0
                        with torch.no_grad():
                            for xb, yb in val_loader:
                                xb, yb = xb.to(device), yb.to(device)
                                logits = model(xb)
                                l = criterion(logits, yb)
                                v_loss += l.item() * yb.size(0)
                                p = logits.argmax(dim=1)
                                v_corr += (p == yb).sum().item()
                                v_tot  += yb.size(0)
                        v_loss = v_loss / v_tot if v_tot else 0.0
                        v_acc  = v_corr / v_tot if v_tot else 0.0
                        val_losses.append(v_loss); val_accs.append(v_acc)

                        # early stop on val_loss
                        if v_loss + 1e-6 < best_metric:
                            best_metric = v_loss
                            patience = 0
                            torch.save(model.state_dict(), best_path)
                        else:
                            patience += 1
                            if patience >= args.tcn_patience:
                                print(f"    → TCN early stop @ epoch {ep}, best val_loss={best_metric:.6f}")
                                break

                    # 画训练曲线
                    plot_train_val_curves(train_losses, val_losses, train_accs, val_accs,
                                          os.path.join(fold_dir, "train_val_curves.png"), dpi=300)

                    # 用 best 模型测试
                    if os.path.isfile(best_path):
                        state = torch.load(best_path, map_location=device)
                        model.load_state_dict(state)
                    model.eval()

                    y_true, y_pred, y_prob = [], [], []
                    with torch.no_grad():
                        for xb, yb in test_loader:
                            xb = xb.to(device)
                            logits = model(xb)
                            prob1 = torch.softmax(logits, dim=1)[:, 1]
                            y_true.extend(yb.tolist())
                            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
                            y_prob.extend(prob1.cpu().tolist())
                    pred = np.array(y_pred)
                    prob = np.array(y_prob)

                # ===== 统一评估与可视化（无论传统 or TCN） =====
                acc = accuracy_score(Yte, pred)
                rec = recall_score(Yte, pred, zero_division=0)
                pre = precision_score(Yte, pred, zero_division=0)
                f1s = f1_score(Yte, pred, zero_division=0)
                try:
                    auc = roc_auc_score(Yte, prob)
                except Exception:
                    auc = 0.0

                side_txt = (f"{dtype} {key.upper()} Fold {k+1}\n"
                            f"Acc={acc:.4f}  Rec={rec:.4f}\n"
                            f"Pre={pre:.4f}  F1={f1s:.4f}\n"
                            f"AUC={auc:.4f}")

                fold_dir = os.path.join(res_dir, f"fold_{k+1:02d}")
                os.makedirs(fold_dir, exist_ok=True)

                cm = confusion_matrix(Yte, pred, labels=[0, 1])
                plot_confusion_matrix(cm, acc, len(Yte), side_txt,
                                      os.path.join(fold_dir, "confusion.png"))
                plot_roc_safe(Yte, prob, f"{dtype} {key.upper()} Fold {k+1} ROC",
                              os.path.join(fold_dir, "roc.png"))

                total_conf += cm
                y_t_all += Yte.tolist()
                y_p_all += pred.tolist()
                y_pr_all += prob.tolist()
                accs.append(acc); fold_labels.append(str(k+1))

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

                otxt = (f"{dtype} {key.upper()} Overall\n"
                        f"Acc={oa:.4f}  Rec={orc:.4f}\n"
                        f"Pre={opc:.4f}  F1={of1:.4f}\n"
                        f"AUC={oauc:.4f}")
                plot_confusion_matrix(total_conf, oa, len(y_t_all), otxt,
                                      os.path.join(res_dir, "confusion_overall.png"))
                plot_roc_safe(y_t_all, y_pr_all, f"{dtype} {key.upper()} Overall ROC",
                              os.path.join(res_dir, "roc_overall.png"))

    print("All done.")

# ------------------------ 4/5/6/7 卡 启动器 ------------------------
def launch_4567(args):
    """
    为每个模型启动一个子进程，子进程内部顺序跑 MCI/HC/ALL。
    进程的 CUDA_VISIBLE_DEVICES 轮流设置为 4/5/6/7。
    说明：经典模型主要跑 CPU；TCN 会用到 GPU（若可用）。
    """
    models = [m.strip().lower() for m in (args.launch_models or 'nb,svm,knn,dt,rf,ab,tcn').split(',')]
    gpus = [4, 5, 6, 7]

    procs = []
    for i, m in enumerate(models):
        gpu = gpus[i % len(gpus)]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)

        cmd = [
            sys.executable, os.path.abspath(__file__),
            '--data-root', args.data_root,
            '--result-dir', args.result_dir,
            '--models', m,
            '--datasets', 'MCI,HC,ALL',
            '--window-size', str(args.window_size),
            '--overlap', str(args.overlap),
            '--svm_C', str(args.svm_C),
            '--svm_gamma', str(args.svm_gamma),
            '--rf_estimators', str(args.rf_estimators),
            '--ab_estimators', str(args.ab_estimators),
            '--ab_lr', str(args.ab_lr),
            '--knn_max_k', str(args.knn_max_k),
            '--random_state', str(args.random_state),
            '--n_jobs', str(args.n_jobs),
            '--tcn_epochs', str(args.tcn_epochs),
            '--tcn_batch', str(args.tcn_batch),
            '--tcn_lr', str(args.tcn_lr),
            '--tcn_patience', str(args.tcn_patience),
            '--tcn_kernel', str(args.tcn_kernel),
            '--tcn_dropout', str(args.tcn_dropout),
        ]

        logdir = os.path.join(args.result_dir, m, f'{m}_ALL')
        os.makedirs(logdir, exist_ok=True)
        logfile = os.path.join(logdir, f'run_{m}_gpu{gpu}.log')
        print(f"[LAUNCH] GPU {gpu} ← model={m} | log={logfile}")

        f = open(logfile, 'w', buffering=1)
        p = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        procs.append((p, f))

    # 如需阻塞等待全部结束，取消下面注释：
    # for p, f in procs:
    #     p.wait(); f.close()

# ------------------------ 参数与入口 ------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default=os.path.join('DeepLearning','data_rml'))
    parser.add_argument('--result-dir', type=str, default='Binary_5K_dependent_68')
    parser.add_argument('--models', default='nb,svm,knn,dt,rf,ab', help="nb,svm,knn,dt,rf,ab,tcn 或 all")
    parser.add_argument('--datasets', default='MCI,HC,ALL', help="MCI,HC,ALL（逗号分隔）")
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--n_jobs', type=int, default=-1)

    # 窗口参数（与 MIX_2.py 一致）
    parser.add_argument('--window-size', type=int, default=240)
    parser.add_argument('--overlap', type=float, default=0.0)

    # SVM/RF/AB/KNN 可调
    parser.add_argument('--svm_C', type=float, default=1.0)
    parser.add_argument('--svm_gamma', default='scale')
    parser.add_argument('--rf_estimators', type=int, default=200)
    parser.add_argument('--ab_estimators', type=int, default=5000)
    parser.add_argument('--ab_lr', type=float, default=0.5)
    parser.add_argument('--knn_max_k', type=int, default=20)

    # 启动器选项
    parser.add_argument('--launch4567', action='store_true',
                        help='把不同模型分配到 4/5/6/7 卡上并跑所有数据集（每模型一个进程）')
    parser.add_argument('--launch-models', type=str, default=None,
                        help='自定义要并行启动的模型列表，逗号分隔（默认 nb,svm,knn,dt,rf,ab,tcn）')

    # —— TCN 训练超参 ——
    parser.add_argument('--tcn_epochs', type=int, default=300)
    parser.add_argument('--tcn_batch', type=int, default=256)
    parser.add_argument('--tcn_lr', type=float, default=1e-3)
    parser.add_argument('--tcn_patience', type=int, default=30)
    parser.add_argument('--tcn_kernel', type=int, default=3)
    parser.add_argument('--tcn_dropout', type=float, default=0.25)

    args = parser.parse_args()

    # data_root 兼容 DeepLearning/data_rml
    if not os.path.isdir(args.data_root) and os.path.isdir(os.path.join('DeepLearning', 'data_rml')):
        args.data_root = os.path.join('DeepLearning', 'data_rml')

    return args

def main():
    args = parse_args()
    if args.launch4567:
        launch_4567(args)
    else:
        run_once(args)

if __name__ == '__main__':
    main()
