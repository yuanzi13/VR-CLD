#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mix_rml.py：多模型可选（NB / SVM / KNN / DT / RF / AdaBoost）+ LOSO 留一验证
适配新数据结构：DeepLearning/data_rml/<Population>/<task>/<number>.csv
- Population ∈ {MCI, HC}
- task ∈ {1,2,3,4}  (1/2→label=0, 3/4→label=1)
- number: MCI=1..26, HC=1..42（各自独立编号）

运行示例：
python mix_rml.py --data-root DeepLearning/data_rml --models all --datasets MCI,HC,ALL
python mix_rml.py --data-root DeepLearning/data_rml --models svm,rf --datasets ALL
"""

import os, argparse
import numpy as np
import pandas as pd
import torch
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

# ------------------------ 数据窗口化 ------------------------
def windowize_from_array(arr_ch_t: np.ndarray, label: int, window_size: int = 240, overlap: float = 0.0):
    """arr_ch_t: (channels, time). 返回 list[torch.Tensor(1,-1)], list[int]"""
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
        X.append(torch.from_numpy(seg).reshape(1, -1))
        Y.append(label)
    return X, Y

# ------------------------ LOSO：按受试者聚合后窗口化 ------------------------
def build_loso_subject_windows(data_root, subjects, required_columns, window_size, overlap):
    """
    返回 dict: key=(pop,num)，value={'X':list[tensor(1,-1)], 'Y':list[int]}
    将该 (pop,num) 的所有 task(1..4) 读取、窗口化后合并。
    """
    task_label = {1:0, 2:0, 3:1, 4:1}
    per_subj = {}

    for (pop, num) in subjects:
        X_all, Y_all = [], []
        for task in (1,2,3,4):
            csv_path = os.path.join(data_root, pop, str(task), f'{num}.csv')
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
            C, T = arr.shape
            if T == 0:
                continue

            lab = task_label[task]
            Xs, Ys = windowize_from_array(arr, lab, window_size, overlap)
            X_all.extend(Xs)
            Y_all.extend(Ys)

        per_subj[(pop, num)] = {'X': X_all, 'Y': Y_all}

    return per_subj

# ------------------------ 构建 (pop, number) 列表 ------------------------
def get_subject_list(dataset_type):
    if dataset_type == 'MCI':
        return [('MCI', i) for i in range(1, 26+1)]
    if dataset_type == 'HC':
        return [('HC', i) for i in range(1, 42+1)]
    if dataset_type == 'ALL':
        return [('MCI', i) for i in range(1, 26+1)] + [('HC', i) for i in range(1, 42+1)]
    raise ValueError(f'Unknown dataset type: {dataset_type}')

# ------------------------ 画图 ------------------------
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
    plt.figure(figsize=(10, 4))
    plt.bar(labels, accs)
    plt.ylim(0, 1)
    plt.xlabel('Fold (Subject)')
    plt.ylabel('Accuracy')
    plt.title('LOSO 各折准确率')
    for i, v in enumerate(accs):
        plt.text(i, min(0.98, v + 0.02), f'{v:.3f}', ha='center', fontsize=9, rotation=90)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

# ------------------------ 构建分类器 ------------------------
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
        return AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=1, random_state=args.random_state),
            n_estimators=args.ab_estimators,
            learning_rate=args.ab_lr,
            algorithm='SAMME.R',
            random_state=args.random_state
        )
    raise ValueError(f"Unknown model key: {key}")

# ------------------------ 主程序 ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default=os.path.join('DeepLearning','data_rml'))
    parser.add_argument('--result-dir', type=str, default='Binary_LOSO_68')   # ← 改为 LOSO 目录
    parser.add_argument('--models', default='nb,svm,knn,dt,rf,ab', help="nb,svm,knn,dt,rf,ab 或 all")
    parser.add_argument('--datasets', default='MCI,HC,ALL', help="MCI,HC,ALL（逗号分隔）")
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--n_jobs', type=int, default=-1)

    # 窗口参数
    parser.add_argument('--window-size', type=int, default=240)
    parser.add_argument('--overlap', type=float, default=0.0)

    # SVM/RF/AB/KNN 可调
    parser.add_argument('--svm_C', type=float, default=1.0)
    parser.add_argument('--svm_gamma', default='scale')
    parser.add_argument('--rf_estimators', type=int, default=200)
    parser.add_argument('--ab_estimators', type=int, default=5000)
    parser.add_argument('--ab_lr', type=float, default=0.5)
    parser.add_argument('--knn_max_k', type=int, default=20)

    args = parser.parse_args()

    # 解析模型与数据集
    all_model_keys = ['nb', 'svm', 'knn', 'dt', 'rf', 'ab']
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

    # 跑每个数据集 & 每个模型（LOSO）
    for dtype in datasets_req:
        subjects = get_subject_list(dtype)

        # 每个受试者聚合窗口
        subj_windows = build_loso_subject_windows(
            args.data_root, subjects, REQUIRED_COLUMNS,
            args.window_size, args.overlap
        )

        for key in models_req:
            res_dir = os.path.join(args.result_dir, f"{key}_{dtype}")
            os.makedirs(res_dir, exist_ok=True)

            total_conf = np.zeros((2, 2), int)
            y_t_all, y_p_all, y_pr_all = [], [], []
            accs, fold_labels = [], []

            # 每个 (pop,num) 做一折
            for (pop, num) in subjects:
                te_pack = subj_windows.get((pop, num), {'X': [], 'Y': []})
                Xte_list = te_pack['X']; Yte_list = te_pack['Y']
                Xtr_list, Ytr_list = [], []

                # 训练集：其余所有受试者
                for (pop2, num2), pack in subj_windows.items():
                    if (pop2, num2) == (pop, num):
                        continue
                    Xtr_list += pack['X']
                    Ytr_list += pack['Y']

                fold_name = f"{pop}_{num:02d}"
                print(f"[{key.upper()}][{dtype}] LOSO {fold_name} | train={len(Ytr_list)} windows, test={len(Yte_list)} windows")

                if not Xtr_list or not Xte_list:
                    print(f"⚠️ LOSO {fold_name} 数据不足，跳过")
                    continue

                Xtr = np.vstack([x.numpy() for x in Xtr_list])
                Xte = np.vstack([x.numpy() for x in Xte_list])
                Ytr = np.array(Ytr_list); Yte = np.array(Yte_list)

                scaler = StandardScaler().fit(Xtr)
                Xtr_s = scaler.transform(Xtr); Xte_s = scaler.transform(Xte)

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

                acc = accuracy_score(Yte, pred)
                rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
                pre = precision_score(Yte, pred, zero_division=0)
                f1s = f1_score(Yte, pred, zero_division=0)
                try:
                    auc = roc_auc_score(Yte, prob)
                except Exception:
                    auc = 0.0

                side_txt = (f"{dtype} {key.upper()} LOSO {fold_name}\n"
                            f"Acc={acc:.4f}  Rec={rec:.4f}\n"
                            f"Pre={pre:.4f}  F1={f1s:.4f}\n"
                            f"AUC={auc:.4f}")

                fold_dir = os.path.join(res_dir, f"fold_{fold_name}")
                os.makedirs(fold_dir, exist_ok=True)

                cm = confusion_matrix(Yte, pred, labels=[0, 1])
                plot_confusion_matrix(cm, acc, len(Yte), side_txt,
                                      os.path.join(fold_dir, "confusion.png"))
                plot_roc_safe(Yte, prob, f"{dtype} {key.upper()} LOSO {fold_name} ROC",
                              os.path.join(fold_dir, "roc.png"))

                total_conf += cm
                y_t_all += Yte.tolist()
                y_p_all += pred.tolist()
                y_pr_all += prob.tolist()
                accs.append(acc); fold_labels.append(fold_name)

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

                otxt = (f"{dtype} {key.UPPER()} Overall\n"
                        f"Acc={oa:.4f}  Rec={orc:.4f}\n"
                        f"Pre={opc:.4f}  F1={of1:.4f}\n"
                        f"AUC={oauc:.4f}")
                plot_confusion_matrix(total_conf, oa, len(y_t_all), otxt,
                                      os.path.join(res_dir, "confusion_overall.png"))
                plot_roc_safe(y_t_all, y_pr_all, f"{dtype} {key.upper()} Overall ROC",
                              os.path.join(res_dir, "roc_overall.png"))

    print("All done.")

if __name__ == '__main__':
    main()
