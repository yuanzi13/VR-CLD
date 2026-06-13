#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mix_rml.py：多模型可选（NB / SVM / KNN / DT / RF / AdaBoost）+ 非独立五折（先窗口再5折）
适配新数据结构：DeepLearning/data_rml/<Population>/<task>/<number>.csv
- Population ∈ {MCI, HC}
- task ∈ {1,2,3,4}  (1→label=0, 2→label=1, 3/4→label=2)
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
from sklearn.preprocessing import label_binarize

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

# ------------------------ 5 折构建：先窗口化，再把窗口按时间顺序切 5 份 ------------------------
def build_5fold_windows(data_root, subjects, required_columns, window_size, overlap, n_folds=5):
    """
    返回 folds 列表（长度 n_folds），每个元素是 {'X': list[tensor(1,-1)], 'Y': list[int]}
    对每个 CSV：先对完整序列进行窗口化 → 得到窗口序列（按时间顺序）→ 将该窗口序列按时间均分为 n_folds 份；
    第 k 份加入 folds[k]。这样保证“先窗口再5折”，且不发生跨份泄漏。
    """
    folds = [{'X': [], 'Y': []} for _ in range(n_folds)]
    task_label = {1: 0, 2: 1, 3: 2, 4: 2}

    for (pop, num) in subjects:
        for task in (1, 2, 3, 4):
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
                arr = arr.T  # (C, T)
            C, T = arr.shape
            if T == 0:
                continue

            lab = task_label[task]

            # 先窗口
            X_all, Y_all = windowize_from_array(arr, lab, window_size, overlap)
            if not X_all:
                continue

            # 后按窗口序列均分 5 折
            nW = len(X_all)
            edges = np.linspace(0, nW, n_folds + 1).astype(int)
            for k in range(n_folds):
                beg, end = edges[k], edges[k + 1]
                if beg >= end:
                    continue
                folds[k]['X'].extend(X_all[beg:end])
                folds[k]['Y'].extend(Y_all[beg:end])

    return folds

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
def plot_confusion_matrix(conf_mat, acc, total, side_txt, save_path, n_classes=3):
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

    plt.figure(figsize=(10, 8))
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
                 color='white' if v > thresh else 'black', fontsize=20)

    plt.gca().text(1.05, 0.05, side_txt, transform=plt.gca().transAxes,
                   va='top', ha='left', linespacing=1.3, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    plt.ylabel('真实标签', fontsize=14)
    plt.xlabel('预测标签', fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_safe(y_true, y_score, title, save_path, n_classes=3):
    """多分类ROC曲线"""
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = roc_auc_score(y_true_bin[:, i], y_score[:, i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = roc_auc_score(y_true_bin, y_score, average="micro")

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = roc_auc_score(y_true_bin, y_score, average="macro")

    plt.figure(figsize=(8, 8))
    colors = ['blue', 'red', 'green']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.4f})', color=color)
    plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average (AUC = {roc_auc["micro"]:.4f})', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-average (AUC = {roc_auc["macro"]:.4f})', linestyle=':', linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(title, fontsize=14); plt.legend(loc="lower right")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_acc_curve(accs, labels, save_path):
    plt.figure(figsize=(8, 4))
    plt.bar(labels, accs)
    plt.ylim(0, 1)
    plt.xlabel('Fold'); plt.ylabel('Accuracy')
    plt.title('各折准确率')
    for i, v in enumerate(accs):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
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
            kernel='rbf', C=args.svm_C, gamma=args.svm_gamma,
            probability=True, class_weight='balanced',
            random_state=args.random_state
        )
    if k == 'knn':
        return 'KNN_CV'
    if k == 'dt':
        return DecisionTreeClassifier(random_state=args.random_state)
    if k == 'rf':
        return RandomForestClassifier(
            n_estimators=args.rf_estimators, class_weight='balanced',
            random_state=args.random_state, n_jobs=args.n_jobs
        )
    if k == 'ab':
        return AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=1, random_state=args.random_state),
            n_estimators=args.ab_estimators, learning_rate=args.ab_lr,
            algorithm='SAMME.R', random_state=args.random_state
        )
    raise ValueError(f"Unknown model key: {key}")

# ------------------------ 计算多分类指标（UAR=macro recall） ------------------------
def calculate_multiclass_metrics(y_true, y_pred, y_score=None):
    """计算多分类指标"""
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)  # ← 改成 macro（UAR）
    f1s = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    auc = 0.0
    if y_score is not None:
        try:
            auc = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')
        except Exception:
            try:
                auc = roc_auc_score(y_true, y_score, multi_class='ovo', average='macro')
            except Exception:
                auc = 0.0
    return {'acc': acc, 'pre': pre, 'rec': rec, 'f1': f1s, 'auc': auc}

# ------------------------ 主程序 ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default=os.path.join('data_rml'))
    parser.add_argument('--result-dir', type=str, default='triple_5K_dependent_68')
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

    # 跑每个数据集 & 每个模型
    for dtype in datasets_req:
        subjects = get_subject_list(dtype)

        folds = build_5fold_windows(
            args.data_root, subjects, REQUIRED_COLUMNS,
            args.window_size, args.overlap, n_folds=5
        )

        for key in models_req:
            res_dir = os.path.join(args.result_dir, f"{key}_{dtype}")
            os.makedirs(res_dir, exist_ok=True)

            total_conf = np.zeros((3, 3), int)
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

                # 概率（用于多分类ROC）
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(Xte_s)
                else:
                    prob = np.zeros((len(Yte), 3))
                    for i, p in enumerate(pred):
                        prob[i, p] = 1.0

                # 指标
                metrics = calculate_multiclass_metrics(Yte, pred, prob)
                acc = metrics['acc']
                rec = metrics['rec']   # UAR (macro)
                pre = metrics['pre']
                f1s = metrics['f1']
                auc = metrics['auc']

                side_txt = (f"{dtype} {key.upper()} Fold {k+1}\n"
                            f"Acc={acc:.4f}  UAR(macro)={rec:.4f}\n"
                            f"Pre(w)= {pre:.4f}  F1(w)={f1s:.4f}\n"
                            f"AUC(macro)={auc:.4f}")

                fold_dir = os.path.join(res_dir, f"fold_{k+1:02d}")
                os.makedirs(fold_dir, exist_ok=True)

                cm = confusion_matrix(Yte, pred, labels=[0, 1, 2])
                plot_confusion_matrix(cm, acc, len(Yte), side_txt,
                                      os.path.join(fold_dir, "confusion.png"), n_classes=3)
                plot_roc_safe(Yte, prob, f"{dtype} {key.upper()} Fold {k+1} ROC",
                              os.path.join(fold_dir, "roc.png"), n_classes=3)

                total_conf += cm
                y_t_all += Yte.tolist()
                y_p_all += pred.tolist()
                y_pr_all.extend(prob.tolist())
                accs.append(acc); fold_labels.append(str(k+1))

            if accs:
                plot_acc_curve(accs, fold_labels, os.path.join(res_dir, "accuracy_across_folds.png"))

            if y_t_all:
                y_pr_all_array = np.array(y_pr_all)
                overall_metrics = calculate_multiclass_metrics(y_t_all, y_p_all, y_pr_all_array)
                oa = overall_metrics['acc']
                orc = overall_metrics['rec']   # UAR
                opc = overall_metrics['pre']
                of1 = overall_metrics['f1']
                oauc = overall_metrics['auc']

                otxt = (f"{dtype} {key.upper()} Overall\n"
                        f"Acc={oa:.4f}  UAR(macro)={orc:.4f}\n"
                        f"Pre(w)={opc:.4f}  F1(w)={of1:.4f}\n"
                        f"AUC(macro)={oauc:.4f}")
                plot_confusion_matrix(total_conf, oa, len(y_t_all), otxt,
                                      os.path.join(res_dir, "confusion_overall.png"), n_classes=3)
                plot_roc_safe(y_t_all, y_pr_all_array, f"{dtype} {key.upper()} Overall ROC",
                              os.path.join(res_dir, "roc_overall.png"), n_classes=3)

                # 保存总体结果到文本文件
                result_file = os.path.join(res_dir, "overall_results.txt")
                with open(result_file, 'w') as f:
                    f.write(f"Overall Results for {key.upper()} on {dtype}\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"Accuracy: {oa:.4f}\n")
                    f.write(f"UAR (macro recall): {orc:.4f}\n")
                    f.write(f"Precision (weighted): {opc:.4f}\n")
                    f.write(f"F1 (weighted): {of1:.4f}\n")
                    f.write(f"AUC (macro): {oauc:.4f}\n")
                    f.write(f"Total Samples: {len(y_t_all)}\n")
                    f.write(f"Total Subjects: {len(subjects)}\n")
                    f.write("\nClass Distribution:\n")
                    for i in range(3):
                        count = sum(1 for label in y_t_all if label == i)
                        f.write(f"Class {i}: {count} samples ({count/len(y_t_all):.2%})\n")

    print("All done.")

if __name__ == '__main__':
    main()
