#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mix_rml_loso.py：多模型可选（NB / SVM / KNN / DT / RF / AdaBoost）+ 留一验证（LOSO）

更新点：
- 输出目录：Binary_5K_dependent_68/<MODEL>/<DATASET>/...
- 对每个 CSV：先窗口化，再按时间顺序均分为5份（除不尽分给前面的份）=> subject_data[sid]['folds'][0..4]
- 从磁盘扫描实际存在的编号，避免只出 MCI 的问题


# 在项目根目录执行（包含 mix_rml_loso.py）
CUDA_VISIBLE_DEVICES=4 python Binary_LOSO_68/mix.py --data-root data_rml --result-dir Binary_LOSO_68 --models nb,dt      --datasets MCI,HC,ALL & \
CUDA_VISIBLE_DEVICES=5 python Binary_LOSO_68/mix.py --data-root data_rml --result-dir Binary_LOSO_68 --models svm         --datasets MCI,HC,ALL & \
CUDA_VISIBLE_DEVICES=6 python Binary_LOSO_68/mix.py --data-root data_rml --result-dir Binary_LOSO_68 --models rf          --datasets MCI,HC,ALL & \
CUDA_VISIBLE_DEVICES=7 python Binary_LOSO_68/mix.py --data-root data_rml --result-dir Binary_LOSO_68 --models knn,ab      --datasets MCI,HC,ALL & \
wait
"""

import os, re, argparse
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold

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

# ------------------------ 工具函数 ------------------------
def split_into_k_parts(n: int, k: int = 5):
    """把 n 个元素按顺序均分成 k 份；余数加给前面的份。返回 [(s,e), ...)，e 不含"""
    if n <= 0:
        return [(0, 0)] * k
    q, r = divmod(n, k)
    sizes = [q + 1 if i < r else q for i in range(k)]
    bounds, cur = [], 0
    for size in sizes:
        bounds.append((cur, cur + size))
        cur += size
    return bounds

def windowize_from_array(arr_ch_t: np.ndarray, label: int, window_size: int = 240, overlap: float = 0.0):
    """arr_ch_t: (channels, time). 返回 list[np.ndarray(1,-1)], list[int]（按时间顺序）"""
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
        X.append(seg.reshape(1, -1))
        Y.append(label)
    return X, Y

def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f'读取失败: {path} | {e}')
        return None

def list_numeric_stems(dirpath):
    """列出目录下 .csv 的纯数字文件名（去扩展名），返回 set[int]"""
    out = set()
    if not os.path.isdir(dirpath):
        return out
    for fn in os.listdir(dirpath):
        if not fn.lower().endswith('.csv'):
            continue
        stem = os.path.splitext(fn)[0]
        if re.fullmatch(r'\d+', stem):
            out.add(int(stem))
    return out

def scan_subject_numbers(data_root, pop):
    """扫描 data_root/<pop>/<1..4>/ 下存在的数字文件名，聚合为编号集合"""
    nums = set()
    for task in (1, 2, 3, 4):
        d = os.path.join(data_root, pop, str(task))
        nums |= list_numeric_stems(d)
    return sorted(nums)

# ------------------------ 构建每个受试者的窗口数据（含5份信息） ------------------------
def build_subject_windows(data_root, subjects, required_columns, window_size, overlap):
    """
    返回：{sid: {'X': list[Tensor(1,-1)], 'Y': list[int], 'folds': [list[(np.ndarray,label)]*5]}}
    """
    subject_data = {}
    task_label = {1: 0, 2: 0, 3: 1, 4: 1}

    for (pop, num) in subjects:
        sid = f"{pop}_{num}"
        subject_data[sid] = {'X': [], 'Y': [], 'folds': [[], [], [], [], []]}
        total_windows = 0

        for task in (1, 2, 3, 4):
            csv_path = os.path.join(data_root, pop, str(task), f'{num}.csv')
            if not os.path.isfile(csv_path):
                continue

            df = safe_read_csv(csv_path)
            if df is None or df.empty:
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

            lab = task_label[task]
            Xs, Ys = windowize_from_array(arr, lab, window_size, overlap)
            if not Xs:
                continue

            # 先窗口化，再按顺序均分为 5 份
            bounds = split_into_k_parts(len(Xs), 5)
            for k, (s, e) in enumerate(bounds):
                for i in range(s, e):
                    subject_data[sid]['folds'][k].append((Xs[i], Ys[i]))

            subject_data[sid]['X'].extend([torch.from_numpy(x) for x in Xs])
            subject_data[sid]['Y'].extend(Ys)
            total_windows += len(Xs)

        print(f"  ▶ {sid}: 窗口={total_windows} | 每份={[len(subject_data[sid]['folds'][k]) for k in range(5)]}")

    return subject_data

# ------------------------ 从磁盘扫描受试者列表 ------------------------
def get_subject_list_from_disk(data_root, dataset_type):
    if dataset_type == 'MCI':
        return [('MCI', n) for n in scan_subject_numbers(data_root, 'MCI')]
    if dataset_type == 'HC':
        return [('HC', n) for n in scan_subject_numbers(data_root, 'HC')]
    if dataset_type == 'ALL':
        mci = scan_subject_numbers(data_root, 'MCI')
        hc  = scan_subject_numbers(data_root, 'HC')
        # 修复：删除多余的 ]（语法错误源头）
        return [('MCI', n) for n in mci] + [('HC', n) for n in hc]
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

    thresh = M.max() / 2 if M.size > 0 else 0.5
    for i, j in np.ndindex(M.shape):
        v = M[i, j]
        if i < n and j < n:
            p = v / total if total else 0
            s = f'{int(v)}\n({p:.2%})'
        else:
            s = f'{v*100:.2f}%'
        plt.text(j, i, s, ha='center', va='center',
                 color='white' if v > thresh else 'black', fontsize=16)

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
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(accs)), accs)
    plt.ylim(0, 1)
    plt.xlabel('受试者ID')
    plt.ylabel('准确率')
    plt.title('各受试者准确率')
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    for i, v in enumerate(accs):
        plt.text(i, min(v + 0.02, 0.98), f'{v:.3f}', ha='center', fontsize=8)
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
    parser.add_argument('--result-dir', type=str, default='Binary_LOSO_68')
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
    parser.add_argument('--ab_estimators', type=int, default=1000)  # 适度即可，过大很慢
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
        subjects = get_subject_list_from_disk(args.data_root, dtype)
        if not subjects:
            print(f"[{dtype}] 未在磁盘找到任何受试者编号，请检查路径：{args.data_root}")
            continue

        print(f"[{dtype}] 扫描到受试者 {len(subjects)} 人，开始构建窗口 ...")
        subject_data = build_subject_windows(
            args.data_root, subjects, REQUIRED_COLUMNS,
            args.window_size, args.overlap
        )

        subject_ids = list(subject_data.keys())
        usable = [sid for sid in subject_ids if len(subject_data[sid]['Y']) > 0]
        print(f"[{dtype}] 可用受试者（至少1个窗口）: {len(usable)} / {len(subject_ids)}")

        for key in models_req:
            model_upper = key.upper()
            # 目录结构：Binary_5K_dependent_68/<MODEL>/<DATASET>/
            res_dir = os.path.join(args.result_dir, model_upper, dtype)
            os.makedirs(res_dir, exist_ok=True)

            total_conf = np.zeros((2, 2), int)
            y_t_all, y_p_all, y_pr_all = [], [], []
            accs, subject_labels = [], []

            # 留一验证循环
            for i, test_subject in enumerate(subject_ids):
                Xte_list = subject_data[test_subject]['X']
                Yte_list = subject_data[test_subject]['Y']

                # 训练集：其他受试者
                Xtr_list, Ytr_list = [], []
                for train_subject in subject_ids:
                    if train_subject == test_subject:
                        continue
                    Xtr_list.extend(subject_data[train_subject]['X'])
                    Ytr_list.extend(subject_data[train_subject]['Y'])

                print(f"[{model_upper}][{dtype}] LOSO {i+1}/{len(subject_ids)}: 测试 {test_subject} | 训练 {len(Ytr_list)} 窗口, 测试 {len(Yte_list)} 窗口")
                if not Xtr_list or not Xte_list:
                    print(f"  ⚠️ {test_subject} 数据不足（训练或测试为空），跳过")
                    continue

                Xtr = np.vstack([x.numpy() if isinstance(x, torch.Tensor) else x for x in Xtr_list])
                Xte = np.vstack([x.numpy() if isinstance(x, torch.Tensor) else x for x in Xte_list])
                Ytr = np.array(Ytr_list); Yte = np.array(Yte_list)

                scaler = StandardScaler().fit(Xtr)
                Xtr_s = scaler.transform(Xtr); Xte_s = scaler.transform(Xte)

                clf = make_classifier(key, args)
                if clf == 'KNN_CV':
                    min_class = min((Ytr == 0).sum(), (Ytr == 1).sum())
                    n_splits = min(5, max(2, min_class))  # 保证可分
                    if n_splits < 2:
                        model = KNeighborsClassifier(n_neighbors=min(5, len(Ytr)))
                    else:
                        grid = {'n_neighbors': list(range(1, min(args.knn_max_k, len(Ytr)) + 1))}
                        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.random_state)
                        search = GridSearchCV(KNeighborsClassifier(), grid, cv=cv, scoring='accuracy', n_jobs=args.n_jobs)
                        search.fit(Xtr_s, Ytr)
                        model = search.best_estimator_
                        print(f"    → KNN 最佳 K={search.best_params_['n_neighbors']}（CV={n_splits}折）")
                else:
                    model = clf.fit(Xtr_s, Ytr)

                pred = model.predict(Xte_s)
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(Xte_s)[:, 1]
                else:
                    if hasattr(model, 'decision_function'):
                        scores = model.decision_function(Xte_s).astype(float)
                        mn, mx = scores.min(), scores.max()
                        prob = (scores - mn) / (mx - mn + 1e-12)
                    else:
                        prob = np.full(len(Yte), 0.5, dtype=float)

                acc = accuracy_score(Yte, pred)
                rec = recall_score(Yte, pred, zero_division=0)
                pre = precision_score(Yte, pred, zero_division=0)
                f1s = f1_score(Yte, pred, zero_division=0)
                try:
                    auc = roc_auc_score(Yte, prob)
                except Exception:
                    auc = 0.0

                side_txt = (f"{dtype} {model_upper} {test_subject}\n"
                            f"Acc={acc:.4f}  Rec={rec:.4f}\n"
                            f"Pre={pre:.4f}  F1={f1s:.4f}\n"
                            f"AUC={auc:.4f}")

                subject_dir = os.path.join(res_dir, f"subject_{test_subject}")
                os.makedirs(subject_dir, exist_ok=True)

                cm = confusion_matrix(Yte, pred, labels=[0, 1])
                plot_confusion_matrix(cm, acc, len(Yte), side_txt,
                                      os.path.join(subject_dir, "confusion.png"))
                plot_roc_safe(Yte, prob, f"{dtype} {model_upper} {test_subject} ROC",
                              os.path.join(subject_dir, "roc.png"))

                total_conf += cm
                y_t_all += Yte.tolist()
                y_p_all += pred.tolist()
                y_pr_all += prob.tolist()
                accs.append(acc)
                subject_labels.append(test_subject)

            if accs:
                plot_acc_curve(accs, subject_labels, os.path.join(res_dir, "accuracy_across_subjects.png"))

            if y_t_all:
                oa = accuracy_score(y_t_all, y_p_all)
                orc = recall_score(y_t_all, y_p_all, zero_division=0)
                opc = precision_score(y_t_all, y_p_all, zero_division=0)
                of1 = f1_score(y_t_all, y_p_all, zero_division=0)
                try:
                    oauc = roc_auc_score(y_t_all, y_pr_all)
                except Exception:
                    oauc = 0.0

                otxt = (f"{dtype} {model_upper} Overall\n"
                        f"Acc={oa:.4f}  Rec={orc:.4f}\n"
                        f"Pre={opc:.4f}  F1={of1:.4f}\n"
                        f"AUC={oauc:.4f}")
                plot_confusion_matrix(total_conf, oa, len(y_t_all), otxt,
                                      os.path.join(res_dir, "confusion_overall.png"))
                plot_roc_safe(y_t_all, y_pr_all, f"{dtype} {model_upper} Overall ROC",
                              os.path.join(res_dir, "roc_overall.png"))

                result_file = os.path.join(res_dir, "overall_results.txt")
                with open(result_file, 'w') as f:
                    f.write(f"Overall Results for {model_upper} on {dtype}\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"Accuracy: {oa:.4f}\n")
                    f.write(f"Recall: {orc:.4f}\n")
                    f.write(f"Precision: {opc:.4f}\n")
                    f.write(f"F1 Score: {of1:.4f}\n")
                    f.write(f"AUC: {oauc:.4f}\n")
                    f.write(f"Total Samples: {len(y_t_all)}\n")
                    f.write(f"Total Subjects: {len(subject_ids)}\n")
            else:
                print(f"[{dtype}][{model_upper}] 没有可汇总的样本（可能所有受试者都被跳过）")

    print("All done.")

if __name__ == '__main__':
    main()
