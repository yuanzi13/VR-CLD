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

def merge_subject_folds(subject_data, usable_subjects, n_folds=5):
    """
    将每名受试者已经划分好的5份窗口，按相同fold编号汇总。

    最终：
        folds[0] = 所有受试者各自第1份窗口
        folds[1] = 所有受试者各自第2份窗口
        ...
    """
    folds = [
        {'X': [], 'Y': []}
        for _ in range(n_folds)
    ]

    for sid in usable_subjects:
        subject_folds = subject_data[sid]['folds']

        for k in range(n_folds):
            for x, y in subject_folds[k]:
                folds[k]['X'].append(x)
                folds[k]['Y'].append(y)

    return folds



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
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(accs)), accs)
    plt.ylim(0, 1)

    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy across 5 folds')

    plt.xticks(
        range(len(labels)),
        labels
    )

    for i, value in enumerate(accs):
        plt.text(
            i,
            min(value + 0.02, 0.98),
            f'{value:.3f}',
            ha='center',
            fontsize=9
        )

    plt.tight_layout()
    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )
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
    parser.add_argument('--result-dir', type=str, default='Binary_5K_dependent_68')
    parser.add_argument('--models', default='svm,knn,dt', help="nb,svm,knn,dt,rf,ab 或 all")
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
        subject_data = merge_subject_folds(
            args.data_root, subjects, REQUIRED_COLUMNS,
            args.window_size, args.overlap
        )

    subject_ids = list(subject_data.keys())
    
    usable = [
        sid
        for sid in subject_ids
        if len(subject_data[sid]['Y']) > 0
    ]
    
    print(
        f"[{dtype}] 可用受试者："
        f"{len(usable)} / {len(subject_ids)}"
    )
    
    if not usable:
        print(f"[{dtype}] 没有可用受试者，跳过")
        continue
    
    # 将各受试者的第1～5份窗口分别汇总
    folds = merge_subject_folds(
        subject_data,
        usable,
        n_folds=5
    )
    
    # 检查各折样本与标签分布
    for k in range(5):
        fold_y = np.asarray(folds[k]['Y'])
    
        if fold_y.size == 0:
            print(
                f"[DEBUG][{dtype}] "
                f"Fold {k + 1}: 空折"
            )
        else:
            labels, counts = np.unique(
                fold_y,
                return_counts=True
            )
    
            distribution = dict(
                zip(labels.tolist(), counts.tolist())
            )
    
            print(
                f"[DEBUG][{dtype}] "
                f"Fold {k + 1}: "
                f"窗口={len(fold_y)}, "
                f"标签={distribution}"
            )
    
    for key in models_req:
        model_upper = key.upper()
    
        res_dir = os.path.join(
            args.result_dir,
            model_upper,
            dtype
        )
        os.makedirs(res_dir, exist_ok=True)
    
        total_conf = np.zeros((2, 2), dtype=int)
    
        y_t_all = []
        y_p_all = []
        y_pr_all = []
    
        accs = []
        fold_labels = []
    
        # =========================
        # subject-dependent 5-fold
        # =========================
        for k in range(5):
            # 第k折作为测试集
            Xte_list = folds[k]['X']
            Yte_list = folds[k]['Y']
    
            # 其余4折作为训练集
            Xtr_list = []
            Ytr_list = []
    
            for j in range(5):
                if j == k:
                    continue
    
                Xtr_list.extend(folds[j]['X'])
                Ytr_list.extend(folds[j]['Y'])
    
            print(
                f"[{model_upper}][{dtype}] "
                f"Fold {k + 1}/5 | "
                f"train={len(Ytr_list)} windows, "
                f"test={len(Yte_list)} windows"
            )
    
            if not Xtr_list or not Xte_list:
                print(
                    f"⚠️ Fold {k + 1} "
                    f"训练集或测试集为空，跳过"
                )
                continue
    
            # folds中保存的是np.ndarray
            Xtr = np.vstack(Xtr_list)
            Xte = np.vstack(Xte_list)
    
            Ytr = np.asarray(
                Ytr_list,
                dtype=np.int64
            )
            Yte = np.asarray(
                Yte_list,
                dtype=np.int64
            )
    
            # 每个外层fold只用训练折拟合Scaler
            scaler = StandardScaler()
            scaler.fit(Xtr)
    
            Xtr_s = scaler.transform(Xtr)
            Xte_s = scaler.transform(Xte)
    
            # ---------------------
            # 建立和训练分类器
            # ---------------------
            clf = make_classifier(key, args)
    
            if isinstance(clf, str) and clf == 'KNN_CV':
                min_class_count = min(
                    int((Ytr == 0).sum()),
                    int((Ytr == 1).sum())
                )
    
                n_splits = min(
                    5,
                    min_class_count
                )
    
                if n_splits < 2:
                    best_k = min(
                        5,
                        len(Ytr)
                    )
    
                    model = KNeighborsClassifier(
                        n_neighbors=best_k
                    )
                    model.fit(Xtr_s, Ytr)
    
                    print(
                        f"    → KNN使用固定K={best_k}"
                    )
    
                else:
                    max_k = min(
                        args.knn_max_k,
                        len(Ytr)
                    )
    
                    grid = {
                        'n_neighbors': list(
                            range(1, max_k + 1)
                        )
                    }
    
                    cv = StratifiedKFold(
                        n_splits=n_splits,
                        shuffle=True,
                        random_state=args.random_state
                    )
    
                    search = GridSearchCV(
                        estimator=KNeighborsClassifier(),
                        param_grid=grid,
                        cv=cv,
                        scoring='accuracy',
                        n_jobs=args.n_jobs
                    )
    
                    search.fit(Xtr_s, Ytr)
                    model = search.best_estimator_
    
                    print(
                        f"    → KNN最佳K="
                        f"{search.best_params_['n_neighbors']} "
                        f"（内部CV={n_splits}折）"
                    )
    
            else:
                model = clf
                model.fit(Xtr_s, Ytr)
    
            # ---------------------
            # 外层测试折预测
            # ---------------------
            pred = model.predict(Xte_s)
    
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(
                    Xte_s
                )[:, 1]
    
            elif hasattr(model, 'decision_function'):
                scores = model.decision_function(
                    Xte_s
                ).astype(float)
    
                score_min = scores.min()
                score_max = scores.max()
    
                prob = (
                    scores - score_min
                ) / (
                    score_max - score_min + 1e-12
                )
    
            else:
                prob = np.full(
                    len(Yte),
                    0.5,
                    dtype=float
                )
    
            # ---------------------
            # 本折指标
            # ---------------------
            acc = accuracy_score(
                Yte,
                pred
            )
    
            rec = recall_score(
                Yte,
                pred,
                average='binary',
                pos_label=1,
                zero_division=0
            )
    
            pre = precision_score(
                Yte,
                pred,
                average='binary',
                pos_label=1,
                zero_division=0
            )
    
            f1s = f1_score(
                Yte,
                pred,
                average='binary',
                pos_label=1,
                zero_division=0
            )
    
            try:
                auc = roc_auc_score(
                    Yte,
                    prob
                )
            except ValueError:
                auc = np.nan
    
            side_txt = (
                f"{dtype} {model_upper} "
                f"Fold {k + 1}\n"
                f"Acc={acc:.4f}  "
                f"Rec={rec:.4f}\n"
                f"Pre={pre:.4f}  "
                f"F1={f1s:.4f}\n"
                f"AUC={auc:.4f}"
            )
    
            fold_dir = os.path.join(
                res_dir,
                f"fold_{k + 1:02d}"
            )
            os.makedirs(fold_dir, exist_ok=True)
    
            cm = confusion_matrix(
                Yte,
                pred,
                labels=[0, 1]
            )
    
            plot_confusion_matrix(
                cm,
                acc,
                len(Yte),
                side_txt,
                os.path.join(
                    fold_dir,
                    'confusion.png'
                )
            )
    
            plot_roc_safe(
                Yte,
                prob,
                (
                    f"{dtype} {model_upper} "
                    f"Fold {k + 1} ROC"
                ),
                os.path.join(
                    fold_dir,
                    'roc.png'
                )
            )
    
            # 汇总五折结果
            total_conf += cm
    
            y_t_all.extend(
                Yte.tolist()
            )
            y_p_all.extend(
                pred.tolist()
            )
            y_pr_all.extend(
                prob.tolist()
            )
    
            accs.append(acc)
            fold_labels.append(
                f"Fold {k + 1}"
            )
    
        # 五折准确率
        if accs:
            plot_acc_curve(
                accs,
                fold_labels,
                os.path.join(
                    res_dir,
                    'accuracy_across_folds.png'
                )
            )
    
        # =====================
        # 汇总五折测试窗口指标
        # =====================
        if y_t_all:
            oa = accuracy_score(
                y_t_all,
                y_p_all
            )
    
            orc = recall_score(
                y_t_all,
                y_p_all,
                average='binary',
                pos_label=1,
                zero_division=0
            )
    
            opc = precision_score(
                y_t_all,
                y_p_all,
                average='binary',
                pos_label=1,
                zero_division=0
            )
    
            of1 = f1_score(
                y_t_all,
                y_p_all,
                average='binary',
                pos_label=1,
                zero_division=0
            )
    
            try:
                oauc = roc_auc_score(
                    y_t_all,
                    y_pr_all
                )
            except ValueError:
                oauc = np.nan
    
            otxt = (
                f"{dtype} {model_upper} "
                f"5-Fold Overall\n"
                f"Acc={oa:.4f}  "
                f"Rec={orc:.4f}\n"
                f"Pre={opc:.4f}  "
                f"F1={of1:.4f}\n"
                f"AUC={oauc:.4f}"
            )
    
            plot_confusion_matrix(
                total_conf,
                oa,
                len(y_t_all),
                otxt,
                os.path.join(
                    res_dir,
                    'confusion_overall.png'
                )
            )
    
            plot_roc_safe(
                y_t_all,
                y_pr_all,
                (
                    f"{dtype} {model_upper} "
                    f"5-Fold Overall ROC"
                ),
                os.path.join(
                    res_dir,
                    'roc_overall.png'
                )
            )
    
            result_file = os.path.join(
                res_dir,
                'overall_results.txt'
            )
    
            with open(
                result_file,
                'w',
                encoding='utf-8'
            ) as file:
                file.write(
                    f"5-Fold Overall Results for "
                    f"{model_upper} on {dtype}\n"
                )
                file.write("=" * 50 + "\n")
                file.write(f"Accuracy: {oa:.4f}\n")
                file.write(f"Recall: {orc:.4f}\n")
                file.write(f"Precision: {opc:.4f}\n")
                file.write(f"F1 Score: {of1:.4f}\n")
                file.write(f"AUC: {oauc:.4f}\n")
                file.write(
                    f"Total Test Windows: "
                    f"{len(y_t_all)}\n"
                )
                file.write(
                    f"Completed Folds: "
                    f"{len(accs)}\n"
                )
    
        print("All done.")

if __name__ == '__main__':
    main()
