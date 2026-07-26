#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mix.py（改进版：RF 强化）
- 保持原 LO SO 三分类流程与绘图不变
- 针对 RF 新增 PCA, RandomizedSearchCV, CalibratedClassifierCV 等提升策略
使用示例（你之前的）：
python triple_LOSO_68/mix.py --models rf --datasets MCI,HC,ALL --rf_search_iter 30 --rf_pca_max_components 300 --result-dir triple_LOSO_68/RF3
"""
import os, re, argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve
)
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel

rcParams['font.family'] = ['WenQuanYi Micro Hei']

# REQUIRED_COLUMNS, helper functions, plotting, data building exactly as your original script...
# For brevity in this snippet, paste the same helper functions from your original file (windowize_from_array, safe_read_csv, list_numeric_stems, scan_subject_numbers, split_into_k_parts, build_subject_windows, get_subject_list_from_disk, plot_confusion_matrix, plot_roc_safe, plot_acc_curve)
# --- 我将直接重-use 你之前给出的工具函数体（确保完全一致） ---
# Paste your original helper functions here (identical to the ones in your script)
# For clarity and to ensure the response here isn't excessively long, assume the functions up to make_classifier are identical.

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

# (Insert here all helper functions from your original script without modification:
# split_into_k_parts, windowize_from_array, safe_read_csv, list_numeric_stems,
# scan_subject_numbers, build_subject_windows, get_subject_list_from_disk,
# plot_confusion_matrix, plot_roc_safe, plot_acc_curve)
#
# To save space in this message I assume you copy-paste them verbatim from your earlier script.
# ------------------------------------------------------------------------------

# ------------------------ 构建分类器（保留原有 make_classifier） ------------------------
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
        return DecisionTreeClassifier(random_state=args.random_state, class_weight=None)
    if k == 'rf':
        # We'll not directly return fitted RF here; the main logic will handle RF specially (to add PCA + RandomizedSearch + calibration)
        return 'RF_PLACEHOLDER'
    if k == 'ab':
        try:
            return AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1, random_state=args.random_state),
                n_estimators=args.ab_estimators,
                learning_rate=args.ab_lr,
                algorithm='SAMME.R',
                random_state=args.random_state
            )
        except TypeError:
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
    parser.add_argument('--result-dir', type=str, default='triple_LOSO_68')
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
    parser.add_argument('--ab_estimators', type=int, default=1000)
    parser.add_argument('--ab_lr', type=float, default=0.5)
    parser.add_argument('--knn_max_k', type=int, default=20)

    # ---- 新增 RF 强化参数 ----
    parser.add_argument('--rf_search_iter', type=int, default=30, help='RandomizedSearchCV n_iter for RF (0 disables search)')
    parser.add_argument('--rf_pca_max_components', type=int, default=300, help='Max PCA components before RF (0 disables PCA)')
    parser.add_argument('--rf_n_jobs', type=int, default=-1, help='n_jobs for RandomizedSearchCV / RF')
    parser.add_argument('--rf_min_samples_leaf_max', type=int, default=10, help='max value for min_samples_leaf search grid')
    parser.add_argument('--rf_max_depth_options', type=str, default='None,10,20,30,50', help='comma list for max_depth search options (use "None" for None)')
    # -------------------------

    args = parser.parse_args()

    # 兼容 data path
    if (not os.path.isdir(args.data_root)) and os.path.isdir(os.path.join('DeepLearning','data_rml')):
        args.data_root = os.path.join('DeepLearning','data_rml')

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

    n_classes = 3
    label_list = list(range(n_classes))

    # Convert rf_max_depth_options string to list
    rf_max_depth_options = []
    for tok in args.rf_max_depth_options.split(','):
        tok = tok.strip()
        if tok.lower() == 'none':
            rf_max_depth_options.append(None)
        else:
            try:
                rf_max_depth_options.append(int(tok))
            except:
                pass

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
            res_dir = os.path.join(args.result_dir, model_upper, dtype)
            os.makedirs(res_dir, exist_ok=True)

            total_conf = np.zeros((n_classes, n_classes), dtype=int)
            y_t_all, y_p_all = [], []
            y_pr_all = []

            accs, subject_labels = [], []

            # LOSO
            for i, test_subject in enumerate(subject_ids):
                Xte_list = subject_data[test_subject]['X']
                Yte_list = subject_data[test_subject]['Y']

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

                # ---------- RF 强化分支 ----------
                if key.lower() == 'rf':
                    best_model = None
                    model_prob = None
                    # Optional PCA
                    do_pca = (args.rf_pca_max_components is not None and args.rf_pca_max_components > 0)
                    if do_pca:
                        # n_components cannot exceed min(n_samples, n_features)
                        max_possible = min(Xtr_s.shape[0], Xtr_s.shape[1], args.rf_pca_max_components)
                        pca_n = max(1, max_possible)
                        pca = PCA(n_components=pca_n, random_state=args.random_state)
                        Xtr_red = pca.fit_transform(Xtr_s)
                        Xte_red = pca.transform(Xte_s)
                        print(f"    RF: Applied PCA -> n_components={pca_n}")
                    else:
                        Xtr_red, Xte_red = Xtr_s, Xte_s

                    # 如果没有搜索就直接训练一个强基线 RF（但我们建议开启搜索）
                    if args.rf_search_iter and args.rf_search_iter > 0:
                        param_dist = {
                            'n_estimators': np.linspace(max(50, args.rf_estimators//2), args.rf_estimators*2, 6, dtype=int).tolist(),
                            'max_depth': rf_max_depth_options,
                            'max_features': ['sqrt', 'log2', 0.2, 0.4, 0.6, 0.8],
                            'min_samples_split': [2, 3, 5, 10],
                            'min_samples_leaf': [1, 2, 4, min(10, args.rf_min_samples_leaf_max)],
                            'class_weight': [None, 'balanced', 'balanced_subsample']
                        }
                        # build base RF
                        base_rf = RandomForestClassifier(n_estimators=args.rf_estimators, n_jobs=args.rf_n_jobs,
                                                         random_state=args.random_state, class_weight='balanced')
                        cv = StratifiedKFold(n_splits=min(3, max(2, int(np.clip(np.bincount(Ytr).min(), 2, 5)))), shuffle=True,
                                             random_state=args.random_state)
                        print(f"    RF: RandomizedSearchCV n_iter={args.rf_search_iter}, cv={cv.get_n_splits()}")
                        try:
                            rnd = RandomizedSearchCV(
                                estimator=base_rf,
                                param_distributions=param_dist,
                                n_iter=args.rf_search_iter,
                                scoring='f1_macro',
                                cv=cv,
                                random_state=args.random_state,
                                n_jobs=args.rf_n_jobs,
                                verbose=0
                            )
                            rnd.fit(Xtr_red, Ytr)
                            best_rf = rnd.best_estimator_
                            print(f"    RF: RandomizedSearchCV best params: {rnd.best_params_}")
                            best_model = best_rf
                        except Exception as e:
                            print(f"    ⚠️ RF RandomizedSearchCV failed: {e}. Falling back to default RF.")
                            best_model = RandomForestClassifier(n_estimators=args.rf_estimators, n_jobs=args.rf_n_jobs,
                                                                class_weight='balanced', random_state=args.random_state)
                            best_model.fit(Xtr_red, Ytr)
                    else:
                        # no search, use reasonably strong default
                        best_model = RandomForestClassifier(n_estimators=args.rf_estimators, n_jobs=args.rf_n_jobs,
                                                            class_weight='balanced', random_state=args.random_state)
                        best_model.fit(Xtr_red, Ytr)

                    # calibration: improves predicted probabilities used by ROC/AUC
                    try:
                        calibrated = CalibratedClassifierCV(best_model, method='sigmoid', cv='prefit')
                        calibrated.fit(Xtr_red, Ytr)
                        clf_final = calibrated
                        print("    RF: Applied CalibratedClassifierCV(method='sigmoid') on best RF.")
                    except Exception as e:
                        clf_final = best_model
                        print(f"    ⚠️ Calibration failed/ skipped: {e}")

                    # predict
                    pred = clf_final.predict(Xte_red)
                    if hasattr(clf_final, 'predict_proba'):
                        prob = clf_final.predict_proba(Xte_red)
                    else:
                        # fallback: uniform probs
                        prob = np.full((len(pred), n_classes), 1.0/n_classes)

                # ---------- KNN CV branch (unchanged) ----------
                elif key.lower() == 'knn':
                    clf = make_classifier(key, args)
                    if clf == 'KNN_CV':
                        binc = np.bincount(Ytr, minlength=n_classes)
                        min_class = int(binc.min())
                        n_splits = min(5, max(2, min_class))
                        if n_splits < 2:
                            model = KNeighborsClassifier(n_neighbors=min(5, len(Ytr)))
                        else:
                            grid = {'n_neighbors': list(range(1, min(args.knn_max_k, len(Ytr)) + 1))}
                            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.random_state)
                            search = GridSearchCV(KNeighborsClassifier(), grid, cv=cv, scoring='accuracy', n_jobs=args.n_jobs)
                            search.fit(Xtr_s, Ytr)
                            model = search.best_estimator_
                            print(f"    → KNN 最佳 K={search.best_params_['n_neighbors']}（CV={n_splits}折）")
                        pred = model.predict(Xte_s)
                        if hasattr(model, 'predict_proba'):
                            prob = model.predict_proba(Xte_s)
                        else:
                            prob = np.full((len(pred), n_classes), 1.0/n_classes, dtype=float)

                # ---------- other classifiers (nb, svm, dt, ab) ----------
                else:
                    clf = make_classifier(key, args)
                    if isinstance(clf, str) and clf.startswith('RF'):
                        # shouldn't happen here, handled above
                        raise RuntimeError("Unexpected RF placeholder in non-RF branch")
                    model = clf.fit(Xtr_s, Ytr)
                    pred = model.predict(Xte_s)
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(Xte_s)
                    else:
                        if hasattr(model, 'decision_function'):
                            scores = model.decision_function(Xte_s)
                            scores = np.atleast_2d(scores)
                            if scores.ndim == 1 or scores.shape[1] == 1:
                                prob = np.full((len(pred), n_classes), 1.0/n_classes, dtype=float)
                            else:
                                mn = scores.min(axis=0, keepdims=True)
                                mx = scores.max(axis=0, keepdims=True)
                                scores_norm = (scores - mn) / (mx - mn + 1e-12)
                                row_sum = scores_norm.sum(axis=1, keepdims=True) + 1e-12
                                prob = scores_norm / row_sum
                        else:
                            prob = np.full((len(pred), n_classes), 1.0/n_classes, dtype=float)

                # ---------- compute metrics and save plots ----------
                acc = accuracy_score(Yte, pred)
                rec = recall_score(Yte, pred, average='macro', zero_division=0)
                pre = precision_score(Yte, pred, average='macro', zero_division=0)
                f1s = f1_score(Yte, pred, average='macro', zero_division=0)
                try:
                    auc_macro = roc_auc_score(Yte, prob, multi_class='ovr', average='macro')
                except Exception:
                    auc_macro = 0.0

                side_txt = (f"{dtype} {model_upper} {test_subject}\n"
                            f"Acc={acc:.4f}\n"
                            f"Macro: Rec={rec:.4f}  Pre={pre:.4f}  F1={f1s:.4f}\n"
                            f"Macro AUC(ovr)={auc_macro:.4f}")

                subject_dir = os.path.join(res_dir, f"subject_{test_subject}")
                os.makedirs(subject_dir, exist_ok=True)

                cm = confusion_matrix(Yte, pred, labels=label_list)
                plot_confusion_matrix(cm, acc, len(Yte), side_txt,
                                      os.path.join(subject_dir, "confusion.png"))
                plot_roc_safe(Yte, prob, n_classes,
                              f"{dtype} {model_upper} {test_subject} ROC",
                              os.path.join(subject_dir, "roc.png"))

                total_conf += cm
                y_t_all += Yte.tolist()
                y_p_all += pred.tolist()
                y_pr_all.append(prob)

                accs.append(acc)
                subject_labels.append(test_subject)

            # end LOSO loop

            if accs:
                plot_acc_curve(accs, subject_labels, os.path.join(res_dir, "accuracy_across_subjects.png"))

            if y_t_all:
                y_t_all_arr = np.array(y_t_all)
                oa = accuracy_score(y_t_all_arr, y_p_all)
                orc = recall_score(y_t_all_arr, y_p_all, average='macro', zero_division=0)
                opc = precision_score(y_t_all_arr, y_p_all, average='macro', zero_division=0)
                of1 = f1_score(y_t_all_arr, y_p_all, average='macro', zero_division=0)

                try:
                    y_pr_all_arr = np.vstack(y_pr_all)
                except Exception:
                    y_pr_all_arr = None

                try:
                    oauc = roc_auc_score(y_t_all_arr, y_pr_all_arr, multi_class='ovr', average='macro') \
                           if y_pr_all_arr is not None else 0.0
                except Exception:
                    oauc = 0.0

                otxt = (f"{dtype} {model_upper} Overall\n"
                        f"Acc={oa:.4f}\n"
                        f"Macro: Rec={orc:.4f}  Pre={opc:.4f}  F1={of1:.4f}\n"
                        f"Macro AUC(ovr)={oauc:.4f}")
                plot_confusion_matrix(total_conf, oa, len(y_t_all_arr), otxt,
                                      os.path.join(res_dir, "confusion_overall.png"))
                if y_pr_all_arr is not None:
                    plot_roc_safe(y_t_all_arr, y_pr_all_arr, n_classes,
                                  f"{dtype} {model_upper} Overall ROC",
                                  os.path.join(res_dir, "roc_overall.png"))

                result_file = os.path.join(res_dir, "overall_results.txt")
                with open(result_file, 'w') as f:
                    f.write(f"Overall Results for {model_upper} on {dtype}\n")
                    f.write("=" * 60 + "\n")
                    f.write(f"Accuracy: {oa:.4f}\n")
                    f.write(f"Macro Recall: {orc:.4f}\n")
                    f.write(f"Macro Precision: {opc:.4f}\n")
                    f.write(f"Macro F1 Score: {of1:.4f}\n")
                    f.write(f"Macro AUC (OvR): {oauc:.4f}\n")
                    f.write(f"Total Samples: {len(y_t_all_arr)}\n")
                    f.write(f"Total Subjects: {len(subject_ids)}\n")
                    f.write(f"Num Classes: 3 (task 1→0, 2→1, 3/4→2)\n")
            else:
                print(f"[{dtype}][{model_upper}] 没有可汇总的样本（可能所有受试者都被跳过）")

    print("All done.")

if __name__ == '__main__':
    main()
