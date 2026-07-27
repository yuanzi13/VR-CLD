import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


REQUIRED_COLUMNS: List[str] = [
    "leftEye_gaze_X", "leftEye_gaze_Y", "leftEye_gaze_Z",
    "leftEye_openness", "leftEye_pupil_position_X",
    "leftEye_pupil_position_Y", "leftEye_pupil_dilation",
    "rightEye_gaze_X", "rightEye_gaze_Y", "rightEye_gaze_Z",
    "rightEye_openness", "rightEye_pupil_position_X",
    "rightEye_pupil_position_Y", "rightEye_pupil_dilation",
    "combinedEye_gaze_X", "combinedEye_gaze_Y", "combinedEye_gaze_Z",
]

TASK_LABEL = {1: 0, 2: 0, 3: 1, 4: 1}
CLASS_NAMES = {0: "LCL", 1: "HCL"}
EXPECTED_WINDOWS = {"MCI": 2667, "HC": 4051, "ALL": 6718}


@dataclass
class FoldMetrics:
    dataset: str
    model: str
    fold: int
    n_train: int
    n_test: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_subject_list(dataset_type: str) -> List[Tuple[str, int]]:
    dataset_type = dataset_type.upper()
    mci = [("MCI", i) for i in range(1, 27)]
    hc = [("HC", i) for i in range(1, 43)]
    if dataset_type == "MCI":
        return mci
    if dataset_type == "HC":
        return hc
    if dataset_type == "ALL":
        return mci + hc
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def split_into_k_parts(n: int, k: int = 5) -> List[Tuple[int, int]]:
    """Split n ordered items into k contiguous parts; earlier parts get remainders."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if k <= 1:
        raise ValueError("k must be greater than 1")
    q, r = divmod(n, k)
    sizes = [q + (1 if i < r else 0) for i in range(k)]
    bounds: List[Tuple[int, int]] = []
    start = 0
    for size in sizes:
        end = start + size
        bounds.append((start, end))
        start = end
    return bounds


def windowize_from_array(
    arr_ch_t: np.ndarray,
    label: int,
    window_size: int = 240,
    overlap: float = 0.0,
) -> Tuple[List[np.ndarray], List[int]]:
    """Convert a (channels, time) array into ordered flattened windows."""
    if arr_ch_t.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {arr_ch_t.shape}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must satisfy 0 <= overlap < 1")

    channels, time_points = arr_ch_t.shape
    if channels != len(REQUIRED_COLUMNS):
        raise ValueError(
            f"Expected {len(REQUIRED_COLUMNS)} channels, got {channels}"
        )

    step = window_size - int(window_size * overlap)
    if step <= 0:
        raise ValueError("Window step must be positive")
    if time_points < window_size:
        return [], []

    n_segments = (time_points - window_size) // step + 1
    windows: List[np.ndarray] = []
    labels: List[int] = []

    for i in range(n_segments):
        start = i * step
        end = start + window_size
        segment = arr_ch_t[:, start:end].astype(np.float32, copy=False)
        if segment.shape != (channels, window_size):
            continue
        if not np.isfinite(segment).all():
            raise ValueError("Non-finite values remain after preprocessing")
        windows.append(segment.reshape(1, -1))
        labels.append(label)

    return windows, labels


def read_preprocessed_csv(csv_path: Path) -> np.ndarray:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing task file: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Empty CSV: {csv_path}")

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    numeric = df[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad_count = int(numeric.isna().sum().sum())
        raise ValueError(
            f"{csv_path} contains {bad_count} non-numeric/NaN values. "
            "Use the paper preprocessing script first instead of replacing them with zero."
        )

    # CSV convention: rows=time, columns=channels -> transpose to (channels, time)
    arr = numeric.to_numpy(dtype=np.float32).T
    if arr.shape[0] != len(REQUIRED_COLUMNS):
        raise ValueError(f"Unexpected channel dimension in {csv_path}: {arr.shape}")
    return arr


def build_subject_windows(
    data_root: Path,
    subjects: Sequence[Tuple[str, int]],
    window_size: int,
    overlap: float,
    n_folds: int = 5,
) -> Dict[str, Dict[str, object]]:
    """Window every task, then split each task's windows into 5 contiguous parts."""
    subject_data: Dict[str, Dict[str, object]] = {}

    for population, number in subjects:
        subject_id = f"{population}_{number:02d}"
        folds: List[List[Tuple[np.ndarray, int]]] = [
            [] for _ in range(n_folds)
        ]
        total_windows = 0
        task_counts: Dict[int, int] = {}

        for task in (1, 2, 3, 4):
            csv_path = data_root / population / str(task) / f"{number}.csv"
            arr = read_preprocessed_csv(csv_path)
            windows, labels = windowize_from_array(
                arr,
                TASK_LABEL[task],
                window_size=window_size,
                overlap=overlap,
            )
            if not windows:
                raise ValueError(
                    f"No complete windows produced for {csv_path}; "
                    f"time points={arr.shape[1]}, window_size={window_size}"
                )

            bounds = split_into_k_parts(len(windows), n_folds)
            for fold_index, (start, end) in enumerate(bounds):
                folds[fold_index].extend(
                    (windows[i], labels[i]) for i in range(start, end)
                )

            task_counts[task] = len(windows)
            total_windows += len(windows)

        subject_data[subject_id] = {
            "folds": folds,
            "total_windows": total_windows,
            "task_counts": task_counts,
        }

        fold_sizes = [len(fold) for fold in folds]
        print(
            f"  {subject_id}: total={total_windows}, "
            f"task_counts={task_counts}, fold_sizes={fold_sizes}"
        )

    return subject_data


def merge_subject_folds(
    subject_data: Dict[str, Dict[str, object]],
    n_folds: int = 5,
) -> List[Dict[str, List[object]]]:
    """Merge same-index subject parts into dataset-level subject-dependent folds."""
    merged = [{"X": [], "Y": [], "subject": []} for _ in range(n_folds)]

    for subject_id, payload in subject_data.items():
        subject_folds = payload["folds"]
        if not isinstance(subject_folds, list) or len(subject_folds) != n_folds:
            raise ValueError(f"Invalid folds for {subject_id}")

        for fold_index in range(n_folds):
            for x, y in subject_folds[fold_index]:
                merged[fold_index]["X"].append(x)
                merged[fold_index]["Y"].append(y)
                merged[fold_index]["subject"].append(subject_id)

    return merged


def build_estimator(model_key: str, args: argparse.Namespace) -> BaseEstimator:
    model_key = model_key.lower()

    if model_key == "svm":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel=args.svm_kernel,
                C=args.svm_c,
                gamma=args.svm_gamma,
                probability=True,
                class_weight=args.svm_class_weight,
                random_state=args.seed,
            )),
        ])

    if model_key == "knn":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(
                n_neighbors=args.knn_k,
                weights=args.knn_weights,
                metric=args.knn_metric,
            )),
        ])

    if model_key == "dt":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", DecisionTreeClassifier(
                criterion=args.dt_criterion,
                max_depth=args.dt_max_depth,
                min_samples_split=args.dt_min_samples_split,
                min_samples_leaf=args.dt_min_samples_leaf,
                class_weight=args.dt_class_weight,
                random_state=args.seed,
            )),
        ])

    raise ValueError(f"Unsupported model: {model_key}")


def fit_estimator(
    model_key: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
) -> BaseEstimator:
    estimator = build_estimator(model_key, args)

    if model_key != "knn" or not args.knn_grid_search:
        estimator.fit(x_train, y_train)
        return estimator

    min_class_count = int(np.bincount(y_train, minlength=2).min())
    n_splits = min(args.knn_inner_folds, min_class_count)
    if n_splits < 2:
        estimator.fit(x_train, y_train)
        return estimator

    max_k = min(args.knn_max_k, len(y_train))
    candidate_k = [k for k in range(1, max_k + 1) if k <= len(y_train)]
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=args.seed,
    )
    search = GridSearchCV(
        estimator=estimator,
        param_grid={"model__n_neighbors": candidate_k},
        scoring="accuracy",
        cv=cv,
        n_jobs=args.n_jobs,
        refit=True,
    )
    search.fit(x_train, y_train)
    print(
        f"    KNN best k={search.best_params_['model__n_neighbors']} "
        f"(inner CV={n_splits})"
    )
    return search.best_estimator_


def positive_class_score(model: BaseEstimator, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(x)
        if prob.ndim != 2 or prob.shape[1] < 2:
            raise ValueError("predict_proba did not return two-class probabilities")
        return np.asarray(prob[:, 1], dtype=float)

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(x), dtype=float)
        # Monotonic sigmoid mapping; AUC is invariant to monotonic transforms.
        return 1.0 / (1.0 + np.exp(-np.clip(scores, -50, 50)))

    raise TypeError("Estimator provides neither predict_proba nor decision_function")


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[float],
) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average="binary", pos_label=1, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, average="binary", pos_label=1, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, average="binary", pos_label=1, zero_division=0
        ),
    }
    try:
        metrics["auc"] = roc_auc_score(y_true, y_score)
    except ValueError:
        metrics["auc"] = float("nan")
    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    metrics: Dict[str, float],
    title: str,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    image = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1], ["Pred LCL", "Pred HCL"])
    ax.set_yticks([0, 1], ["True LCL", "True HCL"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    threshold = cm.max() / 2.0 if cm.size else 0.0
    for row in range(2):
        row_total = cm[row].sum()
        for col in range(2):
            value = int(cm[row, col])
            row_pct = value / row_total if row_total else 0.0
            ax.text(
                col,
                row,
                f"{value}\n({row_pct:.2%})",
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=13,
            )

    metric_text = (
        f"ACC={metrics['accuracy']:.4f}\n"
        f"PRE={metrics['precision']:.4f}\n"
        f"REC={metrics['recall']:.4f}\n"
        f"F1 ={metrics['f1']:.4f}\n"
        f"AUC={metrics['auc']:.4f}"
    )
    fig.text(
        0.80,
        0.50,
        metric_text,
        va="center",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    fig.tight_layout(rect=[0, 0, 0.77, 1])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_roc(
    y_true: Sequence[int],
    y_score: Sequence[float],
    title: str,
    save_path: Path,
) -> None:
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_value = roc_auc_score(y_true, y_score)
    except ValueError as exc:
        print(f"  ROC skipped for {title}: {exc}")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC={auc_value:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_fold_accuracies(
    accuracies: Sequence[float],
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    positions = np.arange(len(accuracies))
    ax.bar(positions, accuracies)
    ax.set_xticks(positions, [f"Fold {i + 1}" for i in positions])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy across 5 folds")
    for i, value in enumerate(accuracies):
        ax.text(i, min(value + 0.02, 0.98), f"{value:.3f}", ha="center")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Binary subject-dependent 5-fold classical ML experiments"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("DeepLearning") / "data_rml",
        help="Root of preprocessed CSVs: <root>/<MCI|HC>/<1..4>/<subject>.csv",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("Binary_5K_dependent_68_ML"),
    )
    parser.add_argument("--models", default="svm,knn,dt")
    parser.add_argument("--datasets", default="MCI,HC,ALL")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--window-size", type=int, default=240)
    parser.add_argument("--overlap", type=float, default=0.0)

    # SVM parameters. The manuscript does not fully disclose these values.
    parser.add_argument("--svm-kernel", default="rbf")
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-gamma", default="scale")
    parser.add_argument(
        "--svm-class-weight",
        default=None,
        choices=[None, "balanced"],
    )

    # KNN parameters. Use --knn-grid-search only if this matches the original run.
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--knn-weights", default="uniform", choices=["uniform", "distance"])
    parser.add_argument("--knn-metric", default="minkowski")
    parser.add_argument("--knn-grid-search", action="store_true")
    parser.add_argument("--knn-max-k", type=int, default=20)
    parser.add_argument("--knn-inner-folds", type=int, default=5)

    # Decision-tree parameters. Defaults are sklearn defaults unless stated.
    parser.add_argument("--dt-criterion", default="gini", choices=["gini", "entropy", "log_loss"])
    parser.add_argument("--dt-max-depth", type=int, default=None)
    parser.add_argument("--dt-min-samples-split", type=int, default=2)
    parser.add_argument("--dt-min-samples-leaf", type=int, default=1)
    parser.add_argument(
        "--dt-class-weight",
        default=None,
        choices=[None, "balanced"],
    )

    parser.add_argument(
        "--strict-window-count",
        action="store_true",
        help="Fail when MCI/HC/ALL total windows differ from 2667/4051/6718",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    model_keys = [item.strip().lower() for item in args.models.split(",") if item.strip()]
    dataset_keys = [item.strip().upper() for item in args.datasets.split(",") if item.strip()]

    unsupported_models = sorted(set(model_keys) - {"svm", "knn", "dt"})
    unsupported_datasets = sorted(set(dataset_keys) - {"MCI", "HC", "ALL"})
    if unsupported_models:
        raise ValueError(f"Unsupported models: {unsupported_models}")
    if unsupported_datasets:
        raise ValueError(f"Unsupported datasets: {unsupported_datasets}")
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {args.data_root}")

    args.result_dir.mkdir(parents=True, exist_ok=True)
    with (args.result_dir / "run_config.json").open("w", encoding="utf-8") as file:
        config = vars(args).copy()
        config["data_root"] = str(config["data_root"])
        config["result_dir"] = str(config["result_dir"])
        json.dump(config, file, ensure_ascii=False, indent=2)

    for dataset_key in dataset_keys:
        print(f"\n===== Dataset: {dataset_key} =====")
        subjects = get_subject_list(dataset_key)
        subject_data = build_subject_windows(
            data_root=args.data_root,
            subjects=subjects,
            window_size=args.window_size,
            overlap=args.overlap,
            n_folds=5,
        )
        folds = merge_subject_folds(subject_data, n_folds=5)

        total_windows = sum(len(fold["Y"]) for fold in folds)
        expected = EXPECTED_WINDOWS[dataset_key]
        print(f"[{dataset_key}] total windows={total_windows}, expected={expected}")
        if total_windows != expected:
            message = (
                f"Window count mismatch for {dataset_key}: "
                f"observed={total_windows}, expected={expected}. "
                "Check the upstream preprocessing and first/last 4-second trimming."
            )
            if args.strict_window_count:
                raise ValueError(message)
            print(f"WARNING: {message}")

        for fold_index, fold in enumerate(folds, start=1):
            labels, counts = np.unique(np.asarray(fold["Y"]), return_counts=True)
            print(
                f"[{dataset_key}] Fold {fold_index}: n={len(fold['Y'])}, "
                f"labels={dict(zip(labels.tolist(), counts.tolist()))}"
            )

        for model_key in model_keys:
            print(f"\n--- Model: {model_key.upper()} / {dataset_key} ---")
            model_dir = args.result_dir / model_key.upper() / dataset_key
            model_dir.mkdir(parents=True, exist_ok=True)

            all_true: List[int] = []
            all_pred: List[int] = []
            all_score: List[float] = []
            all_fold_ids: List[int] = []
            fold_metrics: List[FoldMetrics] = []
            fold_accuracies: List[float] = []
            total_cm = np.zeros((2, 2), dtype=int)

            for test_fold in range(5):
                x_test_list = folds[test_fold]["X"]
                y_test = np.asarray(folds[test_fold]["Y"], dtype=np.int64)

                x_train_list: List[np.ndarray] = []
                y_train_list: List[int] = []
                for train_fold in range(5):
                    if train_fold == test_fold:
                        continue
                    x_train_list.extend(folds[train_fold]["X"])
                    y_train_list.extend(folds[train_fold]["Y"])

                if not x_train_list or not x_test_list:
                    raise ValueError(
                        f"Empty train/test data in {dataset_key}, fold {test_fold + 1}"
                    )

                x_train = np.vstack(x_train_list).astype(np.float32, copy=False)
                x_test = np.vstack(x_test_list).astype(np.float32, copy=False)
                y_train = np.asarray(y_train_list, dtype=np.int64)

                print(
                    f"[{model_key.upper()}][{dataset_key}] Fold {test_fold + 1}/5: "
                    f"train={len(y_train)}, test={len(y_test)}"
                )

                model = fit_estimator(model_key, x_train, y_train, args)
                y_pred = np.asarray(model.predict(x_test), dtype=np.int64)
                y_score = positive_class_score(model, x_test)
                metrics = compute_metrics(y_test, y_pred, y_score)
                cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

                fold_dir = model_dir / f"fold_{test_fold + 1:02d}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                plot_confusion_matrix(
                    cm,
                    metrics,
                    f"{dataset_key} {model_key.upper()} Fold {test_fold + 1}",
                    fold_dir / "confusion.png",
                )
                plot_roc(
                    y_test,
                    y_score,
                    f"{dataset_key} {model_key.upper()} Fold {test_fold + 1} ROC",
                    fold_dir / "roc.png",
                )

                prediction_df = pd.DataFrame({
                    "fold": test_fold + 1,
                    "y_true": y_test,
                    "y_pred": y_pred,
                    "y_score_hcl": y_score,
                })
                prediction_df.to_csv(fold_dir / "predictions.csv", index=False)

                fold_metric = FoldMetrics(
                    dataset=dataset_key,
                    model=model_key.upper(),
                    fold=test_fold + 1,
                    n_train=len(y_train),
                    n_test=len(y_test),
                    accuracy=metrics["accuracy"],
                    precision=metrics["precision"],
                    recall=metrics["recall"],
                    f1=metrics["f1"],
                    auc=metrics["auc"],
                )
                fold_metrics.append(fold_metric)
                fold_accuracies.append(metrics["accuracy"])
                total_cm += cm

                all_true.extend(y_test.tolist())
                all_pred.extend(y_pred.tolist())
                all_score.extend(y_score.tolist())
                all_fold_ids.extend([test_fold + 1] * len(y_test))

            fold_metrics_df = pd.DataFrame([asdict(item) for item in fold_metrics])
            fold_metrics_df.to_csv(model_dir / "fold_metrics.csv", index=False)
            plot_fold_accuracies(
                fold_accuracies,
                model_dir / "accuracy_across_folds.png",
            )

            overall_metrics = compute_metrics(all_true, all_pred, all_score)
            plot_confusion_matrix(
                total_cm,
                overall_metrics,
                f"{dataset_key} {model_key.upper()} 5-Fold Overall",
                model_dir / "confusion_overall.png",
            )
            plot_roc(
                all_true,
                all_score,
                f"{dataset_key} {model_key.upper()} 5-Fold Overall ROC",
                model_dir / "roc_overall.png",
            )

            overall_predictions = pd.DataFrame({
                "fold": all_fold_ids,
                "y_true": all_true,
                "y_pred": all_pred,
                "y_score_hcl": all_score,
            })
            overall_predictions.to_csv(
                model_dir / "overall_predictions.csv",
                index=False,
            )

            overall_record = {
                "dataset": dataset_key,
                "model": model_key.upper(),
                "completed_folds": len(fold_metrics),
                "total_test_windows": len(all_true),
                **overall_metrics,
            }
            pd.DataFrame([overall_record]).to_csv(
                model_dir / "overall_metrics.csv",
                index=False,
            )
            with (model_dir / "overall_results.json").open("w", encoding="utf-8") as file:
                json.dump(overall_record, file, ensure_ascii=False, indent=2)

            print(
                f"Overall {model_key.upper()} / {dataset_key}: "
                f"ACC={overall_metrics['accuracy']:.4f}, "
                f"PRE={overall_metrics['precision']:.4f}, "
                f"REC={overall_metrics['recall']:.4f}, "
                f"F1={overall_metrics['f1']:.4f}, "
                f"AUC={overall_metrics['auc']:.4f}"
            )

    print("\nAll done.")


if __name__ == "__main__":
    main()
