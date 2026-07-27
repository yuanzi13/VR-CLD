import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

CLASS_NAMES: Mapping[int, str] = {
    0: "LCL",
    1: "MCL",
    2: "HCL",
}
N_CLASSES = 3
LABELS = [0, 1, 2]

STAGE_LABELS: Sequence[Tuple[str, int]] = (
    ("1", 0),  # Test A -> LCL
    ("2", 1),  # Test B -> MCL
    ("3", 2),  # Test C -> HCL
    ("4", 2),  # Test D -> HCL
)

REQUIRED_COLUMNS: Sequence[str] = (
    "leftEye_gaze_X",
    "leftEye_gaze_Y",
    "leftEye_gaze_Z",
    "leftEye_openness",
    "leftEye_pupil_position_X",
    "leftEye_pupil_position_Y",
    "leftEye_pupil_dilation",
    "rightEye_gaze_X",
    "rightEye_gaze_Y",
    "rightEye_gaze_Z",
    "rightEye_openness",
    "rightEye_pupil_position_X",
    "rightEye_pupil_position_Y",
    "rightEye_pupil_dilation",
    "combinedEye_gaze_X",
    "combinedEye_gaze_Y",
    "combinedEye_gaze_Z",
)

EXPECTED_SUBJECT_COUNTS = {
    "MCI": 26,
    "HC": 42,
    "ALL": 68,
}

EXPECTED_WINDOW_COUNTS = {
    "MCI": 2667,
    "HC": 4051,
    "ALL": 6718,
}


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    population: str
    number: int
    X: np.ndarray  # shape: (N, 17*window_size)
    y: np.ndarray  # shape: (N,)
    stage_counts: Mapping[str, int]


@dataclass(frozen=True)
class FoldMetrics:
    subject_id: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    auc_macro_ovr: float
    n_test_windows: int


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# -----------------------------------------------------------------------------
# Data loading and windowing
# -----------------------------------------------------------------------------

def subject_list(dataset_type: str) -> List[Tuple[str, int]]:
    dataset_type = dataset_type.upper()
    mci = [("MCI", i) for i in range(1, 27)]
    hc = [("HC", i) for i in range(1, 43)]
    if dataset_type == "MCI":
        return mci
    if dataset_type == "HC":
        return hc
    if dataset_type == "ALL":
        return mci + hc
    raise ValueError(f"Unsupported dataset: {dataset_type}")


def read_preprocessed_csv(path: Path) -> np.ndarray:
    """Read one manuscript-preprocessed CSV and return C x T float32 data."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing required task file: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc

    if df.empty:
        raise ValueError(f"Empty CSV: {path}")

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required channels: {missing}")

    numeric = df.loc[:, REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    bad_columns = numeric.columns[numeric.isna().any()].tolist()
    if bad_columns:
        raise ValueError(
            f"{path} still contains missing/non-numeric values in {bad_columns}. "
            "Supply the manuscript-preprocessed CSV rather than raw data."
        )

    # CSV layout is expected to be time x channels; T yields channels x time.
    arr = numeric.to_numpy(dtype=np.float32).T
    if arr.shape[0] != len(REQUIRED_COLUMNS):
        raise ValueError(f"Unexpected channel count in {path}: {arr.shape[0]}")
    if arr.shape[1] == 0:
        raise ValueError(f"No time samples in {path}")
    return arr


def windowize(
    arr_ch_t: np.ndarray,
    label: int,
    window_size: int,
    overlap: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create chronological full windows and flatten each to 17*window_size."""
    if arr_ch_t.ndim != 2:
        raise ValueError(f"Expected C x T data, got shape {arr_ch_t.shape}")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must satisfy 0 <= overlap < 1")

    channels, time_points = arr_ch_t.shape
    if channels != len(REQUIRED_COLUMNS):
        raise ValueError(f"Expected {len(REQUIRED_COLUMNS)} channels, got {channels}")

    step = window_size - int(window_size * overlap)
    if step <= 0:
        raise ValueError("Window step must be positive")
    if time_points < window_size:
        return (
            np.empty((0, channels * window_size), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    n_windows = (time_points - window_size) // step + 1
    X = np.empty((n_windows, channels * window_size), dtype=np.float32)
    y = np.full((n_windows,), label, dtype=np.int64)

    for i in range(n_windows):
        start = i * step
        segment = arr_ch_t[:, start : start + window_size]
        X[i] = segment.reshape(-1)

    return X, y


def build_subject_records(
    data_root: Path,
    dataset_type: str,
    window_size: int,
    overlap: float,
) -> List[SubjectRecord]:
    records: List[SubjectRecord] = []

    for population, number in subject_list(dataset_type):
        stage_X: List[np.ndarray] = []
        stage_y: List[np.ndarray] = []
        counts: Dict[str, int] = {}

        for stage, label in STAGE_LABELS:
            csv_path = data_root / population / stage / f"{number}.csv"
            arr = read_preprocessed_csv(csv_path)
            X_part, y_part = windowize(arr, label, window_size, overlap)
            if len(y_part) == 0:
                raise ValueError(
                    f"{csv_path} produced zero full windows with window_size={window_size}"
                )
            stage_X.append(X_part)
            stage_y.append(y_part)
            counts[stage] = int(len(y_part))

        X_subject = np.vstack(stage_X)
        y_subject = np.concatenate(stage_y)
        subject_id = f"{population}_{number:02d}"

        # Every complete subject must contain all three ternary labels.
        present = set(np.unique(y_subject).tolist())
        if present != set(LABELS):
            raise ValueError(
                f"{subject_id} does not contain all three classes. Present labels: {sorted(present)}"
            )

        records.append(
            SubjectRecord(
                subject_id=subject_id,
                population=population,
                number=number,
                X=X_subject,
                y=y_subject,
                stage_counts=counts,
            )
        )

    expected_subjects = EXPECTED_SUBJECT_COUNTS[dataset_type]
    if len(records) != expected_subjects:
        raise ValueError(
            f"{dataset_type} has {len(records)} complete subjects; expected {expected_subjects}."
        )
    return records


def summarize_records(records: Sequence[SubjectRecord]) -> Tuple[int, Dict[str, int]]:
    total = sum(len(record.y) for record in records)
    stage_totals = {stage: 0 for stage, _ in STAGE_LABELS}
    for record in records:
        for stage, count in record.stage_counts.items():
            stage_totals[stage] += int(count)
    return total, stage_totals


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

def build_classifier(model_name: str, args: argparse.Namespace):
    model_name = model_name.lower()
    if model_name == "svm":
        class_weight = None if args.svm_class_weight == "none" else args.svm_class_weight
        return SVC(
            kernel=args.svm_kernel,
            C=args.svm_c,
            gamma=args.svm_gamma,
            probability=True,
            class_weight=class_weight,
            decision_function_shape="ovr",
            random_state=args.seed,
        )
    if model_name == "knn":
        return KNeighborsClassifier(
            n_neighbors=args.knn_k,
            weights=args.knn_weights,
            metric=args.knn_metric,
            n_jobs=args.n_jobs,
        )
    if model_name == "dt":
        class_weight = None if args.dt_class_weight == "none" else args.dt_class_weight
        return DecisionTreeClassifier(
            criterion=args.dt_criterion,
            max_depth=args.dt_max_depth,
            min_samples_split=args.dt_min_samples_split,
            min_samples_leaf=args.dt_min_samples_leaf,
            class_weight=class_weight,
            random_state=args.seed,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def aligned_predict_proba(model, X: np.ndarray) -> np.ndarray:
    """Return probabilities ordered as classes 0,1,2."""
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"{type(model).__name__} does not provide predict_proba")
    raw = np.asarray(model.predict_proba(X), dtype=float)
    classes = np.asarray(model.classes_, dtype=int)
    out = np.zeros((len(X), N_CLASSES), dtype=float)
    for source_col, class_id in enumerate(classes):
        if class_id not in LABELS:
            raise ValueError(f"Unexpected model class: {class_id}")
        out[:, class_id] = raw[:, source_col]
    row_sums = out.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Invalid predicted probabilities: at least one row sums to zero")
    return out / row_sums


# -----------------------------------------------------------------------------
# Metrics and plots
# -----------------------------------------------------------------------------

def macro_auc_ovr(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(
        roc_auc_score(
            y_true,
            y_prob,
            labels=LABELS,
            average="macro",
            multi_class="ovr",
        )
    )


def evaluate_predictions(
    subject_id: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> FoldMetrics:
    try:
        auc_value = macro_auc_ovr(y_true, y_prob)
    except ValueError as exc:
        print(f"Warning: {subject_id} macro-OvR AUC unavailable: {exc}", file=sys.stderr)
        auc_value = float("nan")

    return FoldMetrics(
        subject_id=subject_id,
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision_macro=float(
            precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
        ),
        recall_macro=float(
            recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
        ),
        f1_macro=float(
            f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
        ),
        auc_macro_ovr=auc_value,
        n_test_windows=int(len(y_true)),
    )


def row_normalize_confusion(cm: np.ndarray) -> np.ndarray:
    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)


def plot_subject_confusion(
    cm: np.ndarray,
    metrics: FoldMetrics,
    title: str,
    output_path: Path,
) -> None:
    normalized = row_normalize_confusion(cm)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    image = ax.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(N_CLASSES), [CLASS_NAMES[i] for i in LABELS])
    ax.set_yticks(range(N_CLASSES), [CLASS_NAMES[i] for i in LABELS])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            count = int(cm[i, j])
            pct = normalized[i, j]
            ax.text(
                j,
                i,
                f"{count}\n({pct:.1%})",
                ha="center",
                va="center",
                color="white" if pct > 0.5 else "black",
            )

    text = (
        f"Accuracy = {metrics.accuracy:.4f}\n"
        f"Precision (macro) = {metrics.precision_macro:.4f}\n"
        f"Recall (macro) = {metrics.recall_macro:.4f}\n"
        f"F1 (macro) = {metrics.f1_macro:.4f}\n"
        f"AUC (macro-OvR) = {metrics.auc_macro_ovr:.4f}"
    )
    ax.text(
        1.04,
        0.02,
        text,
        transform=ax.transAxes,
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_average_subject_confusion(
    mean_normalized_cm: np.ndarray,
    pooled_metrics: FoldMetrics,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    image = ax.imshow(mean_normalized_cm, vmin=0.0, vmax=1.0, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(N_CLASSES), [CLASS_NAMES[i] for i in LABELS])
    ax.set_yticks(range(N_CLASSES), [CLASS_NAMES[i] for i in LABELS])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            pct = mean_normalized_cm[i, j]
            ax.text(
                j,
                i,
                f"{pct:.1%}",
                ha="center",
                va="center",
                color="white" if pct > 0.5 else "black",
            )

    text = (
        "Matrix: subject-wise row-normalized mean\n"
        "Metrics: pooled window-level predictions\n\n"
        f"Accuracy = {pooled_metrics.accuracy:.4f}\n"
        f"Precision (macro) = {pooled_metrics.precision_macro:.4f}\n"
        f"Recall (macro) = {pooled_metrics.recall_macro:.4f}\n"
        f"F1 (macro) = {pooled_metrics.f1_macro:.4f}\n"
        f"AUC (macro-OvR) = {pooled_metrics.auc_macro_ovr:.4f}"
    )
    ax.text(
        1.04,
        0.02,
        text,
        transform=ax.transAxes,
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    fig.tight_layout(rect=[0.0, 0.0, 0.76, 1.0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pooled_confusion(
    pooled_cm: np.ndarray,
    pooled_metrics: FoldMetrics,
    title: str,
    output_path: Path,
) -> None:
    plot_subject_confusion(pooled_cm, pooled_metrics, title, output_path)


def plot_multiclass_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    y_binary = label_binarize(y_true, classes=LABELS)
    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    class_curves: List[Tuple[np.ndarray, np.ndarray]] = []
    for class_id in LABELS:
        target = y_binary[:, class_id]
        if target.min() == target.max():
            continue
        fpr, tpr, _ = roc_curve(target, y_prob[:, class_id])
        auc_value = roc_auc_score(target, y_prob[:, class_id])
        class_curves.append((fpr, tpr))
        ax.plot(fpr, tpr, label=f"{CLASS_NAMES[class_id]} (AUC={auc_value:.3f})")

    if class_curves:
        all_fpr = np.unique(np.concatenate([curve[0] for curve in class_curves]))
        mean_tpr = np.zeros_like(all_fpr)
        for fpr, tpr in class_curves:
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        mean_tpr /= len(class_curves)
        try:
            macro_auc = macro_auc_ovr(y_true, y_prob)
            ax.plot(all_fpr, mean_tpr, linestyle="--", linewidth=2,
                    label=f"Macro-OvR (AUC={macro_auc:.3f})")
        except ValueError:
            pass

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_by_subject(
    fold_metrics: Sequence[FoldMetrics],
    title: str,
    output_path: Path,
) -> None:
    labels = [item.subject_id for item in fold_metrics]
    values = [item.accuracy for item in fold_metrics]
    width = max(10.0, len(labels) * 0.25)
    fig, ax = plt.subplots(figsize=(width, 4.5))
    ax.bar(np.arange(len(values)), values)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_xlabel("Held-out subject")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# LOSO experiment
# -----------------------------------------------------------------------------

def run_loso_model(
    records: Sequence[SubjectRecord],
    dataset_type: str,
    model_name: str,
    args: argparse.Namespace,
) -> None:
    output_dir = Path(args.result_dir) / model_name.upper() / dataset_type
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics: List[FoldMetrics] = []
    normalized_subject_cms: List[np.ndarray] = []
    pooled_true: List[np.ndarray] = []
    pooled_pred: List[np.ndarray] = []
    pooled_prob: List[np.ndarray] = []
    prediction_frames: List[pd.DataFrame] = []

    for fold_index, test_record in enumerate(records, start=1):
        train_records = [record for record in records if record.subject_id != test_record.subject_id]
        X_train = np.vstack([record.X for record in train_records])
        y_train = np.concatenate([record.y for record in train_records])
        X_test = test_record.X
        y_test = test_record.y

        if set(np.unique(y_train).tolist()) != set(LABELS):
            raise ValueError(f"Training fold for {test_record.subject_id} lacks at least one class")
        if set(np.unique(y_test).tolist()) != set(LABELS):
            raise ValueError(f"Test subject {test_record.subject_id} lacks at least one class")

        # Manuscript requirement: fit StandardScaler only on outer training windows.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = build_classifier(model_name, args)
        model.fit(X_train_scaled, y_train)
        y_pred = np.asarray(model.predict(X_test_scaled), dtype=np.int64)
        y_prob = aligned_predict_proba(model, X_test_scaled)

        metrics = evaluate_predictions(test_record.subject_id, y_test, y_pred, y_prob)
        cm = confusion_matrix(y_test, y_pred, labels=LABELS)
        normalized_subject_cms.append(row_normalize_confusion(cm))
        fold_metrics.append(metrics)
        pooled_true.append(y_test)
        pooled_pred.append(y_pred)
        pooled_prob.append(y_prob)

        fold_dir = output_dir / f"subject_{test_record.subject_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        if not args.no_fold_plots:
            plot_subject_confusion(
                cm,
                metrics,
                f"{dataset_type} {model_name.upper()} LOSO: {test_record.subject_id}",
                fold_dir / "confusion.png",
            )
            plot_multiclass_roc(
                y_test,
                y_prob,
                f"{dataset_type} {model_name.upper()} LOSO: {test_record.subject_id} ROC",
                fold_dir / "roc.png",
            )

        fold_prediction = pd.DataFrame(
            {
                "dataset": dataset_type,
                "model": model_name.upper(),
                "held_out_subject": test_record.subject_id,
                "y_true": y_test,
                "y_pred": y_pred,
                "p_LCL": y_prob[:, 0],
                "p_MCL": y_prob[:, 1],
                "p_HCL": y_prob[:, 2],
            }
        )
        fold_prediction.to_csv(fold_dir / "predictions.csv", index=False)
        prediction_frames.append(fold_prediction)

        print(
            f"[{model_name.upper()}][{dataset_type}] "
            f"LOSO {fold_index}/{len(records)} {test_record.subject_id}: "
            f"Acc={metrics.accuracy:.4f}, Macro-F1={metrics.f1_macro:.4f}, "
            f"Macro-AUC={metrics.auc_macro_ovr:.4f}"
        )

    y_true_all = np.concatenate(pooled_true)
    y_pred_all = np.concatenate(pooled_pred)
    y_prob_all = np.vstack(pooled_prob)
    pooled_metrics = evaluate_predictions("OVERALL", y_true_all, y_pred_all, y_prob_all)
    pooled_cm = confusion_matrix(y_true_all, y_pred_all, labels=LABELS)
    mean_subject_cm = np.mean(np.stack(normalized_subject_cms), axis=0)

    plot_average_subject_confusion(
        mean_subject_cm,
        pooled_metrics,
        f"{dataset_type} {model_name.upper()} LOSO average confusion matrix",
        output_dir / "confusion_subject_average.png",
    )
    plot_pooled_confusion(
        pooled_cm,
        pooled_metrics,
        f"{dataset_type} {model_name.upper()} LOSO pooled confusion matrix",
        output_dir / "confusion_pooled.png",
    )
    plot_multiclass_roc(
        y_true_all,
        y_prob_all,
        f"{dataset_type} {model_name.upper()} LOSO overall ROC",
        output_dir / "roc_overall.png",
    )
    plot_accuracy_by_subject(
        fold_metrics,
        f"{dataset_type} {model_name.upper()} LOSO accuracy by subject",
        output_dir / "accuracy_across_subjects.png",
    )

    pd.DataFrame([asdict(item) for item in fold_metrics]).to_csv(
        output_dir / "fold_metrics.csv", index=False
    )
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output_dir / "predictions_all_subjects.csv", index=False
    )
    np.savetxt(output_dir / "confusion_subject_average.csv", mean_subject_cm, delimiter=",")
    np.savetxt(output_dir / "confusion_pooled_counts.csv", pooled_cm, delimiter=",", fmt="%d")

    summary = {
        "dataset": dataset_type,
        "model": model_name.upper(),
        "evaluation": "subject-independent LOSO",
        "n_subjects": len(records),
        "n_test_windows_pooled": int(len(y_true_all)),
        "labels": {str(k): v for k, v in CLASS_NAMES.items()},
        "overall_metrics_pooled_window_level": asdict(pooled_metrics),
        "confusion_matrix_for_manuscript_figure": (
            "row-normalize each held-out subject matrix, then average across subjects"
        ),
    }
    with open(output_dir / "overall_results.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    with open(output_dir / "overall_results.txt", "w", encoding="utf-8") as handle:
        handle.write(f"Dataset: {dataset_type}\n")
        handle.write(f"Model: {model_name.upper()}\n")
        handle.write("Evaluation: subject-independent LOSO\n")
        handle.write(f"Subjects: {len(records)}\n")
        handle.write(f"Pooled test windows: {len(y_true_all)}\n")
        handle.write(f"Accuracy: {pooled_metrics.accuracy:.6f}\n")
        handle.write(f"Macro Precision: {pooled_metrics.precision_macro:.6f}\n")
        handle.write(f"Macro Recall: {pooled_metrics.recall_macro:.6f}\n")
        handle.write(f"Macro F1: {pooled_metrics.f1_macro:.6f}\n")
        handle.write(f"Macro OvR AUC: {pooled_metrics.auc_macro_ovr:.6f}\n")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manuscript-aligned ternary LOSO traditional ML baselines"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("DeepLearning") / "data_rml",
        help="Root directory containing manuscript-preprocessed CSV files",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("Ternary_LOSO_ML_68"),
    )
    parser.add_argument(
        "--models",
        default="svm,knn,dt",
        help="Comma-separated manuscript ML baselines: svm,knn,dt",
    )
    parser.add_argument(
        "--datasets",
        default="MCI,HC,ALL",
        help="Comma-separated groups: MCI,HC,ALL",
    )
    parser.add_argument("--window-size", type=int, default=240)
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--strict-window-count",
        action="store_true",
        help="Require manuscript window totals: MCI=2667, HC=4051, ALL=6718",
    )
    parser.add_argument(
        "--no-fold-plots",
        action="store_true",
        help="Skip per-subject confusion and ROC plots to reduce runtime/storage",
    )

    # SVM parameters. The manuscript does not fully disclose them, so all are recorded/exposed.
    parser.add_argument("--svm-kernel", choices=["linear", "rbf", "poly", "sigmoid"], default="rbf")
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-gamma", default="scale")
    parser.add_argument("--svm-class-weight", choices=["none", "balanced"], default="none")

    # KNN parameters.
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--knn-weights", choices=["uniform", "distance"], default="uniform")
    parser.add_argument("--knn-metric", default="minkowski")

    # Decision-tree parameters.
    parser.add_argument("--dt-criterion", choices=["gini", "entropy", "log_loss"], default="gini")
    parser.add_argument("--dt-max-depth", type=int, default=None)
    parser.add_argument("--dt-min-samples-split", type=int, default=2)
    parser.add_argument("--dt-min-samples-leaf", type=int, default=1)
    parser.add_argument("--dt-class-weight", choices=["none", "balanced"], default="none")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_root = args.data_root
    if not data_root.is_dir():
        fallback = Path("DeepLearning") / "data_rml"
        if fallback.is_dir():
            data_root = fallback
        else:
            raise FileNotFoundError(f"Data root not found: {args.data_root}")
    args.data_root = data_root
    args.result_dir.mkdir(parents=True, exist_ok=True)

    requested_models = [item.strip().lower() for item in args.models.split(",") if item.strip()]
    valid_models = {"svm", "knn", "dt"}
    if not requested_models or any(item not in valid_models for item in requested_models):
        raise ValueError(f"Models must be selected from {sorted(valid_models)}")

    requested_datasets = [item.strip().upper() for item in args.datasets.split(",") if item.strip()]
    valid_datasets = {"MCI", "HC", "ALL"}
    if not requested_datasets or any(item not in valid_datasets for item in requested_datasets):
        raise ValueError(f"Datasets must be selected from {sorted(valid_datasets)}")

    run_config = {
        "script": "ternary_loso_ml_article.py",
        "method": "three-class subject-independent LOSO traditional ML",
        "data_requirement": "manuscript-preprocessed CSV files",
        "class_mapping": {"1/A": "LCL", "2/B": "MCL", "3/C and 4/D": "HCL"},
        "required_columns": list(REQUIRED_COLUMNS),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "note": (
            "Traditional-ML hyperparameters are not fully disclosed in the manuscript; "
            "this file records and exposes all selected values."
        ),
    }
    with open(args.result_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, ensure_ascii=False, indent=2)

    for dataset_type in requested_datasets:
        print(f"\nLoading {dataset_type} ...")
        records = build_subject_records(
            data_root=args.data_root,
            dataset_type=dataset_type,
            window_size=args.window_size,
            overlap=args.overlap,
        )
        total_windows, stage_totals = summarize_records(records)
        expected_windows = EXPECTED_WINDOW_COUNTS[dataset_type]
        print(
            f"{dataset_type}: subjects={len(records)}, windows={total_windows}, "
            f"stage counts={stage_totals}, expected windows={expected_windows}"
        )
        if args.strict_window_count and total_windows != expected_windows:
            raise ValueError(
                f"{dataset_type} window count mismatch: observed={total_windows}, "
                f"expected={expected_windows}. Check upstream manuscript preprocessing."
            )

        for model_name in requested_models:
            print(f"\nRunning {model_name.upper()} on {dataset_type} ...")
            run_loso_model(records, dataset_type, model_name, args)

    print("\nAll requested ternary LOSO ML experiments completed.")


if __name__ == "__main__":
    main()
