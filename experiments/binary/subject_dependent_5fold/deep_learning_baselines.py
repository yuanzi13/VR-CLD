import argparse
import csv
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


REQUIRED_COLUMNS: Tuple[str, ...] = (
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

STAGE_LABELS: Tuple[Tuple[str, int], ...] = (
    ("1", 0),  # Test A -> LCL
    ("2", 0),  # Test B -> LCL
    ("3", 1),  # Test C -> HCL
    ("4", 1),  # Test D -> HCL
)

CLASS_NAMES: Tuple[str, str] = ("LCL", "HCL")


@dataclass(frozen=True)
class WindowMeta:
    population: str
    subject: int
    stage: int
    source_file: str
    window_index: int
    start_sample: int
    end_sample: int


@dataclass
class FoldData:
    x: List[np.ndarray]
    y: List[int]
    meta: List[WindowMeta]


def set_reproducibility(seed: int, deterministic: bool = True) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, TypeError):
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def select_device(gpu_id: Optional[int]) -> torch.device:
    if not torch.cuda.is_available():
        print("[INFO] CUDA is unavailable; using CPU.")
        return torch.device("cpu")

    visible = torch.cuda.device_count()
    idx = 0 if gpu_id is None else int(gpu_id)
    if idx < 0 or idx >= visible:
        raise ValueError(
            "Invalid --gpu-id {}. This process can see {} CUDA device(s).".format(
                idx, visible
            )
        )

    device = torch.device("cuda:{}".format(idx))
    print(
        "[INFO] Using {} ({})".format(
            device, torch.cuda.get_device_name(idx)
        )
    )
    return device


def resolve_data_root(value: str) -> Path:
    candidates = [
        Path(value),
        Path("DeepLearning") / value,
        Path("DeepLearning") / "data_rml",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Data root was not found. Checked: {}".format(
            ", ".join(str(p) for p in candidates)
        )
    )


def subjects_for_dataset(dataset_name: str) -> List[Tuple[str, int]]:
    dataset_name = dataset_name.upper()
    mci = [("MCI", i) for i in range(1, 27)]
    hc = [("HC", i) for i in range(1, 43)]

    if dataset_name == "MCI":
        return mci
    if dataset_name == "HC":
        return hc
    if dataset_name == "ALL":
        return mci + hc
    raise ValueError("Unsupported dataset: {}".format(dataset_name))


def read_preprocessed_csv(path: Path) -> np.ndarray:
    """
    Read one preprocessed task CSV and return an array shaped (channels, time).
    """
    frame = pd.read_csv(str(path))
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "{} is missing {} required channel(s): {}".format(
                path, len(missing), ", ".join(missing)
            )
        )

    numeric = frame.loc[:, REQUIRED_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    numeric = numeric.ffill().bfill()
    if numeric.isna().any().any():
        bad_columns = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(
            "{} still contains NaN values after filling: {}".format(
                path, ", ".join(bad_columns)
            )
        )

    values = numeric.to_numpy(dtype=np.float32, copy=True)
    if values.ndim != 2 or values.shape[1] != len(REQUIRED_COLUMNS):
        raise ValueError(
            "Unexpected array shape {} in {}".format(values.shape, path)
        )
    return values.T  # (C, T)


def segment_recording(
    signal: np.ndarray,
    label: int,
    population: str,
    subject: int,
    stage: int,
    source_file: Path,
    window_size: int,
    overlap: float,
) -> Tuple[List[np.ndarray], List[int], List[WindowMeta]]:
    if signal.ndim != 2 or signal.shape[0] != len(REQUIRED_COLUMNS):
        raise ValueError(
            "Expected signal shape ({}, T), got {}".format(
                len(REQUIRED_COLUMNS), signal.shape
            )
        )
    if window_size <= 0:
        raise ValueError("--window-size must be positive.")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("--overlap must satisfy 0 <= overlap < 1.")

    step = max(1, window_size - int(window_size * overlap))
    total_points = signal.shape[1]
    if total_points < window_size:
        return [], [], []

    windows: List[np.ndarray] = []
    labels: List[int] = []
    metadata: List[WindowMeta] = []
    count = (total_points - window_size) // step + 1

    for window_index in range(count):
        start = window_index * step
        end = start + window_size
        window = signal[:, start:end]
        if window.shape != (len(REQUIRED_COLUMNS), window_size):
            continue
        windows.append(window.astype(np.float32, copy=False))
        labels.append(int(label))
        metadata.append(
            WindowMeta(
                population=population,
                subject=int(subject),
                stage=int(stage),
                source_file=str(source_file),
                window_index=int(window_index),
                start_sample=int(start),
                end_sample=int(end),
            )
        )

    return windows, labels, metadata


def sequential_fold_slices(n_items: int, n_folds: int) -> List[slice]:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    base, remainder = divmod(n_items, n_folds)
    slices: List[slice] = []
    start = 0
    for fold_index in range(n_folds):
        size = base + (1 if fold_index < remainder else 0)
        end = start + size
        slices.append(slice(start, end))
        start = end
    return slices


def build_subject_dependent_folds(
    data_root: Path,
    subjects: Sequence[Tuple[str, int]],
    window_size: int,
    overlap: float,
    n_folds: int,
    allow_missing_files: bool,
) -> Tuple[List[FoldData], Dict[str, Any]]:
    """
    Split every subject-task recording sequentially into n_folds and aggregate
    corresponding parts across subjects.
    """
    folds = [FoldData(x=[], y=[], meta=[]) for _ in range(n_folds)]
    missing_files: List[str] = []
    short_files: List[str] = []
    file_window_counts: Dict[str, int] = {}
    subject_window_counts: Dict[str, int] = {}

    for population, subject in subjects:
        subject_key = "{}-{:02d}".format(population, subject)
        subject_window_counts.setdefault(subject_key, 0)

        for stage_text, label in STAGE_LABELS:
            path = data_root / population / stage_text / "{}.csv".format(subject)
            if not path.is_file():
                missing_files.append(str(path))
                continue

            signal = read_preprocessed_csv(path)
            windows, labels, metadata = segment_recording(
                signal=signal,
                label=label,
                population=population,
                subject=subject,
                stage=int(stage_text),
                source_file=path,
                window_size=window_size,
                overlap=overlap,
            )
            file_window_counts[str(path)] = len(windows)
            subject_window_counts[subject_key] += len(windows)

            if len(windows) < n_folds:
                short_files.append(
                    "{} ({} windows)".format(path, len(windows))
                )

            for fold_index, item_slice in enumerate(
                sequential_fold_slices(len(windows), n_folds)
            ):
                folds[fold_index].x.extend(windows[item_slice])
                folds[fold_index].y.extend(labels[item_slice])
                folds[fold_index].meta.extend(metadata[item_slice])

    if missing_files and not allow_missing_files:
        preview = "\n".join(missing_files[:20])
        suffix = (
            "\n... and {} more".format(len(missing_files) - 20)
            if len(missing_files) > 20
            else ""
        )
        raise FileNotFoundError(
            "Missing {} expected CSV file(s):\n{}{}".format(
                len(missing_files), preview, suffix
            )
        )

    if short_files:
        raise ValueError(
            "{} recording(s) contain fewer than {} complete windows and "
            "cannot contribute to every fold:\n{}".format(
                len(short_files), n_folds, "\n".join(short_files[:20])
            )
        )

    for fold_index, fold in enumerate(folds):
        if not fold.x:
            raise ValueError("Fold {} is empty.".format(fold_index + 1))
        present = set(fold.y)
        if present != {0, 1}:
            raise ValueError(
                "Fold {} does not contain both classes; labels present: {}".format(
                    fold_index + 1, sorted(present)
                )
            )

    audit = {
        "data_root": str(data_root),
        "n_subjects_requested": len(subjects),
        "n_expected_files": len(subjects) * len(STAGE_LABELS),
        "n_missing_files": len(missing_files),
        "missing_files": missing_files,
        "file_window_counts": file_window_counts,
        "subject_window_counts": subject_window_counts,
        "fold_window_counts": [len(fold.y) for fold in folds],
        "fold_label_counts": [
            {
                CLASS_NAMES[label]: int(np.sum(np.asarray(fold.y) == label))
                for label in (0, 1)
            }
            for fold in folds
        ],
        "total_windows": int(sum(len(fold.y) for fold in folds)),
    }
    return folds, audit


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (512, 256),
        dropout: float = 0.5,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = hidden
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(previous, num_classes)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x.reshape(x.size(0), -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract(x))


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=effective_dropout,
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return output[:, -1, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract(x))


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        sequence_length: int,
        model_dim: int = 32,
        nhead: int = 4,
        dim_feedforward: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if model_dim % nhead != 0:
            raise ValueError("model_dim must be divisible by nhead.")

        self.projection = nn.Linear(feature_dim, model_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, sequence_length, model_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.classifier = nn.Linear(model_dim, num_classes)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) != self.position_embedding.size(1):
            raise ValueError(
                "Sequence length changed from {} to {}.".format(
                    self.position_embedding.size(1), x.size(1)
                )
            )
        encoded = self.encoder(
            self.projection(x) + self.position_embedding
        )
        return encoded.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract(x))


class CNNClassifier(nn.Module):
    def __init__(
        self,
        input_channels: int,
        dropout: float = 0.0,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        modules: List[nn.Module] = [
            nn.Conv1d(input_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        ]
        if dropout > 0.0:
            modules.append(nn.Dropout(dropout))
        self.features = nn.Sequential(*modules)
        self.classifier = nn.Linear(32, num_classes)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        # Input is (B, T, C); Conv1d expects (B, C, T).
        return self.features(x.permute(0, 2, 1)).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract(x))


def build_model(
    model_name: str,
    feature_dim: int,
    sequence_length: int,
    args: argparse.Namespace,
) -> nn.Module:
    name = model_name.upper()
    if name == "MLP":
        return MLPClassifier(
            input_dim=feature_dim * sequence_length,
            hidden_dims=(args.mlp_hidden_1, args.mlp_hidden_2),
            dropout=args.mlp_dropout,
        )
    if name == "LSTM":
        return LSTMClassifier(
            feature_dim=feature_dim,
            hidden_dim=args.lstm_hidden,
            num_layers=args.lstm_layers,
            dropout=args.lstm_dropout,
        )
    if name == "TRANSFORMER":
        return TransformerClassifier(
            feature_dim=feature_dim,
            sequence_length=sequence_length,
            model_dim=args.transformer_dim,
            nhead=args.transformer_heads,
            dim_feedforward=args.transformer_ffn,
            num_layers=args.transformer_layers,
            dropout=args.transformer_dropout,
        )
    if name == "CNN":
        return CNNClassifier(
            input_channels=feature_dim,
            dropout=args.cnn_dropout,
        )
    raise ValueError("Unsupported model: {}".format(model_name))


def safe_auc(y_true: np.ndarray, hcl_probability: np.ndarray) -> float:
    try:
        if np.unique(y_true).size < 2:
            return float("nan")
        return float(roc_auc_score(y_true, hcl_probability))
    except ValueError:
        return float("nan")


def binary_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    hcl_probability: Sequence[float],
) -> Dict[str, float]:
    truth = np.asarray(y_true, dtype=np.int64)
    prediction = np.asarray(y_pred, dtype=np.int64)
    probability = np.asarray(hcl_probability, dtype=np.float64)

    return {
        "accuracy": float(
            accuracy_score(truth, prediction)
        ),
        
        # Paper-facing binary metrics:
        # HCL=1 is consistently treated as the positive class.
        "precision": float(
            precision_score(
                truth,
                prediction,
                pos_label=1,
                zero_division=0
            )
        ),
        
        "recall": float(
            recall_score(
                truth,
                prediction,
                pos_label=1,
                zero_division=0
            )
        ),
        
        "f1": float(
            f1_score(
                truth,
                prediction,
                pos_label=1,
                zero_division=0
            )
        ),
        
        "auc": safe_auc(
            truth,
            probability
        ),
        "precision_lcl": float(
            precision_score(
                truth, prediction, pos_label=0, zero_division=0
            )
        ),
        "recall_lcl": float(
            recall_score(
                truth, prediction, pos_label=0, zero_division=0
            )
        ),
        "f1_lcl": float(
            f1_score(truth, prediction, pos_label=0, zero_division=0)
        ),
        "precision_hcl": float(
            precision_score(
                truth, prediction, pos_label=1, zero_division=0
            )
        ),
        "recall_hcl": float(
            recall_score(
                truth, prediction, pos_label=1, zero_division=0
            )
        ),
        "f1_hcl": float(
            f1_score(truth, prediction, pos_label=1, zero_division=0)
        ),
        "precision_macro": float(
            precision_score(
                truth, prediction, average="macro", zero_division=0
            )
        ),
        "recall_macro": float(
            recall_score(
                truth, prediction, average="macro", zero_division=0
            )
        ),
        "f1_macro": float(
            f1_score(
                truth, prediction, average="macro", zero_division=0
            )
        ),
        "n_windows": int(truth.size),
    }


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        return value

    with path.open("w", encoding="utf-8") as handle:
        json.dump(convert(payload), handle, ensure_ascii=False, indent=2)


def save_history(path: Path, history: Dict[str, List[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    frame = pd.DataFrame(
        {
            "epoch": list(epochs),
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "train_accuracy": history["train_accuracy"],
            "val_accuracy": history["val_accuracy"],
        }
    )
    frame.to_csv(str(path), index=False)


def plot_training_curves(
    history: Dict[str, List[float]], path: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], label="Train")
    axes[1].plot(epochs, history["val_accuracy"], label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_summary(
    matrix: np.ndarray,
    metrics: Dict[str, float],
    path: Path,
    title: str,
) -> None:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (2, 2):
        raise ValueError("Expected a 2x2 confusion matrix.")

    total = matrix.sum()
    row_sum = matrix.sum(axis=1)
    column_sum = matrix.sum(axis=0)
    recall = np.divide(
        np.diag(matrix),
        row_sum,
        out=np.zeros(2, dtype=np.float64),
        where=row_sum > 0,
    )
    precision = np.divide(
        np.diag(matrix),
        column_sum,
        out=np.zeros(2, dtype=np.float64),
        where=column_sum > 0,
    )

    display = np.full((3, 3), np.nan, dtype=np.float64)
    display[:2, :2] = matrix
    display[:2, 2] = recall       # right column
    display[2, :2] = precision    # bottom row
    display[2, 2] = metrics["accuracy"]

    figure, axis = plt.subplots(figsize=(7.2, 5.8))
    image = axis.imshow(display, cmap="Blues", interpolation="nearest")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    axis.set_xticks([0, 1, 2])
    axis.set_xticklabels(["LCL", "HCL", "Recall"])
    axis.set_yticks([0, 1, 2])
    axis.set_yticklabels(["LCL", "HCL", "Precision"])
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(title)

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row in range(3):
        for column in range(3):
            value = display[row, column]
            if row < 2 and column < 2:
                fraction = value / total if total else 0.0
                text = "{:d}\n({:.2%})".format(int(value), fraction)
                color = "white" if value > threshold else "black"
            else:
                text = "{:.2%}".format(value)
                color = "black"
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                color=color,
                fontsize=12,
            )

    summary = (
        "ACC={accuracy:.4f}\n"
        "PRE(HCL)={precision:.4f}\n"
        "REC(HCL)={recall:.4f}\n"
        "F1(HCL)={f1:.4f}\n"
        "AUC(HCL)={auc:.4f}"
    ).format(**metrics)
    figure.text(
        0.79,
        0.50,
        summary,
        va="center",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    figure.tight_layout(rect=[0.0, 0.0, 0.76, 1.0])
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_roc(
    y_true: Sequence[int],
    hcl_probability: Sequence[float],
    path: Path,
    title: str,
) -> None:
    truth = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(hcl_probability, dtype=np.float64)
    figure, axis = plt.subplots(figsize=(5.8, 5.0))

    if np.unique(truth).size < 2:
        axis.text(
            0.5, 0.5, "ROC unavailable: only one class",
            ha="center", va="center"
        )
        axis.set_axis_off()
    else:
        false_positive_rate, true_positive_rate, _ = roc_curve(
            truth, score, pos_label=1
        )
        auc_value = safe_auc(truth, score)
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label="AUC={:.4f}".format(auc_value),
        )
        axis.plot([0, 1], [0, 1], linestyle="--")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("False-positive rate")
        axis.set_ylabel("True-positive rate")
        axis.set_title(title)
        axis.legend(loc="lower right")
        axis.grid(alpha=0.2)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    path: Path,
    title: str,
    seed: int,
    max_points: int,
) -> Dict[str, float]:
    features = np.nan_to_num(np.asarray(features), copy=False)
    labels = np.asarray(labels, dtype=np.int64)
    if features.ndim != 2 or features.shape[0] != labels.size:
        raise ValueError("Invalid t-SNE feature/label shapes.")
    if labels.size < 3 or np.unique(labels).size < 2:
        return {
            "silhouette_score": float("nan"),
            "davies_bouldin_index": float("nan"),
            "n_points": int(labels.size),
        }

    if max_points > 0 and labels.size > max_points:
        rng = np.random.RandomState(seed)
        chosen = rng.choice(labels.size, max_points, replace=False)
        features = features[chosen]
        labels = labels[chosen]

    perplexity = min(30.0, max(2.0, labels.size / 3.0))
    perplexity = min(perplexity, labels.size - 1.0)
    embedding = TSNE(
        n_components=2,
        init="pca",
        learning_rate=200.0,
        perplexity=perplexity,
        random_state=seed,
    ).fit_transform(features)

    try:
        silhouette = float(silhouette_score(embedding, labels))
    except ValueError:
        silhouette = float("nan")
    try:
        dbi = float(davies_bouldin_score(embedding, labels))
    except ValueError:
        dbi = float("nan")

    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    for label, marker in ((0, "o"), (1, "s")):
        mask = labels == label
        axis.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=18,
            alpha=0.65,
            marker=marker,
            label="{} (n={})".format(CLASS_NAMES[label], int(mask.sum())),
        )
    axis.set_title(title)
    axis.legend()
    axis.grid(alpha=0.2, linestyle="--")
    axis.text(
        1.02,
        0.04,
        "SC={:.3f}\nDBI={:.3f}".format(silhouette, dbi),
        transform=axis.transAxes,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    figure.tight_layout(rect=[0.0, 0.0, 0.86, 1.0])
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(figure)

    return {
        "silhouette_score": silhouette,
        "davies_bouldin_index": dbi,
        "n_points": int(labels.size),
    }


def make_train_validation_indices(
    labels: np.ndarray,
    validation_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < validation_ratio < 1.0):
        raise ValueError("--val-ratio must satisfy 0 < ratio < 1.")
    indices = np.arange(labels.size)
    if labels.size < 4:
        raise ValueError("At least four outer-training windows are required.")

    class_counts = np.bincount(labels, minlength=2)
    stratify = labels if np.all(class_counts >= 2) else None
    train_index, validation_index = train_test_split(
        indices,
        test_size=validation_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    if train_index.size == 0 or validation_index.size == 0:
        raise ValueError("Internal train/validation split is empty.")
    return np.asarray(train_index), np.asarray(validation_index)


def standardize_splits(
    outer_train_x: np.ndarray,
    test_x: np.ndarray,
    train_index: np.ndarray,
    validation_index: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Match the repository's baseline convention: reshape each (T,C) window to
    one feature vector, but fit the scaler on internal-training windows only.
    """
    n_outer, sequence_length, feature_dim = outer_train_x.shape
    train_flat = outer_train_x[train_index].reshape(train_index.size, -1)
    validation_flat = outer_train_x[validation_index].reshape(
        validation_index.size, -1
    )
    test_flat = test_x.reshape(test_x.shape[0], -1)

    scaler = StandardScaler()
    scaler.fit(train_flat)

    train_scaled = scaler.transform(train_flat).reshape(
        -1, sequence_length, feature_dim
    )
    validation_scaled = scaler.transform(validation_flat).reshape(
        -1, sequence_length, feature_dim
    )
    test_scaled = scaler.transform(test_flat).reshape(
        -1, sequence_length, feature_dim
    )
    return (
        train_scaled.astype(np.float32),
        validation_scaled.astype(np.float32),
        test_scaled.astype(np.float32),
        scaler,
    )


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(x.astype(np.float32, copy=False)),
        torch.from_numpy(y.astype(np.int64, copy=False)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    collect_features: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    hcl_probability: List[float] = []
    feature_batches: List[np.ndarray] = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs)
            probabilities = torch.softmax(logits, dim=1)

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            hcl_probability.extend(
                probabilities[:, 1].cpu().numpy().tolist()
            )
            if collect_features:
                feature_batches.append(
                    model.extract(inputs).cpu().numpy()
                )

    features = (
        np.vstack(feature_batches)
        if collect_features and feature_batches
        else None
    )
    return (
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred, dtype=np.int64),
        np.asarray(hcl_probability, dtype=np.float64),
        features,
    )


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    min_delta: float,
) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    model.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 1.0], device=device)
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_path = output_dir / "best.pth"
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item()) * labels.size(0)
            train_correct += int(
                (logits.argmax(dim=1) == labels).sum().item()
            )
            train_total += int(labels.size(0))

        model.eval()
        validation_loss_sum = 0.0
        validation_correct = 0
        validation_total = 0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(inputs)
                loss = criterion(logits, labels)
                validation_loss_sum += float(loss.item()) * labels.size(0)
                validation_correct += int(
                    (logits.argmax(dim=1) == labels).sum().item()
                )
                validation_total += int(labels.size(0))

        train_loss = train_loss_sum / max(train_total, 1)
        train_accuracy = train_correct / max(train_total, 1)
        validation_loss = validation_loss_sum / max(validation_total, 1)
        validation_accuracy = validation_correct / max(validation_total, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(validation_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(validation_accuracy)

        if validation_loss < best_validation_loss - min_delta:
            best_validation_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                },
                str(best_path),
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(
                    "[INFO] Early stopping at epoch {}; "
                    "best epoch={}, val_loss={:.6f}".format(
                        epoch, best_epoch, best_validation_loss
                    )
                )
                break

    if not best_path.is_file():
        raise RuntimeError("No best checkpoint was saved.")

    checkpoint = torch.load(str(best_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    training_info = {
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation_loss),
        "epochs_completed": len(history["train_loss"]),
        "checkpoint": str(best_path),
    }
    return history, training_info


def save_predictions(
    path: Path,
    metadata: Sequence[WindowMeta],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    hcl_probability: np.ndarray,
) -> None:
    if not (
        len(metadata)
        == y_true.size
        == y_pred.size
        == hcl_probability.size
    ):
        raise ValueError("Prediction arrays and metadata have different lengths.")

    rows: List[Dict[str, Any]] = []
    for meta, truth, prediction, hcl_prob in zip(
        metadata, y_true, y_pred, hcl_probability
    ):
        row = asdict(meta)
        row.update(
            {
                "y_true": int(truth),
                "true_class": CLASS_NAMES[int(truth)],
                "y_pred": int(prediction),
                "predicted_class": CLASS_NAMES[int(prediction)],
                "probability_lcl": float(1.0 - hcl_prob),
                "probability_hcl": float(hcl_prob),
            }
        )
        rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(str(path), index=False)


def run_model_dataset(
    model_name: str,
    dataset_name: str,
    folds: Sequence[FoldData],
    device: torch.device,
    output_root: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    dataset_output = output_root / model_name / dataset_name
    dataset_output.mkdir(parents=True, exist_ok=True)

    pooled_true: List[int] = []
    pooled_pred: List[int] = []
    pooled_hcl_probability: List[float] = []
    pooled_metadata: List[WindowMeta] = []
    fold_summaries: List[Dict[str, Any]] = []

    for fold_index in range(args.n_folds):
        fold_number = fold_index + 1
        fold_seed = args.seed + fold_index
        set_reproducibility(fold_seed, args.deterministic)

        test_fold = folds[fold_index]
        train_windows: List[np.ndarray] = []
        train_labels: List[int] = []
        for other_index, other_fold in enumerate(folds):
            if other_index == fold_index:
                continue
            train_windows.extend(other_fold.x)
            train_labels.extend(other_fold.y)

        if not train_windows or not test_fold.x:
            raise ValueError(
                "{} {} fold {} has an empty train or test set.".format(
                    model_name, dataset_name, fold_number
                )
            )

        # Stored windows are (C,T); all deep-learning baselines receive (T,C).
        outer_train_x = np.stack(train_windows).transpose(0, 2, 1)
        outer_train_y = np.asarray(train_labels, dtype=np.int64)
        test_x = np.stack(test_fold.x).transpose(0, 2, 1)
        test_y = np.asarray(test_fold.y, dtype=np.int64)

        train_index, validation_index = make_train_validation_indices(
            outer_train_y,
            validation_ratio=args.val_ratio,
            seed=fold_seed,
        )
        train_x, validation_x, test_x_scaled, scaler = standardize_splits(
            outer_train_x,
            test_x,
            train_index,
            validation_index,
        )
        train_y = outer_train_y[train_index]
        validation_y = outer_train_y[validation_index]

        train_loader = make_loader(
            train_x,
            train_y,
            batch_size=args.batch_size,
            shuffle=True,
            seed=fold_seed,
        )
        validation_loader = make_loader(
            validation_x,
            validation_y,
            batch_size=args.batch_size,
            shuffle=False,
            seed=fold_seed,
        )
        test_loader = make_loader(
            test_x_scaled,
            test_y,
            batch_size=args.batch_size,
            shuffle=False,
            seed=fold_seed,
        )

        sequence_length = train_x.shape[1]
        feature_dim = train_x.shape[2]
        model = build_model(
            model_name=model_name,
            feature_dim=feature_dim,
            sequence_length=sequence_length,
            args=args,
        )

        fold_output = dataset_output / "fold_{:02d}".format(fold_number)
        fold_output.mkdir(parents=True, exist_ok=True)
        print(
            "[{}][{}] Fold {}/{} | train={} val={} test={}".format(
                model_name,
                dataset_name,
                fold_number,
                args.n_folds,
                train_y.size,
                validation_y.size,
                test_y.size,
            )
        )

        history, training_info = train_one_fold(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            device=device,
            output_dir=fold_output,
            epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            min_delta=args.min_delta,
        )
        save_history(fold_output / "train_history.csv", history)
        plot_training_curves(history, fold_output / "training_curves.png")

        y_true, y_pred, hcl_probability, features = evaluate_model(
            model,
            test_loader,
            device,
            collect_features=args.tsne,
        )
        metrics = binary_metrics(y_true, y_pred, hcl_probability)
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

        save_predictions(
            fold_output / "predictions.csv",
            test_fold.meta,
            y_true,
            y_pred,
            hcl_probability,
        )
        plot_confusion_summary(
            matrix,
            metrics,
            fold_output / "confusion_matrix.png",
            "{} {} fold {}".format(
                dataset_name, model_name, fold_number
            ),
        )
        plot_roc(
            y_true,
            hcl_probability,
            fold_output / "roc.png",
            "{} {} fold {} ROC".format(
                dataset_name, model_name, fold_number
            ),
        )

        tsne_metrics: Optional[Dict[str, float]] = None
        if args.tsne and features is not None:
            tsne_metrics = plot_tsne(
                features=features,
                labels=y_true,
                path=fold_output / "tsne.png",
                title="{} {} fold {} t-SNE".format(
                    dataset_name, model_name, fold_number
                ),
                seed=fold_seed,
                max_points=args.tsne_max_points,
            )

        fold_summary = {
            "fold": fold_number,
            "seed": fold_seed,
            "n_outer_train": int(outer_train_y.size),
            "n_internal_train": int(train_y.size),
            "n_validation": int(validation_y.size),
            "n_test": int(test_y.size),
            "train_class_counts": {
                CLASS_NAMES[label]: int(np.sum(train_y == label))
                for label in (0, 1)
            },
            "validation_class_counts": {
                CLASS_NAMES[label]: int(np.sum(validation_y == label))
                for label in (0, 1)
            },
            "test_class_counts": {
                CLASS_NAMES[label]: int(np.sum(test_y == label))
                for label in (0, 1)
            },
            "metrics": metrics,
            "confusion_matrix": matrix.tolist(),
            "training": training_info,
            "tsne": tsne_metrics,
            "scaler_mean_shape": list(scaler.mean_.shape),
        }
        save_json(fold_output / "metrics.json", fold_summary)
        fold_summaries.append(fold_summary)

        pooled_true.extend(y_true.tolist())
        pooled_pred.extend(y_pred.tolist())
        pooled_hcl_probability.extend(hcl_probability.tolist())
        pooled_metadata.extend(test_fold.meta)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pooled_true_array = np.asarray(pooled_true, dtype=np.int64)
    pooled_pred_array = np.asarray(pooled_pred, dtype=np.int64)
    pooled_probability_array = np.asarray(
        pooled_hcl_probability, dtype=np.float64
    )
    pooled_metrics = binary_metrics(
        pooled_true_array,
        pooled_pred_array,
        pooled_probability_array,
    )
    pooled_matrix = confusion_matrix(
        pooled_true_array, pooled_pred_array, labels=[0, 1]
    )

    save_predictions(
        dataset_output / "predictions_all_folds.csv",
        pooled_metadata,
        pooled_true_array,
        pooled_pred_array,
        pooled_probability_array,
    )
    plot_confusion_summary(
        pooled_matrix,
        pooled_metrics,
        dataset_output / "confusion_matrix_overall.png",
        "{} {} pooled five-fold result".format(dataset_name, model_name),
    )
    plot_roc(
        pooled_true_array,
        pooled_probability_array,
        dataset_output / "roc_overall.png",
        "{} {} pooled five-fold ROC".format(dataset_name, model_name),
    )

    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "protocol": "subject-dependent sequential 5-fold",
        "primary_positive_class": "HCL (label 0)",
        "auc_positive_class": "HCL (label 1)",
        "metrics": pooled_metrics,
        "confusion_matrix": pooled_matrix.tolist(),
        "folds": fold_summaries,
    }
    save_json(dataset_output / "summary.json", summary)
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Binary subject-dependent five-fold training for MLP, LSTM, "
            "Transformer, and CNN baselines."
        )
    )
    parser.add_argument(
        "--model",
        default="ALL",
        choices=["MLP", "LSTM", "Transformer", "CNN", "ALL"],
        help="Train one baseline or all four baselines.",
    )
    parser.add_argument(
        "--datasets",
        default="MCI,HC,ALL",
        help="Comma-separated subset of MCI, HC, ALL.",
    )
    parser.add_argument(
        "--data-root",
        default=os.path.join("DeepLearning", "data_rml"),
        help="Root of the preprocessed CSV hierarchy.",
    )
    parser.add_argument(
        "--result-dir",
        default=os.path.join(
            "Binary_5K_dependent_68", "deep_learning_baselines"
        ),
    )
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--window-size", type=int, default=240)
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--mlp-hidden-1", type=int, default=512)
    parser.add_argument("--mlp-hidden-2", type=int, default=256)
    parser.add_argument("--mlp-dropout", type=float, default=0.5)

    parser.add_argument("--lstm-hidden", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--lstm-dropout", type=float, default=0.0)

    parser.add_argument("--transformer-dim", type=int, default=32)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-ffn", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)

    parser.add_argument("--cnn-dropout", type=float, default=0.0)

    # Python 3.8-compatible boolean flags.
    parser.add_argument(
        "--allow-missing-files",
        action="store_true",
        help=(
            "Permit absent expected subject/task CSVs. Intended only for "
            "smoke tests; full paper runs should leave this disabled."
        ),
    )
    parser.add_argument(
        "--non-deterministic",
        dest="deterministic",
        action="store_false",
        help="Allow nondeterministic CUDA operations.",
    )
    parser.set_defaults(deterministic=True)
    parser.add_argument(
        "--tsne",
        action="store_true",
        help="Generate a separate t-SNE plot for each fold.",
    )
    parser.add_argument("--tsne-max-points", type=int, default=8000)
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.n_folds != 5:
        raise ValueError(
            "This paper protocol is five-fold; --n-folds must remain 5."
        )
    if args.batch_size <= 0 or args.epochs <= 0:
        raise ValueError("--batch-size and --epochs must be positive.")
    if args.patience <= 0:
        raise ValueError("--patience must be positive.")

    data_root = resolve_data_root(args.data_root)
    result_root = Path(args.result_dir).resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    device = select_device(args.gpu_id)
    set_reproducibility(args.seed, args.deterministic)

    requested_datasets = [
        item.strip().upper()
        for item in args.datasets.split(",")
        if item.strip()
    ]
    if not requested_datasets:
        raise ValueError("No datasets were requested.")
    for dataset_name in requested_datasets:
        if dataset_name not in ("MCI", "HC", "ALL"):
            raise ValueError(
                "Unsupported dataset '{}'; choose MCI, HC, or ALL.".format(
                    dataset_name
                )
            )

    if args.model == "ALL":
        requested_models = ["MLP", "LSTM", "Transformer", "CNN"]
    else:
        requested_models = [args.model]

    run_config = vars(args).copy()
    run_config.update(
        {
            "resolved_data_root": str(data_root),
            "resolved_result_root": str(result_root),
            "device": str(device),
            "timestamp": datetime.now().isoformat(),
            "required_columns": list(REQUIRED_COLUMNS),
            "label_mapping": {
                "1": "LCL (0)",
                "2": "LCL (0)",
                "3": "HCL (1)",
                "4": "HCL (1)",
            },
            "primary_positive_class": "HCL (1)",
            "auc_positive_class": "HCL (1)",
            "python_version": sys.version,
            "torch_version": torch.__version__,
        }
    )
    save_json(result_root / "run_configuration.json", run_config)

    all_summaries: List[Dict[str, Any]] = []
    fold_cache: Dict[str, List[FoldData]] = {}

    for dataset_name in requested_datasets:
        subjects = subjects_for_dataset(dataset_name)
        folds, audit = build_subject_dependent_folds(
            data_root=data_root,
            subjects=subjects,
            window_size=args.window_size,
            overlap=args.overlap,
            n_folds=args.n_folds,
            allow_missing_files=args.allow_missing_files,
        )
        fold_cache[dataset_name] = folds
        save_json(result_root / "data_audit_{}.json".format(dataset_name), audit)
        print(
            "[DATA][{}] total windows={} | fold sizes={}".format(
                dataset_name,
                audit["total_windows"],
                audit["fold_window_counts"],
            )
        )

    for model_name in requested_models:
        for dataset_name in requested_datasets:
            summary = run_model_dataset(
                model_name=model_name,
                dataset_name=dataset_name,
                folds=fold_cache[dataset_name],
                device=device,
                output_root=result_root,
                args=args,
            )
            all_summaries.append(summary)

    summary_rows: List[Dict[str, Any]] = []
    for summary in all_summaries:
        row: Dict[str, Any] = {
            "model": summary["model"],
            "dataset": summary["dataset"],
        }
        row.update(summary["metrics"])
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(
        str(result_root / "all_results.csv"), index=False
    )
    save_json(result_root / "all_results.json", {"results": all_summaries})
    print("[DONE] Results saved to {}".format(result_root))


if __name__ == "__main__":
    main()
