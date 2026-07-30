from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
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

STAGE_LABEL: Tuple[Tuple[str, int], ...] = (
    ("1", 0),
    ("2", 0),
    ("3", 1),
    ("4", 1),
)
CLASS_NAMES = {0: "LCL", 1: "HCL"}
SubjectKey = Tuple[str, int]


@dataclass(frozen=True)
class SubjectWindows:
    x: np.ndarray  # (N, C, T)
    y: np.ndarray  # (N,)
    stage: np.ndarray  # (N,)
    source_file: np.ndarray  # (N,)


@dataclass(frozen=True)
class FoldResult:
    dataset: str
    subject_group: str
    subject_id: int
    n_train: int
    n_val: int
    n_test: int
    best_epoch: int
    accuracy: float
    precision_hcl: float
    recall_hcl: float
    f1_hcl: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    auc: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a binary TCN using LOSO cross-validation."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("DeepLearning") / "data_rml",
        help="Root containing MCI/<stage>/<id>.csv and HC/<stage>/<id>.csv.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("Binary_LOSO_68") / "FINAL_TCN" / "tcn_2",
        help="Output directory.",
    )
    parser.add_argument(
        "--datasets",
        default="MCI,HC,ALL",
        help="Comma-separated subset of MCI, HC, ALL.",
    )
    parser.add_argument("--window-size", type=int, default=240)
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--channels", default="64,64,128,128")
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dilation-base", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        help="Use deterministic PyTorch operations when available.",
    )
    parser.add_argument(
        "--no-deterministic",
        dest="deterministic",
        action="store_false",
        help="Disable deterministic PyTorch operations.",
    )

    # t-SNE defaults to False.
    parser.add_argument(
        "--tsne",
        dest="tsne",
        action="store_true",
        help="Create a t-SNE plot for each held-out subject.",
    )
    parser.add_argument(
        "--no-tsne",
        dest="tsne",
        action="store_false",
        help="Do not create t-SNE plots.",
    )
    parser.add_argument("--tsne-max-points", type=int, default=3000)

    # Strict column checking defaults to True.
    parser.add_argument(
        "--strict-columns",
        dest="strict_columns",
        action="store_true",
        help="Reject files missing any of the 17 required channels.",
    )
    parser.add_argument(
        "--no-strict-columns",
        dest="strict_columns",
        action="store_false",
        help="Allow files with missing required channels.",
    )

    parser.set_defaults(
        deterministic=True,
        tsne=False,
        strict_columns=True,
    )
    parser.add_argument(
        "--expected-mci",
        type=int,
        default=26,
        help="Expected MCI subject count; only used for a warning.",
    )
    parser.add_argument(
        "--expected-hc",
        type=int,
        default=42,
        help="Expected HC subject count; only used for a warning.",
    )
    args = parser.parse_args()

    if args.window_size <= 0:
        parser.error("--window-size must be positive")
    if not 0.0 <= args.overlap < 1.0:
        parser.error("--overlap must satisfy 0 <= overlap < 1")
    if not 0.0 < args.val_ratio < 1.0:
        parser.error("--val-ratio must satisfy 0 < val-ratio < 1")
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive")
    if args.kernel_size <= 0 or args.dilation_base <= 0:
        parser.error("--kernel-size and --dilation-base must be positive")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must satisfy 0 <= dropout < 1")
    return args


def set_reproducibility(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def select_device(gpu_id: Optional[int]) -> torch.device:
    if not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU.")
        return torch.device("cpu")
    visible = torch.cuda.device_count()
    index = 0 if gpu_id is None else int(gpu_id)
    if index < 0 or index >= visible:
        raise ValueError(f"Invalid --gpu-id={index}; visible GPU count is {visible}.")
    device = torch.device(f"cuda:{index}")
    print(f"Using {device}: {torch.cuda.get_device_name(index)}")
    return device


def resolve_data_root(path: Path) -> Path:
    candidates = [path, Path("DeepLearning") / "data_rml", Path("data_rml")]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Data root not found. Checked: " + ", ".join(str(p) for p in candidates)
    )


def subject_file(stage_dir: Path, subject_id: int) -> Optional[Path]:
    candidates = [stage_dir / f"{subject_id}.csv", stage_dir / f"{subject_id:02d}.csv"]
    for path in candidates:
        if path.is_file():
            return path
    for path in stage_dir.glob("*.csv"):
        try:
            if int(path.stem) == subject_id:
                return path
        except ValueError:
            continue
    return None


def discover_subjects(data_root: Path, group: str) -> List[SubjectKey]:
    ids = set()
    for stage, _ in STAGE_LABEL:
        stage_dir = data_root / group / stage
        if not stage_dir.is_dir():
            continue
        for path in stage_dir.glob("*.csv"):
            try:
                ids.add(int(path.stem))
            except ValueError:
                print(f"Warning: ignoring non-numeric subject file: {path}")
    return [(group, sid) for sid in sorted(ids)]


def read_signal_file(path: Path, strict_columns: bool) -> np.ndarray:
    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing and strict_columns:
        raise ValueError(f"{path} is missing required columns: {missing}")

    columns = list(REQUIRED_COLUMNS) if not missing else [
        column for column in REQUIRED_COLUMNS if column in df.columns
    ]
    if not columns:
        raise ValueError(f"{path} contains none of the required signal columns")

    numeric = df.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    array = numeric.to_numpy(dtype=np.float32).T  # (channels, time)

    if strict_columns and array.shape[0] != len(REQUIRED_COLUMNS):
        raise ValueError(
            f"{path} produced {array.shape[0]} channels; expected {len(REQUIRED_COLUMNS)}"
        )
    return array


def windowize(
    signal: np.ndarray,
    label: int,
    stage: str,
    source_file: Path,
    window_size: int,
    overlap: float,
) -> Tuple[List[np.ndarray], List[int], List[str], List[str]]:
    if signal.ndim != 2:
        raise ValueError(f"Expected a 2-D signal, got shape {signal.shape}")
    _, length = signal.shape
    step = max(1, window_size - int(round(window_size * overlap)))
    if length < window_size:
        return [], [], [], []

    x_list: List[np.ndarray] = []
    y_list: List[int] = []
    stage_list: List[str] = []
    file_list: List[str] = []
    for start in range(0, length - window_size + 1, step):
        segment = signal[:, start : start + window_size]
        if segment.shape[1] != window_size:
            continue
        x_list.append(np.ascontiguousarray(segment, dtype=np.float32))
        y_list.append(label)
        stage_list.append(stage)
        file_list.append(str(source_file))
    return x_list, y_list, stage_list, file_list


def build_subject_windows(
    data_root: Path,
    subjects: Sequence[SubjectKey],
    window_size: int,
    overlap: float,
    strict_columns: bool,
) -> Dict[SubjectKey, SubjectWindows]:
    output: Dict[SubjectKey, SubjectWindows] = {}
    for group, subject_id in subjects:
        x_all: List[np.ndarray] = []
        y_all: List[int] = []
        stages_all: List[str] = []
        files_all: List[str] = []

        for stage, label in STAGE_LABEL:
            path = subject_file(data_root / group / stage, subject_id)
            if path is None:
                continue
            try:
                signal = read_signal_file(path, strict_columns=strict_columns)
                x, y, stages, files = windowize(
                    signal,
                    label=label,
                    stage=stage,
                    source_file=path,
                    window_size=window_size,
                    overlap=overlap,
                )
            except Exception as exc:
                print(f"Warning: skipped {path}: {exc}")
                continue
            x_all.extend(x)
            y_all.extend(y)
            stages_all.extend(stages)
            files_all.extend(files)

        if x_all:
            x_array = np.stack(x_all).astype(np.float32, copy=False)
            y_array = np.asarray(y_all, dtype=np.int64)
            if x_array.shape[1] != len(REQUIRED_COLUMNS):
                print(
                    f"Warning: skipped {group}-{subject_id:02d}; "
                    f"found {x_array.shape[1]} channels, expected {len(REQUIRED_COLUMNS)}."
                )
                continue
            output[(group, subject_id)] = SubjectWindows(
                x=x_array,
                y=y_array,
                stage=np.asarray(stages_all),
                source_file=np.asarray(files_all),
            )
            labels, counts = np.unique(y_array, return_counts=True)
            distribution = {CLASS_NAMES[int(k)]: int(v) for k, v in zip(labels, counts)}
            print(
                f"Loaded {group}-{subject_id:02d}: {len(y_array)} windows, "
                f"labels={distribution}"
            )
    return output


def concatenate_subjects(
    subject_data: Mapping[SubjectKey, SubjectWindows],
    keys: Iterable[SubjectKey],
) -> Tuple[np.ndarray, np.ndarray]:
    keys = list(keys)
    if not keys:
        raise ValueError("No subjects supplied")
    x = np.concatenate([subject_data[key].x for key in keys], axis=0)
    y = np.concatenate([subject_data[key].y for key in keys], axis=0)
    return x, y


def stratified_inner_split(
    y: np.ndarray, val_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    if len(indices) < 2:
        raise ValueError("At least two outer-training windows are required")

    class_counts = np.bincount(y, minlength=2)
    stratify = y if np.all(class_counts >= 2) else None
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Inner training/validation split is empty")
    return np.sort(train_idx), np.sort(val_idx)


def fit_scaler(x_train: np.ndarray) -> StandardScaler:
    """Fit z-score parameters on flattened training windows only.

    This matches the normalization convention used by the repository's existing TCN experiment scripts: each channel-time position is treated
    as one feature during StandardScaler fitting.
    """
    flattened = x_train.reshape(len(x_train), -1)
    return StandardScaler().fit(flattened)


def apply_scaler(x: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    original_shape = x.shape
    scaled = scaler.transform(x.reshape(len(x), -1))
    return scaled.reshape(original_shape).astype(np.float32, copy=False)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.network = nn.Sequential(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Conv1d(n_inputs, n_outputs, kernel_size=1)
            if n_inputs != n_outputs
            else nn.Identity()
        )
        self.activation = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.network(x) + self.residual(x))


class TCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_size: int = 3,
        dilation_base: int = 2,
        dropout: float = 0.2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("channels must not be empty")
        blocks: List[nn.Module] = []
        previous = in_channels
        for level, output_channels in enumerate(channels):
            blocks.append(
                TemporalBlock(
                    n_inputs=previous,
                    n_outputs=output_channels,
                    kernel_size=kernel_size,
                    dilation=dilation_base**level,
                    dropout=dropout,
                )
            )
            previous = output_channels
        self.tcn = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(previous, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.tcn(x)
        return self.pool(features).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(x.astype(np.float32, copy=False)),
        torch.from_numpy(y.astype(np.int64, copy=False)),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            batch_size = y_batch.size(0)
            loss_sum += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total += batch_size

    return loss_sum / max(total, 1), correct / max(total, 1)


def predict(
    model: TCN,
    loader: DataLoader,
    device: torch.device,
    collect_features: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    model.eval()
    true_values: List[np.ndarray] = []
    predictions: List[np.ndarray] = []
    scores: List[np.ndarray] = []
    feature_batches: List[np.ndarray] = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            features = model.extract_features(x_batch)
            logits = model.classifier(features)
            probabilities = torch.softmax(logits, dim=1)
            true_values.append(y_batch.numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            scores.append(probabilities[:, 1].cpu().numpy())
            if collect_features:
                feature_batches.append(features.cpu().numpy())

    y_true = np.concatenate(true_values)
    y_pred = np.concatenate(predictions)
    y_score = np.concatenate(scores)
    features = np.concatenate(feature_batches) if feature_batches else None
    return y_true, y_pred, y_score, features


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    if np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray
) -> Dict[str, Optional[float]]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_hcl": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall_hcl": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_hcl": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc": safe_auc(y_true, y_score),
    }


def save_json(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)


def plot_history(history: pd.DataFrame, path: Path) -> None:
    fig, axis_loss = plt.subplots(figsize=(7.5, 4.5))
    axis_loss.plot(history["epoch"], history["train_loss"], label="Train loss")
    axis_loss.plot(history["epoch"], history["val_loss"], label="Validation loss")
    axis_loss.set_xlabel("Epoch")
    axis_loss.set_ylabel("Loss")
    axis_accuracy = axis_loss.twinx()
    axis_accuracy.plot(history["epoch"], history["train_accuracy"], label="Train accuracy")
    axis_accuracy.plot(history["epoch"], history["val_accuracy"], label="Validation accuracy")
    axis_accuracy.set_ylabel("Accuracy")
    lines = axis_loss.get_lines() + axis_accuracy.get_lines()
    axis_loss.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    path: Path,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    )
    fig, axis = plt.subplots(figsize=(5.8, 5.0))
    image = axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks([0, 1], ["LCL", "HCL"])
    axis.set_yticks([0, 1], ["LCL", "HCL"])
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(title)
    for row in range(2):
        for column in range(2):
            value = normalized[row, column]
            axis.text(
                column,
                row,
                f"{matrix[row, column]}\n{value:.1%}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_roc(y_true: np.ndarray, y_score: np.ndarray, title: str, path: Path) -> None:
    auc_value = safe_auc(y_true, y_score)
    if auc_value is None:
        return
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_score)
    fig, axis = plt.subplots(figsize=(5.8, 5.0))
    axis.plot(false_positive_rate, true_positive_rate, label=f"AUC = {auc_value:.4f}")
    axis.plot([0, 1], [0, 1], linestyle="--")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title(title)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    title: str,
    path: Path,
    seed: int,
    max_points: int,
) -> None:
    if len(features) < 3 or np.unique(labels).size < 2:
        return
    rng = np.random.default_rng(seed)
    selected: List[int] = []
    per_class = max(1, max_points // 2)
    for label in (0, 1):
        class_indices = np.flatnonzero(labels == label)
        if len(class_indices) > per_class:
            class_indices = rng.choice(class_indices, size=per_class, replace=False)
        selected.extend(class_indices.tolist())
    selected_array = np.asarray(selected, dtype=int)
    features = features[selected_array]
    labels = labels[selected_array]

    pca_components = min(50, features.shape[1], len(features) - 1)
    reduced = PCA(n_components=pca_components, random_state=seed).fit_transform(features)
    perplexity = min(40.0, max(2.0, (len(reduced) - 1) / 3.0))
    tsne_kwargs = {
        "n_components": 2,
        "init": "pca",
        "learning_rate": 200.0,
        "perplexity": perplexity,
        "metric": "cosine",
        "random_state": seed,
    }
    if "max_iter" in inspect.signature(TSNE).parameters:
        tsne_kwargs["max_iter"] = 1500
    else:
        tsne_kwargs["n_iter"] = 1500
    embedding = TSNE(**tsne_kwargs).fit_transform(reduced)

    fig, axis = plt.subplots(figsize=(6.4, 5.0))
    for label in (0, 1):
        mask = labels == label
        axis.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=12,
            alpha=0.65,
            label=f"{CLASS_NAMES[label]} (n={int(mask.sum())})",
        )
    axis.set_title(title)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def train_fold(
    model: TCN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    min_delta: float,
    fold_dir: Path,
) -> Tuple[int, pd.DataFrame]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_loss = math.inf
    best_epoch = 0
    bad_epochs = 0
    history_rows: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
            }
        )

        improved = val_loss < best_loss - min_delta
        if improved:
            best_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), fold_dir / "best.pth")
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 10 == 0 or improved or epoch == epochs:
            print(
                f"  epoch={epoch:03d} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} train_acc={train_accuracy:.4f} "
                f"val_acc={val_accuracy:.4f}"
            )
        if bad_epochs >= patience:
            print(f"  early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    if best_epoch == 0:
        raise RuntimeError("No checkpoint was saved")
    model.load_state_dict(torch.load(fold_dir / "best.pth", map_location=device))
    history = pd.DataFrame(history_rows)
    history.to_csv(fold_dir / "train_history.csv", index=False)
    plot_history(history, fold_dir / "train_history.png")
    return best_epoch, history


def run_dataset(
    dataset_name: str,
    subjects: Sequence[SubjectKey],
    subject_data: Mapping[SubjectKey, SubjectWindows],
    args: argparse.Namespace,
    channels: Sequence[int],
    device: torch.device,
) -> None:
    valid_subjects = [key for key in subjects if key in subject_data]
    if len(valid_subjects) < 2:
        print(f"Skipping {dataset_name}: fewer than two valid subjects.")
        return

    dataset_dir = args.result_dir / f"tcn_{dataset_name}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    fold_results: List[FoldResult] = []
    all_predictions: List[pd.DataFrame] = []

    for fold_index, test_subject in enumerate(valid_subjects, start=1):
        group, subject_id = test_subject
        fold_seed = args.seed + fold_index
        fold_dir = dataset_dir / f"subj_{group}_{subject_id:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[TCN-LOSO][{dataset_name}] fold {fold_index}/{len(valid_subjects)}: "
            f"held out {group}-{subject_id:02d}"
        )

        outer_train_subjects = [key for key in valid_subjects if key != test_subject]
        x_outer, y_outer = concatenate_subjects(subject_data, outer_train_subjects)
        test_data = subject_data[test_subject]
        x_test, y_test = test_data.x, test_data.y

        train_idx, val_idx = stratified_inner_split(
            y_outer, val_ratio=args.val_ratio, seed=fold_seed
        )
        x_train_raw, y_train = x_outer[train_idx], y_outer[train_idx]
        x_val_raw, y_val = x_outer[val_idx], y_outer[val_idx]

        scaler = fit_scaler(x_train_raw)
        x_train = apply_scaler(x_train_raw, scaler)
        x_val = apply_scaler(x_val_raw, scaler)
        x_test_scaled = apply_scaler(x_test, scaler)

        train_loader = make_loader(
            x_train,
            y_train,
            args.batch_size,
            shuffle=True,
            seed=fold_seed,
            num_workers=args.num_workers,
        )
        val_loader = make_loader(
            x_val,
            y_val,
            args.batch_size,
            shuffle=False,
            seed=fold_seed,
            num_workers=args.num_workers,
        )
        test_loader = make_loader(
            x_test_scaled,
            y_test,
            args.batch_size,
            shuffle=False,
            seed=fold_seed,
            num_workers=args.num_workers,
        )

        model = TCN(
            in_channels=len(REQUIRED_COLUMNS),
            channels=channels,
            kernel_size=args.kernel_size,
            dilation_base=args.dilation_base,
            dropout=args.dropout,
            num_classes=2,
        )
        best_epoch, _ = train_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            min_delta=args.min_delta,
            fold_dir=fold_dir,
        )

        y_true, y_pred, y_score, features = predict(
            model, test_loader, device, collect_features=args.tsne
        )
        metrics = calculate_metrics(y_true, y_pred, y_score)
        prediction_frame = pd.DataFrame(
            {
                "dataset": dataset_name,
                "subject_group": group,
                "subject_id": subject_id,
                "stage": test_data.stage,
                "source_file": test_data.source_file,
                "window_index": np.arange(len(y_true)),
                "true_label": y_true,
                "predicted_label": y_pred,
                "probability_hcl": y_score,
            }
        )
        prediction_frame.to_csv(fold_dir / "predictions.csv", index=False)
        all_predictions.append(prediction_frame)

        checkpoint = {
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "model": {
                "input_channels": len(REQUIRED_COLUMNS),
                "channels": list(channels),
                "kernel_size": args.kernel_size,
                "dilation_base": args.dilation_base,
                "dropout": args.dropout,
                "num_classes": 2,
            },
            "scaler": {
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
                "var": scaler.var_.tolist(),
                "n_features_in": int(scaler.n_features_in_),
            },
            "split": {
                "dataset": dataset_name,
                "test_subject": [group, subject_id],
                "outer_train_subjects": [list(key) for key in outer_train_subjects],
                "n_train_windows": int(len(y_train)),
                "n_val_windows": int(len(y_val)),
                "n_test_windows": int(len(y_test)),
                "fold_seed": fold_seed,
            },
            "best_epoch": best_epoch,
            "metrics": metrics,
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
        torch.save(checkpoint, fold_dir / "checkpoint.pth")
        save_json(
            {
                "dataset": dataset_name,
                "test_subject": [group, subject_id],
                "best_epoch": best_epoch,
                "n_train_windows": len(y_train),
                "n_val_windows": len(y_val),
                "n_test_windows": len(y_test),
                **metrics,
            },
            fold_dir / "metrics.json",
        )
        plot_confusion(
            y_true,
            y_pred,
            title=f"{dataset_name} LOSO: {group}-{subject_id:02d}",
            path=fold_dir / "confusion_matrix.png",
        )
        plot_roc(
            y_true,
            y_score,
            title=f"{dataset_name} LOSO: {group}-{subject_id:02d}",
            path=fold_dir / "roc.png",
        )
        if args.tsne and features is not None:
            plot_tsne(
                features,
                y_true,
                title=f"{dataset_name} LOSO: {group}-{subject_id:02d}",
                path=fold_dir / "tsne.png",
                seed=fold_seed,
                max_points=args.tsne_max_points,
            )

        fold_result = FoldResult(
            dataset=dataset_name,
            subject_group=group,
            subject_id=subject_id,
            n_train=len(y_train),
            n_val=len(y_val),
            n_test=len(y_test),
            best_epoch=best_epoch,
            accuracy=float(metrics["accuracy"]),
            precision_hcl=float(metrics["precision_hcl"]),
            recall_hcl=float(metrics["recall_hcl"]),
            f1_hcl=float(metrics["f1_hcl"]),
            precision_macro=float(metrics["precision_macro"]),
            recall_macro=float(metrics["recall_macro"]),
            f1_macro=float(metrics["f1_macro"]),
            auc=metrics["auc"],
        )
        fold_results.append(fold_result)
        print(
            f"  test accuracy={fold_result.accuracy:.4f}, "
            f"macro-F1={fold_result.f1_macro:.4f}, "
            f"AUC={fold_result.auc if fold_result.auc is not None else 'NA'}"
        )

    if not all_predictions:
        print(f"No completed folds for {dataset_name}.")
        return

    fold_frame = pd.DataFrame([asdict(result) for result in fold_results])
    fold_frame.to_csv(dataset_dir / "subject_metrics.csv", index=False)
    prediction_frame = pd.concat(all_predictions, ignore_index=True)
    prediction_frame.to_csv(dataset_dir / "predictions_all_folds.csv", index=False)

    y_true_all = prediction_frame["true_label"].to_numpy(dtype=np.int64)
    y_pred_all = prediction_frame["predicted_label"].to_numpy(dtype=np.int64)
    y_score_all = prediction_frame["probability_hcl"].to_numpy(dtype=float)
    overall_metrics = calculate_metrics(y_true_all, y_pred_all, y_score_all)
    subject_summary = {
        "accuracy_mean": float(fold_frame["accuracy"].mean()),
        "accuracy_sd": float(fold_frame["accuracy"].std(ddof=1))
        if len(fold_frame) > 1
        else 0.0,
        "f1_macro_mean": float(fold_frame["f1_macro"].mean()),
        "f1_macro_sd": float(fold_frame["f1_macro"].std(ddof=1))
        if len(fold_frame) > 1
        else 0.0,
    }
    save_json(
        {
            "dataset": dataset_name,
            "n_subjects": len(fold_results),
            "n_test_windows": len(prediction_frame),
            "pooled_window_metrics": overall_metrics,
            "mean_across_subject_folds": subject_summary,
        },
        dataset_dir / "overall_metrics.json",
    )
    plot_confusion(
        y_true_all,
        y_pred_all,
        title=f"{dataset_name} LOSO overall",
        path=dataset_dir / "confusion_matrix_overall.png",
    )
    plot_roc(
        y_true_all,
        y_score_all,
        title=f"{dataset_name} LOSO overall",
        path=dataset_dir / "roc_overall.png",
    )
    print(f"Completed {dataset_name}: {len(fold_results)} LOSO folds.")


def main() -> None:
    args = parse_args()
    set_reproducibility(args.seed, deterministic=args.deterministic)
    args.data_root = resolve_data_root(args.data_root)
    args.result_dir = args.result_dir.resolve()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.gpu_id)

    datasets = [item.strip().upper() for item in args.datasets.split(",") if item.strip()]
    unsupported = sorted(set(datasets) - {"MCI", "HC", "ALL"})
    if unsupported:
        raise ValueError(f"Unsupported datasets: {unsupported}")
    channels = [int(value.strip()) for value in args.channels.split(",") if value.strip()]
    if not channels or any(value <= 0 for value in channels):
        raise ValueError("--channels must be a comma-separated list of positive integers")

    mci_subjects = discover_subjects(args.data_root, "MCI")
    hc_subjects = discover_subjects(args.data_root, "HC")
    if len(mci_subjects) != args.expected_mci:
        print(
            f"Warning: discovered {len(mci_subjects)} MCI subjects; "
            f"expected {args.expected_mci}."
        )
    if len(hc_subjects) != args.expected_hc:
        print(
            f"Warning: discovered {len(hc_subjects)} HC subjects; "
            f"expected {args.expected_hc}."
        )

    all_subjects = mci_subjects + hc_subjects
    subject_data = build_subject_windows(
        data_root=args.data_root,
        subjects=all_subjects,
        window_size=args.window_size,
        overlap=args.overlap,
        strict_columns=args.strict_columns,
    )
    dataset_subjects = {
        "MCI": mci_subjects,
        "HC": hc_subjects,
        "ALL": all_subjects,
    }

    save_json(
        {
            "data_root": str(args.data_root),
            "result_dir": str(args.result_dir),
            "required_columns": list(REQUIRED_COLUMNS),
            "stage_label": {stage: label for stage, label in STAGE_LABEL},
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "discovered_subjects": {
                "MCI": [subject_id for _, subject_id in mci_subjects],
                "HC": [subject_id for _, subject_id in hc_subjects],
            },
        },
        args.result_dir / "run_configuration.json",
    )

    for dataset_name in datasets:
        run_dataset(
            dataset_name=dataset_name,
            subjects=dataset_subjects[dataset_name],
            subject_data=subject_data,
            args=args,
            channels=channels,
            device=device,
        )
    print("All requested LOSO experiments completed.")


if __name__ == "__main__":
    main()
