import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


ET_COLUMNS: List[str] = [
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
]


def natural_sort_key(value: str) -> List[object]:
    """Sort strings naturally, e.g. 2 before 10."""
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", value)
    ]


def normalize_identifier(value: object) -> str:
    """Normalize directory and metadata identifiers for matching."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"-?\d+\.0", text):
        text = text[:-2]
    return text


def validate_columns(df: pd.DataFrame, required: Sequence[str], source: Path) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "Missing required ET columns in {}: {}".format(
                source, ", ".join(missing)
            )
        )


def preprocess_signal_file(
    sensor_csv: Path,
    sample_rate: int = 120,
    pupil_threshold_mm: float = 2.0,
    trim_seconds: float = 4.0,
    window_seconds: float = 2.0,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Preprocess one task-level sensor CSV."""
    df = pd.read_csv(sensor_csv)
    validate_columns(df, ET_COLUMNS, sensor_csv)

    # The reported model input consists only of the 17 ET channels.
    et = df.loc[:, ET_COLUMNS].apply(pd.to_numeric, errors="coerce")
    original_rows = len(et)

    # Remove records with missing left-eye openness.
    missing_left_openness = et["leftEye_openness"].isna()
    removed_missing_left_openness = int(missing_left_openness.sum())
    et = et.loc[~missing_left_openness].copy()

    if et.empty:
        raise ValueError(
            "No rows remain after removing missing left-eye openness: {}".format(
                sensor_csv
            )
        )

    # Preserve eye-closure indicators before blink reconstruction.
    left_openness = et["leftEye_openness"].copy()
    right_openness = et["rightEye_openness"].copy()

    # Either pupil below the threshold identifies a blink-related invalid time.
    blink_mask = (
        (et["leftEye_pupil_dilation"] < pupil_threshold_mm)
        | (et["rightEye_pupil_dilation"] < pupil_threshold_mm)
    )
    blink_timestamps = int(blink_mask.sum())

    # Temporarily mark all 17 ET channels at invalid timestamps as missing.
    et.loc[blink_mask, ET_COLUMNS] = np.nan

    # Forward filling only: no bfill() and no fillna(0) are applied here.
    et.loc[:, ET_COLUMNS] = et.loc[:, ET_COLUMNS].ffill()

    # Restore the original openness channels.
    et.loc[:, "leftEye_openness"] = left_openness
    et.loc[:, "rightEye_openness"] = right_openness

    trim_samples = int(round(trim_seconds * sample_rate))
    window_samples = int(round(window_seconds * sample_rate))

    if trim_samples < 0 or window_samples <= 0:
        raise ValueError(
            "trim_seconds must be non-negative and window_seconds must be positive."
        )

    if len(et) <= 2 * trim_samples:
        raise ValueError(
            "Recording is too short to remove {} samples from each end: {}".format(
                trim_samples, sensor_csv
            )
        )

    # Remove the first and last 4 s.
    if trim_samples:
        et = et.iloc[trim_samples:-trim_samples].copy()

    # Keep complete, non-overlapping 2 s windows only.
    complete_rows = (len(et) // window_samples) * window_samples
    discarded_terminal_rows = len(et) - complete_rows
    et = et.iloc[:complete_rows].reset_index(drop=True)

    if et.empty:
        raise ValueError(
            "No complete {}-sample window remains: {}".format(
                window_samples, sensor_csv
            )
        )

    # Do not silently introduce undocumented bfill or zero replacement.
    residual_missing_values = int(et.loc[:, ET_COLUMNS].isna().sum().sum())
    if residual_missing_values:
        raise ValueError(
            "{} residual missing ET values remain after forward filling in {}. "
            "Inspect the leading invalid interval rather than silently applying "
            "backward filling or zero replacement.".format(
                residual_missing_values, sensor_csv
            )
        )

    stats = {
        "original_rows": original_rows,
        "removed_missing_left_openness": removed_missing_left_openness,
        "blink_timestamps": blink_timestamps,
        "retained_rows": len(et),
        "complete_windows": len(et) // window_samples,
        "discarded_terminal_rows": discarded_terminal_rows,
        "residual_missing_values": residual_missing_values,
    }
    return et.loc[:, ET_COLUMNS], stats


def load_metadata(
    metadata_csv: Path,
    subject_id_column: str,
    label_column: str,
) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_csv)
    required = [subject_id_column, label_column]
    missing = [column for column in required if column not in metadata.columns]
    if missing:
        raise ValueError(
            "Missing metadata columns in {}: {}".format(
                metadata_csv, ", ".join(missing)
            )
        )

    metadata = metadata.copy()
    metadata["_subject_key"] = metadata[subject_id_column].map(normalize_identifier)

    duplicated = metadata["_subject_key"].duplicated(keep=False)
    if duplicated.any():
        duplicate_ids = sorted(metadata.loc[duplicated, "_subject_key"].unique())
        raise ValueError(
            "Duplicate subject IDs in metadata: {}".format(", ".join(duplicate_ids))
        )

    return metadata.set_index("_subject_key", drop=False)


def discover_subject_directories(input_root: Path) -> List[Path]:
    """Discover subjects without a hard-coded participant list."""
    subjects = [path for path in input_root.iterdir() if path.is_dir()]
    subjects.sort(key=lambda path: natural_sort_key(path.name))
    if not subjects:
        raise ValueError("No subject directories found under {}".format(input_root))
    return subjects


def group_name_from_label(raw_label: object, hc_label: int, mci_label: int) -> str:
    try:
        label = int(raw_label)
    except (TypeError, ValueError):
        raise ValueError("Participant label is not an integer: {!r}".format(raw_label))

    if label == hc_label:
        return "HC"
    if label == mci_label:
        return "MCI"
    raise ValueError(
        "Unsupported label {}. Expected HC label {} or MCI label {}.".format(
            label, hc_label, mci_label
        )
    )


def preprocess_dataset(args: argparse.Namespace) -> pd.DataFrame:
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    metadata_csv = args.metadata.resolve()

    if not input_root.is_dir():
        raise FileNotFoundError("Input root does not exist: {}".format(input_root))
    if not metadata_csv.is_file():
        raise FileNotFoundError("Metadata file does not exist: {}".format(metadata_csv))

    output_root.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(
        metadata_csv,
        subject_id_column=args.subject_id_column,
        label_column=args.label_column,
    )
    subject_dirs = discover_subject_directories(input_root)

    group_counters: Dict[str, int] = {"HC": 0, "MCI": 0}
    manifest_rows: List[Dict[str, object]] = []

    for subject_dir in subject_dirs:
        subject_key = normalize_identifier(subject_dir.name)
        if subject_key not in metadata.index:
            logging.warning(
                "Skipping directory without metadata match: %s", subject_dir.name
            )
            continue

        row = metadata.loc[subject_key]
        group = group_name_from_label(
            row[args.label_column],
            hc_label=args.hc_label,
            mci_label=args.mci_label,
        )

        # Automatically assign an anonymized group-specific index.
        group_counters[group] += 1
        subject_index = group_counters[group]
        anonymized_subject = "{}_{:03d}".format(group, subject_index)

        for task in args.tasks:
            sensor_csv = subject_dir / args.sensor_pattern.format(task=task)
            if not sensor_csv.is_file():
                logging.warning("Missing task file: %s", sensor_csv)
                continue

            try:
                processed, stats = preprocess_signal_file(
                    sensor_csv=sensor_csv,
                    sample_rate=args.sample_rate,
                    pupil_threshold_mm=args.pupil_threshold_mm,
                    trim_seconds=args.trim_seconds,
                    window_seconds=args.window_seconds,
                )
            except Exception as exc:
                if args.skip_errors:
                    logging.error("Skipping %s because: %s", sensor_csv, exc)
                    continue
                raise

            # Compatible layout: output_root/HC/1/1.csv, etc.
            task_dir = output_root / group / str(task)
            task_dir.mkdir(parents=True, exist_ok=True)
            output_csv = task_dir / "{}.csv".format(subject_index)

            if output_csv.exists() and not args.overwrite:
                raise FileExistsError(
                    "Output exists; use --overwrite to replace it: {}".format(
                        output_csv
                    )
                )

            processed.to_csv(output_csv, index=False)
            manifest_rows.append(
                {
                    "subject": anonymized_subject,
                    "group": group,
                    "task": task,
                    "output_file": str(output_csv.relative_to(output_root)),
                    **stats,
                }
            )
            logging.info(
                "Saved %s task %s: %s complete windows -> %s",
                anonymized_subject,
                task,
                stats["complete_windows"],
                output_csv,
            )

    manifest = pd.DataFrame(manifest_rows)
    if manifest.empty:
        raise RuntimeError("No files were processed.")

    manifest_path = args.manifest.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    logging.info("Saved preprocessing manifest: %s", manifest_path)
    logging.info(
        "Processed %d task files and retained %d complete windows.",
        len(manifest),
        int(manifest["complete_windows"].sum()),
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess the 17-channel VR-CLD eye-tracking recordings."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Directory containing one subdirectory per participant.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="CSV containing participant IDs and HC/MCI labels.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination for preprocessed task CSV files.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("preprocessing_manifest.csv"),
        help="Destination for the anonymized preprocessing summary CSV.",
    )
    parser.add_argument(
        "--subject-id-column",
        default="id",
        help="Participant-ID column in the metadata CSV.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="HC/MCI label column in the metadata CSV.",
    )
    parser.add_argument("--hc-label", type=int, default=0)
    parser.add_argument("--mci-label", type=int, default=1)
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Task identifiers to process.",
    )
    parser.add_argument(
        "--sensor-pattern",
        default="sensordata{task}.csv",
        help="Task filename pattern; use {task} as the placeholder.",
    )
    parser.add_argument("--sample-rate", type=int, default=120)
    parser.add_argument("--pupil-threshold-mm", type=float, default=2.0)
    parser.add_argument("--trim-seconds", type=float, default=4.0)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )
    preprocess_dataset(args)


if __name__ == "__main__":
    main()
