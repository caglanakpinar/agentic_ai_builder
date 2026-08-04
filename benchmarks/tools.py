"""The tool functions the benchmark's agents call, one per `caller` in `agentic_configurations.yaml`.

Every tool here does the real thing on the real dataset — reads the CSV, imputes what is missing, fits
the model, scores the holdout — rather than describing what it would do. That is the point of the
benchmark: what the agents produce can be checked against what the tools actually computed.

Nothing outside the standard library is imported, so the whole workflow runs on a bare `poetry install`
with no ML stack present. The models are small on purpose (logistic regression by gradient descent,
k-means, a one-hidden-layer MLP, PCA by power iteration) — enough to learn the signal
`dataset.py` puts in the data, and small enough to read.

Two conventions hold throughout:

  - Every tool returns a JSON-serialisable dict, because its return value goes back to a model as the
    result of a tool call.
  - Arguments are coerced rather than trusted: a model that sends `top_k="10"` or `drop_duplicates="true"`
    gets the same behaviour as one that sends the right type.

Artifacts (cleaned CSVs, fitted models, manifests) are written under `benchmarks/artifacts/`, so a later
step in the pipeline can pick up what an earlier one produced.
"""

import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

BENCHMARK_DIR = Path(__file__).parent
REPO_ROOT = BENCHMARK_DIR.parent
DATA_DIR = BENCHMARK_DIR / "data"
ARTIFACTS_DIR = BENCHMARK_DIR / "artifacts"

TRAIN_DATA = DATA_DIR / "churn_train.csv"
HOLDOUT_DATA = DATA_DIR / "churn_holdout.csv"

TARGET = "churned"
ID_COLUMN = "customer_id"

Row = dict[str, str]
Matrix = list[list[float]]


# --------------------------------------------------------------------------------------------------
# argument coercion — a model calling a tool sends whatever it sends
# --------------------------------------------------------------------------------------------------


def as_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in ('true', 'yes', '1', 'y', 'on')


def as_list(value: Any) -> list[Any]:
    """Accept a list, a comma-separated string, or a single value, and return a list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [value]


def shown(path: str | Path) -> str:
    """Render a path the way this repo talks about one — `./benchmarks/...`, never a home directory.

    Paths travel further than they look: they go back to the model inside tool results, into the run
    transcript, and into the JSON artifacts that get committed and diffed. An absolute path makes all
    three specific to one machine, so everything that leaves this module is written relative to the
    repository root. A path genuinely outside the repo has nothing relative to say and is left alone.
    """
    resolved = Path(path).resolve()
    try:
        return f"./{resolved.relative_to(REPO_ROOT)}"
    except ValueError:
        return str(resolved)


def resolve(path: str | Path | None, default: Path) -> Path:
    """Resolve a data path given as absolute, relative to the repo root, or relative to `benchmarks/`."""
    if not path:
        return default

    candidate = Path(path)
    if candidate.exists():
        return candidate

    for base in (BENCHMARK_DIR, BENCHMARK_DIR.parent):
        if (base / path).exists():
            return base / path

    return candidate  # let the caller fail on a path that genuinely isn't there


# --------------------------------------------------------------------------------------------------
# csv and column handling
# --------------------------------------------------------------------------------------------------


def read_csv(path: str | Path | None = None, default: Path = TRAIN_DATA) -> list[Row]:
    """Read a CSV into a list of dicts, keeping every value as the string the file holds."""
    resolved = resolve(path, default)
    if not resolved.exists():
        raise FileNotFoundError(f"no such dataset: {shown(resolved)}. Run `python benchmarks/dataset.py` first.")

    with open(resolved, newline='') as file:
        return list(csv.DictReader(file))


def write_csv(path: str | Path, rows: Sequence[Row], columns: Sequence[str] | None = None) -> str:
    """Write rows back out, and return the path written to."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = list(columns or (rows[0].keys() if rows else []))

    with open(target, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    return shown(target)


def number(value: Any) -> float | None:
    """Parse a cell as a float, returning None for anything that isn't one (including the empty cell)."""
    if value is None or value == '':
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def column_kinds(rows: Sequence[Row], threshold: float = 0.9) -> dict[str, str]:
    """Label each column "numeric" or "categorical", by how much of it parses as a number."""
    kinds: dict[str, str] = {}
    for column in (rows[0].keys() if rows else []):
        present = [row[column] for row in rows if row.get(column) not in (None, '')]
        numeric = sum(1 for value in present if number(value) is not None)
        kinds[column] = "numeric" if present and numeric / len(present) >= threshold else "categorical"

    return kinds


def feature_columns(rows: Sequence[Row], target: str = TARGET) -> list[str]:
    """Every column a model may learn from — everything but the id and the target."""
    return [column for column in (rows[0].keys() if rows else []) if column not in (ID_COLUMN, target)]


def design_matrix(
    rows: Sequence[Row],
    target: str = TARGET,
    columns: Sequence[str] | None = None,
) -> tuple[Matrix, list[int], list[str]]:
    """Turn rows into the `(X, y, feature_names)` every model here trains on.

    Numeric columns are used as they are, with a missing value filled by the column mean; categorical
    columns are one-hot expanded, one feature per level seen. The feature names come back so a model's
    weights can be reported against the column they belong to.
    """
    columns = list(columns or feature_columns(rows, target))
    kinds = column_kinds(rows)

    means: dict[str, float] = {}
    levels: dict[str, list[str]] = {}
    for column in columns:
        if kinds.get(column) == "numeric":
            values = [number(row.get(column)) for row in rows]
            present = [value for value in values if value is not None]
            means[column] = statistics.fmean(present) if present else 0.0
        else:
            levels[column] = sorted({row.get(column, '') for row in rows})

    names: list[str] = []
    for column in columns:
        names.extend([column] if column in means else [f"{column}={level}" for level in levels[column]])

    matrix: Matrix = []
    labels: list[int] = []
    for row in rows:
        features: list[float] = []
        for column in columns:
            if column in means:
                value = number(row.get(column))
                features.append(means[column] if value is None else value)
            else:
                features.extend(1.0 if row.get(column, '') == level else 0.0 for level in levels[column])

        matrix.append(features)
        labels.append(as_int(number(row.get(target)), 0))

    return matrix, labels, names


def standardize(matrix: Matrix) -> tuple[Matrix, list[float], list[float]]:
    """Centre and scale every column, returning the means and deviations used, so scoring can reuse them."""
    if not matrix:
        return [], [], []

    width = len(matrix[0])
    means = [statistics.fmean(row[index] for row in matrix) for index in range(width)]
    deviations = [
        statistics.pstdev([row[index] for row in matrix], means[index]) or 1.0 for index in range(width)
    ]
    scaled = [
        [(row[index] - means[index]) / deviations[index] for index in range(width)] for row in matrix
    ]
    return scaled, means, deviations


def apply_scaling(matrix: Matrix, means: Sequence[float], deviations: Sequence[float]) -> Matrix:
    """Scale a matrix with means and deviations measured somewhere else — a training set, usually."""
    return [
        [(value - means[index]) / deviations[index] for index, value in enumerate(row)] for row in matrix
    ]


# --------------------------------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------------------------------


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def fit_logistic(
    matrix: Matrix,
    labels: Sequence[int],
    learning_rate: float = 0.1,
    epochs: int = 300,
    l2: float = 0.01,
    class_weight: bool = True,
) -> list[float]:
    """Fit a logistic regression by full-batch gradient descent, returning `[bias, *weights]`.

    `class_weight` scales each class's contribution by how rare it is, which matters here: churn is the
    minority class, and without it the model can do well on accuracy by rarely predicting churn at all.
    """
    width = len(matrix[0]) if matrix else 0
    weights = [0.0] * (width + 1)
    positives = sum(labels) or 1
    negatives = len(labels) - positives or 1
    weight_for = {
        1: len(labels) / (2 * positives) if class_weight else 1.0,
        0: len(labels) / (2 * negatives) if class_weight else 1.0,
    }

    for _ in range(epochs):
        gradient = [0.0] * (width + 1)
        for row, label in zip(matrix, labels):
            error = (sigmoid(weights[0] + sum(w * x for w, x in zip(weights[1:], row))) - label)
            error *= weight_for[label]
            gradient[0] += error
            for index, value in enumerate(row):
                gradient[index + 1] += error * value

        weights[0] -= learning_rate * gradient[0] / len(matrix)
        for index in range(width):
            step = gradient[index + 1] / len(matrix) + l2 * weights[index + 1]
            weights[index + 1] -= learning_rate * step

    return weights


def logistic_scores(matrix: Matrix, weights: Sequence[float]) -> list[float]:
    """Predicted probabilities for every row."""
    return [sigmoid(weights[0] + sum(w * x for w, x in zip(weights[1:], row))) for row in matrix]


def fit_linear(
    matrix: Matrix,
    values: Sequence[float],
    learning_rate: float = 0.05,
    epochs: int = 400,
    l2: float = 0.0,
) -> list[float]:
    """Fit a least-squares linear regression by gradient descent, returning `[bias, *weights]`."""
    width = len(matrix[0]) if matrix else 0
    weights = [0.0] * (width + 1)

    for _ in range(epochs):
        gradient = [0.0] * (width + 1)
        for row, value in zip(matrix, values):
            error = (weights[0] + sum(w * x for w, x in zip(weights[1:], row))) - value
            gradient[0] += error
            for index, feature in enumerate(row):
                gradient[index + 1] += error * feature

        weights[0] -= learning_rate * gradient[0] / len(matrix)
        for index in range(width):
            weights[index + 1] -= learning_rate * (gradient[index + 1] / len(matrix) + l2 * weights[index + 1])

    return weights


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the ROC curve, computed from the rank of the positives (ties share their rank)."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return 0.5

    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[position]]:
            end += 1
        shared = (position + end) / 2 + 1  # average rank across the tie, 1-based
        for index in range(position, end + 1):
            ranks[order[index]] = shared
        position = end + 1

    positive_ranks = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (positive_ranks - positives * (positives + 1) / 2) / (positives * negatives)


def confusion(labels: Sequence[int], scores: Sequence[float], threshold: float) -> dict[str, int]:
    matrix = {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}
    for label, score in zip(labels, scores):
        predicted = 1 if score >= threshold else 0
        if predicted == 1 and label == 1:
            matrix["true_positive"] += 1
        elif predicted == 1:
            matrix["false_positive"] += 1
        elif label == 1:
            matrix["false_negative"] += 1
        else:
            matrix["true_negative"] += 1

    return matrix


def classification_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Accuracy, precision, recall, F1 and ROC AUC, plus the confusion matrix they come from."""
    matrix = confusion(labels, scores, threshold)
    predicted_positive = matrix["true_positive"] + matrix["false_positive"]
    actual_positive = matrix["true_positive"] + matrix["false_negative"]

    precision = matrix["true_positive"] / predicted_positive if predicted_positive else 0.0
    recall = matrix["true_positive"] / actual_positive if actual_positive else 0.0
    return {
        "threshold": round(threshold, 4),
        "accuracy": round((matrix["true_positive"] + matrix["true_negative"]) / len(labels), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        "roc_auc": round(roc_auc(labels, scores), 4),
        "confusion_matrix": matrix,
        "positive_rate": round(actual_positive / len(labels), 4),
    }


def save_model(model: dict[str, Any], path: str | Path | None, name: str) -> str:
    """Write a fitted model to JSON so a later pipeline step can score with it."""
    target = Path(path) if path else ARTIFACTS_DIR / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(model, indent=2))
    return shown(target)


def load_model(path: str | Path | None = None) -> dict[str, Any]:
    """Read back a model saved by one of the training tools."""
    resolved = resolve(path, ARTIFACTS_DIR / "classification_model.json")
    if not resolved.exists():
        raise FileNotFoundError(f"no model at {shown(resolved)}; train one with classification_tool first.")

    return json.loads(resolved.read_text())


def score_rows(model: dict[str, Any], rows: Sequence[Row]) -> tuple[list[float], list[int]]:
    """Score rows with a saved logistic model, rebuilding its exact feature layout."""
    matrix, labels, names = design_matrix(rows, model.get("target", TARGET), model.get("columns"))

    # A holdout split can be missing a category the training set had, so features are matched by name
    # rather than by position — a name the model never saw contributes nothing.
    index_of = {name: position for position, name in enumerate(names)}
    aligned = [
        [row[index_of[name]] if name in index_of else 0.0 for name in model["features"]] for row in matrix
    ]
    scaled = apply_scaling(aligned, model["means"], model["deviations"])
    return logistic_scores(scaled, model["weights"]), labels


# --------------------------------------------------------------------------------------------------
# data tools
# --------------------------------------------------------------------------------------------------


def data_ingestion_caller(
    data_source: str | None = None,
    preprocessing_steps: list[str] | None = None,
) -> dict[str, Any]:
    """Read a dataset in and apply the named preprocessing steps, writing what it ingested.

    Steps: "strip_whitespace", "drop_duplicates", "drop_missing_target", "coerce_numeric".
    """
    rows = read_csv(data_source)
    steps = as_list(preprocessing_steps) or ["strip_whitespace", "drop_missing_target"]
    applied: dict[str, int] = {}

    if "strip_whitespace" in steps:
        touched = 0
        for row in rows:
            for column, value in row.items():
                if isinstance(value, str) and value != value.strip():
                    row[column] = value.strip()
                    touched += 1
        applied["strip_whitespace"] = touched

    if "drop_duplicates" in steps:
        seen: set[tuple] = set()
        kept = []
        for row in rows:
            signature = tuple(sorted(row.items()))
            if signature not in seen:
                seen.add(signature)
                kept.append(row)
        applied["drop_duplicates"] = len(rows) - len(kept)
        rows = kept

    if "drop_missing_target" in steps:
        kept = [row for row in rows if row.get(TARGET) not in (None, '')]
        applied["drop_missing_target"] = len(rows) - len(kept)
        rows = kept

    if "coerce_numeric" in steps:
        kinds = column_kinds(rows)
        coerced = 0
        for row in rows:
            for column, kind in kinds.items():
                if kind == "numeric" and row.get(column) not in (None, '') and number(row[column]) is None:
                    row[column] = ''
                    coerced += 1
        applied["coerce_numeric"] = coerced

    path = write_csv(ARTIFACTS_DIR / "ingested.csv", rows)
    return {
        "source": shown(resolve(data_source, TRAIN_DATA)),
        "output_path": path,
        "rows": len(rows),
        "columns": list(rows[0].keys()) if rows else [],
        "steps_applied": applied,
    }


def data_reader_caller(
    data_source: str | None = None,
    limit: int = 5,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """Read a dataset and describe it: shape, column kinds, missing values, target balance, a few rows.

    This is the tool that answers "what am I working with?", so its output is what an agent quotes when
    it reasons about the data rather than about the problem statement.
    """
    rows = read_csv(data_source)
    wanted = as_list(columns) or list(rows[0].keys())
    kinds = column_kinds(rows)

    summary: dict[str, Any] = {}
    for column in wanted:
        values = [row.get(column, '') for row in rows]
        missing = sum(1 for value in values if value in (None, ''))
        entry: dict[str, Any] = {"kind": kinds.get(column), "missing": missing}
        if kinds.get(column) == "numeric":
            present = [number(value) for value in values if number(value) is not None]
            if present:
                entry.update(
                    min=round(min(present), 3),
                    max=round(max(present), 3),
                    mean=round(statistics.fmean(present), 3),
                    median=round(statistics.median(present), 3),
                    stdev=round(statistics.pstdev(present), 3),
                )
        else:
            counts: dict[str, int] = {}
            for value in values:
                counts[value] = counts.get(value, 0) + 1
            entry["levels"] = dict(sorted(counts.items(), key=lambda item: -item[1])[:8])

        summary[column] = entry

    labels = [as_int(number(row.get(TARGET)), 0) for row in rows if row.get(TARGET) not in (None, '')]
    return {
        "source": shown(resolve(data_source, TRAIN_DATA)),
        "rows": len(rows),
        "columns": len(wanted),
        "target": TARGET,
        "class_balance": {
            "positive": sum(labels),
            "negative": len(labels) - sum(labels),
            "positive_rate": round(sum(labels) / len(labels), 4) if labels else 0.0,
        },
        "schema": summary,
        "sample": [{column: row.get(column) for column in wanted} for row in rows[: as_int(limit, 5)]],
    }


def data_cleaning_caller(
    data_source: str | None = None,
    missing_strategy: str = "median",
    drop_duplicates: bool = True,
    outlier_method: str = "iqr",
    output_path: str | None = None,
) -> dict[str, Any]:
    """Impute what is missing, clip the outliers, drop the duplicates, and report every change made.

    `missing_strategy` is "median", "mean" or "drop"; `outlier_method` is "iqr" (clip beyond 1.5×IQR)
    or "none".
    """
    rows = read_csv(data_source)
    kinds = column_kinds(rows)
    report: dict[str, Any] = {
        "imputed": {},
        "clipped": {},
        "clipping_skipped": {},
        "duplicates_removed": 0,
        "rows_dropped": 0,
    }

    if as_bool(drop_duplicates):
        seen: set[tuple] = set()
        kept = []
        for row in rows:
            signature = tuple(sorted(row.items()))
            if signature not in seen:
                seen.add(signature)
                kept.append(row)
        report["duplicates_removed"] = len(rows) - len(kept)
        rows = kept

    for column, kind in kinds.items():
        if kind != "numeric" or column in (ID_COLUMN, TARGET):
            continue

        present = [number(row[column]) for row in rows if number(row.get(column)) is not None]
        if not present:
            continue

        if missing_strategy == "drop":
            before = len(rows)
            rows = [row for row in rows if number(row.get(column)) is not None]
            report["rows_dropped"] += before - len(rows)
        else:
            fill = statistics.median(present) if missing_strategy == "median" else statistics.fmean(present)
            filled = 0
            for row in rows:
                if number(row.get(column)) is None:
                    row[column] = f"{round(fill, 3)}"
                    filled += 1
            if filled:
                report["imputed"][column] = {"strategy": missing_strategy, "count": filled, "value": round(fill, 3)}

        if outlier_method == "iqr":
            values = sorted(number(row[column]) for row in rows if number(row.get(column)) is not None)
            if len(values) < 4:
                continue

            # An IQR rule only means anything on a continuous measurement. On a flag it is destructive —
            # `is_senior` has a zero IQR, so bounds of [0, 0] would quietly rewrite every senior to 0 —
            # and on a count it clips the tail that carries the signal. Both are left alone, and the
            # decision is reported rather than made silently.
            lower_quartile = values[len(values) // 4]
            upper_quartile = values[3 * len(values) // 4]
            spread = upper_quartile - lower_quartile
            if spread <= 0:
                report["clipping_skipped"][column] = "zero interquartile range — a flag or a constant"
                continue
            if all(float(value).is_integer() for value in values):
                report["clipping_skipped"][column] = "whole-number column — a count or an ordinal, not a measurement"
                continue

            low, high = lower_quartile - 1.5 * spread, upper_quartile + 1.5 * spread
            clipped = 0
            for row in rows:
                value = number(row.get(column))
                if value is None:
                    continue
                if value < low or value > high:
                    row[column] = f"{round(min(high, max(low, value)), 3)}"
                    clipped += 1
            if clipped:
                report["clipped"][column] = {"count": clipped, "bounds": [round(low, 3), round(high, 3)]}

    report["output_path"] = write_csv(output_path or ARTIFACTS_DIR / "cleaned.csv", rows)
    report["rows"] = len(rows)
    return report


def data_transformation_caller(
    data_source: str | None = None,
    transformations: list[dict[str, str]] | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Apply column transformations — "log", "sqrt", "zscore", "bucket" — writing the transformed data.

    Each transformation is a `{"column": ..., "operation": ...}` mapping. Without any, the skewed money
    columns get a log transform, which is the usual default for this dataset.
    """
    rows = read_csv(data_source)
    requested = as_list(transformations) or [
        {"column": "total_charges", "operation": "log"},
        {"column": "monthly_charges", "operation": "log"},
    ]

    applied: list[dict[str, Any]] = []
    for transformation in requested:
        if isinstance(transformation, str):  # "total_charges:log" is accepted too
            column, _, operation = transformation.partition(":")
            transformation = {"column": column, "operation": operation or "log"}

        column = transformation.get("column", '')
        operation = str(transformation.get("operation", "log")).lower()
        values = [number(row.get(column)) for row in rows]
        if not any(value is not None for value in values):
            applied.append({"column": column, "operation": operation, "skipped": "not a numeric column"})
            continue

        present = [value for value in values if value is not None]
        mean = statistics.fmean(present)
        deviation = statistics.pstdev(present) or 1.0
        quartiles = sorted(present)

        for row in rows:
            value = number(row.get(column))
            if value is None:
                continue
            if operation == "log":
                row[f"{column}_log"] = f"{round(math.log1p(max(0.0, value)), 5)}"
            elif operation == "sqrt":
                row[f"{column}_sqrt"] = f"{round(math.sqrt(max(0.0, value)), 5)}"
            elif operation == "zscore":
                row[f"{column}_zscore"] = f"{round((value - mean) / deviation, 5)}"
            elif operation == "bucket":
                position = sum(1 for edge in (quartiles[len(quartiles) // 4], statistics.median(quartiles),
                                              quartiles[3 * len(quartiles) // 4]) if value > edge)
                row[f"{column}_bucket"] = f"q{position + 1}"

        applied.append({"column": column, "operation": operation, "new_column": f"{column}_{operation}"})

    return {
        "output_path": write_csv(output_path or ARTIFACTS_DIR / "transformed.csv", rows),
        "rows": len(rows),
        "transformations": applied,
    }


def data_validation_caller(
    data_source: str | None = None,
    target: str = TARGET,
    expectations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check a dataset is fit to train on, and return one pass/fail per check with the numbers behind it.

    Checks: the target is present, binary and not degenerate; ids are unique; no column is entirely
    missing; the row count clears a minimum; missingness stays under a ceiling.
    """
    rows = read_csv(data_source)
    expectations = expectations or {}
    minimum_rows = as_int(expectations.get("min_rows"), 500)
    maximum_missing = as_float(expectations.get("max_missing_rate"), 0.10)
    balance_floor = as_float(expectations.get("min_positive_rate"), 0.02)

    labels = [row.get(target, '') for row in rows]
    distinct = {label for label in labels if label != ''}
    positives = sum(1 for label in labels if label == "1")
    identifiers = [row.get(ID_COLUMN, '') for row in rows]

    missing_rates = {
        column: round(sum(1 for row in rows if row.get(column) in (None, '')) / len(rows), 4)
        for column in (rows[0].keys() if rows else [])
    }
    worst_column = max(missing_rates, key=lambda column: missing_rates[column]) if missing_rates else ''

    checks = [
        {"check": "row_count", "passed": len(rows) >= minimum_rows, "rows": len(rows), "minimum": minimum_rows},
        {"check": "target_present", "passed": all(label != '' for label in labels), "missing": labels.count('')},
        {"check": "target_binary", "passed": distinct <= {"0", "1"}, "values": sorted(distinct)},
        {
            "check": "class_balance",
            "passed": len(rows) > 0 and positives / len(rows) >= balance_floor,
            "positive_rate": round(positives / len(rows), 4) if rows else 0.0,
            "floor": balance_floor,
        },
        {
            "check": "unique_ids",
            "passed": len(set(identifiers)) == len(identifiers),
            "duplicates": len(identifiers) - len(set(identifiers)),
        },
        {
            "check": "missingness",
            "passed": all(rate <= maximum_missing for rate in missing_rates.values()),
            "worst_column": worst_column,
            "worst_rate": missing_rates.get(worst_column, 0.0),
            "ceiling": maximum_missing,
        },
    ]

    return {
        "source": shown(resolve(data_source, TRAIN_DATA)),
        "passed": all(check["passed"] for check in checks),
        "failed_checks": [check["check"] for check in checks if not check["passed"]],
        "checks": checks,
    }


# --------------------------------------------------------------------------------------------------
# feature tools
# --------------------------------------------------------------------------------------------------


DERIVED_FEATURES: dict[str, Callable[[Row], float | str | None]] = {
    # name -> how it is computed from one row; these are the domain features this dataset supports
    "charges_per_month": lambda row: (
        number(row.get("total_charges")) / max(1.0, number(row.get("tenure_months")) or 1.0)
        if number(row.get("total_charges")) is not None else None
    ),
    "tickets_per_year": lambda row: (
        (number(row.get("support_tickets")) or 0.0) * 12 / max(1.0, number(row.get("tenure_months")) or 1.0)
    ),
    "engagement_index": lambda row: (
        (number(row.get("avg_session_minutes")) or 0.0) * (number(row.get("num_products")) or 1.0)
    ),
    "revenue_per_product": lambda row: (
        (number(row.get("monthly_charges")) or 0.0) / max(1.0, number(row.get("num_products")) or 1.0)
    ),
    "is_month_to_month": lambda row: 1.0 if row.get("contract_type") == "month-to-month" else 0.0,
    "pays_by_echeck": lambda row: 1.0 if row.get("payment_method") == "electronic_check" else 0.0,
    "tenure_bucket": lambda row: (
        "new" if (number(row.get("tenure_months")) or 0) <= 6
        else "growing" if (number(row.get("tenure_months")) or 0) <= 24
        else "established"
    ),
}


def add_features(rows: list[Row], names: Iterable[str]) -> list[str]:
    """Add the named derived features to every row, and return the ones that were added."""
    added = []
    for name in names:
        build = DERIVED_FEATURES.get(name)
        if not build:
            continue
        for row in rows:
            value = build(row)
            row[name] = '' if value is None else (value if isinstance(value, str) else f"{round(value, 5)}")
        added.append(name)

    return added


def feature_engineering_caller(
    feature_list: list[str] | None = None,
    transformation_steps: list[str] | None = None,
    data_source: str | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Derive the named domain features, then apply the named transformation steps to them.

    Available features: charges_per_month, tickets_per_year, engagement_index, revenue_per_product,
    is_month_to_month, pays_by_echeck, tenure_bucket. Steps are the operations `data_transformation_tool`
    understands, applied to whatever was derived.
    """
    rows = read_csv(data_source)
    wanted = as_list(feature_list) or list(DERIVED_FEATURES)
    added = add_features(rows, wanted)

    path = write_csv(ARTIFACTS_DIR / "engineered.csv", rows)
    result: dict[str, Any] = {
        "output_path": path,
        "rows": len(rows),
        "features_added": added,
        "features_skipped": [name for name in wanted if name not in added],
    }

    steps = as_list(transformation_steps)
    if steps:
        transformations = [
            {"column": name, "operation": step}
            for name in added
            for step in steps
            if step in ("log", "sqrt", "zscore", "bucket")
        ]
        transformed = data_transformation_caller(path, transformations, output_path or path)
        result["transformations"] = transformed["transformations"]
        result["output_path"] = transformed["output_path"]

    return result


def feature_generator_caller(
    data_source: str | None = None,
    max_features: int = 6,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Generate ratio features automatically from the numeric columns, keeping the most target-correlated.

    Every numeric pair gives a ratio; the ones whose correlation with the target beats their parents'
    are kept, up to `max_features`. This is the crude version of automated feature generation, and its
    value is that the correlations it reports are measured rather than assumed.
    """
    rows = read_csv(data_source)
    kinds = column_kinds(rows)
    numeric = [
        column for column, kind in kinds.items()
        if kind == "numeric" and column not in (ID_COLUMN, TARGET)
    ]
    labels = [as_int(number(row.get(TARGET)), 0) for row in rows]

    def correlation(values: Sequence[float]) -> float:
        """Point-biserial correlation between a numeric column and the binary target."""
        if len(set(values)) <= 1:
            return 0.0
        mean = statistics.fmean(values)
        deviation = statistics.pstdev(values, mean) or 1.0
        label_mean = statistics.fmean(labels)
        label_deviation = statistics.pstdev(labels, label_mean) or 1.0
        covariance = statistics.fmean(
            [(value - mean) * (label - label_mean) for value, label in zip(values, labels)]
        )
        return covariance / (deviation * label_deviation)

    base = {}
    for column in numeric:
        values = [number(row.get(column)) or 0.0 for row in rows]
        base[column] = abs(correlation(values))

    candidates: list[dict[str, Any]] = []
    for index, left in enumerate(numeric):
        for right in numeric[index + 1:]:
            values = [
                (number(row.get(left)) or 0.0) / (abs(number(row.get(right)) or 0.0) + 1e-6) for row in rows
            ]
            strength = abs(correlation(values))
            if strength > max(base[left], base[right]):
                candidates.append({
                    "name": f"{left}_per_{right}",
                    "correlation": round(strength, 4),
                    "beats": {left: round(base[left], 4), right: round(base[right], 4)},
                    "values": values,
                })

    kept = sorted(candidates, key=lambda candidate: -candidate["correlation"])[: as_int(max_features, 6)]
    for candidate in kept:
        for row, value in zip(rows, candidate.pop("values")):
            row[candidate["name"]] = f"{round(value, 5)}"

    return {
        "output_path": write_csv(output_path or ARTIFACTS_DIR / "generated_features.csv", rows),
        "rows": len(rows),
        "generated": kept,
        "candidates_considered": len(numeric) * (len(numeric) - 1) // 2,
    }


def feature_encoding_caller(
    data_source: str | None = None,
    columns: list[str] | None = None,
    method: str = "one_hot",
    output_path: str | None = None,
) -> dict[str, Any]:
    """Encode categorical columns as numbers — "one_hot" (a column per level) or "ordinal"."""
    rows = read_csv(data_source)
    kinds = column_kinds(rows)
    wanted = as_list(columns) or [
        column for column, kind in kinds.items()
        if kind == "categorical" and column not in (ID_COLUMN, TARGET)
    ]

    encoded: dict[str, Any] = {}
    for column in wanted:
        levels = sorted({row.get(column, '') for row in rows})
        if method == "ordinal":
            positions = {level: index for index, level in enumerate(levels)}
            for row in rows:
                row[f"{column}_ordinal"] = f"{positions.get(row.get(column, ''), -1)}"
            encoded[column] = {"method": "ordinal", "levels": positions}
        else:
            for level in levels:
                for row in rows:
                    row[f"{column}={level}"] = "1" if row.get(column, '') == level else "0"
            encoded[column] = {"method": "one_hot", "columns": [f"{column}={level}" for level in levels]}

    return {
        "output_path": write_csv(output_path or ARTIFACTS_DIR / "encoded.csv", rows),
        "rows": len(rows),
        "encoded": encoded,
    }


def feature_scaling_caller(
    data_source: str | None = None,
    columns: list[str] | None = None,
    method: str = "standard",
    output_path: str | None = None,
) -> dict[str, Any]:
    """Scale numeric columns — "standard" (zero mean, unit variance) or "minmax" (to 0..1)."""
    rows = read_csv(data_source)
    kinds = column_kinds(rows)
    wanted = as_list(columns) or [
        column for column, kind in kinds.items()
        if kind == "numeric" and column not in (ID_COLUMN, TARGET)
    ]

    scaled: dict[str, Any] = {}
    for column in wanted:
        values = [number(row.get(column)) for row in rows]
        present = [value for value in values if value is not None]
        if not present:
            continue

        if method == "minmax":
            low, high = min(present), max(present)
            spread = (high - low) or 1.0
            for row in rows:
                value = number(row.get(column))
                if value is not None:
                    row[column] = f"{round((value - low) / spread, 5)}"
            scaled[column] = {"method": "minmax", "min": round(low, 3), "max": round(high, 3)}
        else:
            mean = statistics.fmean(present)
            deviation = statistics.pstdev(present, mean) or 1.0
            for row in rows:
                value = number(row.get(column))
                if value is not None:
                    row[column] = f"{round((value - mean) / deviation, 5)}"
            scaled[column] = {"method": "standard", "mean": round(mean, 3), "stdev": round(deviation, 3)}

    return {
        "output_path": write_csv(output_path or ARTIFACTS_DIR / "scaled.csv", rows),
        "rows": len(rows),
        "scaled": scaled,
    }


def feature_selection_caller(
    data_source: str | None = None,
    target: str = TARGET,
    top_k: int = 10,
    method: str = "correlation",
) -> dict[str, Any]:
    """Rank features by how much they tell you about the target, and return the strongest `top_k`.

    "correlation" is the absolute point-biserial correlation; "mutual_information" bins each feature into
    quartiles and measures the mutual information with the label. Both are reported per feature, so the
    ranking can be argued with rather than taken on faith.
    """
    rows = read_csv(data_source)
    matrix, labels, names = design_matrix(rows, target)
    if not matrix:
        return {"selected": [], "scores": []}

    label_mean = statistics.fmean(labels)
    label_deviation = statistics.pstdev(labels, label_mean) or 1.0

    scores: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        values = [row[index] for row in matrix]
        mean = statistics.fmean(values)
        deviation = statistics.pstdev(values, mean) or 1.0
        covariance = statistics.fmean(
            [(value - mean) * (label - label_mean) for value, label in zip(values, labels)]
        )
        correlation = covariance / (deviation * label_deviation)

        # mutual information over quartile bins — catches a relationship correlation would miss
        ordered = sorted(values)
        edges = [ordered[len(ordered) // 4], statistics.median(ordered), ordered[3 * len(ordered) // 4]]
        information = 0.0
        for bin_index in range(4):
            in_bin = [
                label for value, label in zip(values, labels)
                if sum(1 for edge in edges if value > edge) == bin_index
            ]
            if not in_bin:
                continue
            bin_probability = len(in_bin) / len(labels)
            for label_value in (0, 1):
                joint = sum(1 for label in in_bin if label == label_value) / len(labels)
                marginal = (label_mean if label_value else 1 - label_mean) or 1e-9
                if joint > 0:
                    information += joint * math.log(joint / (bin_probability * marginal) + 1e-12)

        scores.append({
            "feature": name,
            "correlation": round(correlation, 4),
            "abs_correlation": round(abs(correlation), 4),
            "mutual_information": round(max(0.0, information), 5),
        })

    key = "mutual_information" if method == "mutual_information" else "abs_correlation"
    ranked = sorted(scores, key=lambda score: -score[key])
    return {
        "method": method,
        "selected": [score["feature"] for score in ranked[: as_int(top_k, 10)]],
        "scores": ranked,
    }


def feature_extraction_caller(
    data_source: str | None = None,
    aggregations: list[str] | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Extract composite features that summarise several columns at once.

    Unlike `feature_generator_tool`, which searches pairs mechanically, these are the composites this
    domain suggests: what a customer is worth, how much friction they hit, and how locked in they are.
    """
    rows = read_csv(data_source)
    wanted = as_list(aggregations) or ["value_score", "friction_score", "commitment_score"]

    lock_in = {"month-to-month": 0.0, "one-year": 0.5, "two-year": 1.0}
    composites: dict[str, Callable[[Row], float]] = {
        "value_score": lambda row: (
            (number(row.get("monthly_charges")) or 0.0) * (number(row.get("num_products")) or 1.0) / 100.0
        ),
        "friction_score": lambda row: (
            (number(row.get("support_tickets")) or 0.0) / 3.0
            - (number(row.get("avg_session_minutes")) or 0.0) / 60.0
        ),
        "commitment_score": lambda row: (
            lock_in.get(row.get("contract_type", ''), 0.0)
            + min(1.0, (number(row.get("tenure_months")) or 0.0) / 48.0)
            + (0.5 if row.get("has_premium_support") == "yes" else 0.0)
        ),
    }

    extracted = []
    for name in wanted:
        build = composites.get(name)
        if not build:
            continue
        for row in rows:
            row[name] = f"{round(build(row), 5)}"
        extracted.append(name)

    return {
        "output_path": write_csv(output_path or ARTIFACTS_DIR / "extracted.csv", rows),
        "rows": len(rows),
        "extracted": extracted,
    }


# --------------------------------------------------------------------------------------------------
# model tools
# --------------------------------------------------------------------------------------------------


def classification_caller(
    data_source: str | None = None,
    target: str = TARGET,
    hyperparameters: dict[str, Any] | None = None,
    model_path: str | None = None,
) -> dict[str, Any]:
    """Train the binary classifier: a logistic regression, class-weighted for the minority target.

    Hyperparameters: `learning_rate`, `epochs`, `l2`, `class_weight`, `threshold`. The fitted model is
    saved to `model_path` (JSON) so `model_evaluation_tool` and `ranking_tool` can score with it.
    """
    settings = hyperparameters or {}
    rows = read_csv(data_source)
    matrix, labels, names = design_matrix(rows, target)
    scaled, means, deviations = standardize(matrix)

    weights = fit_logistic(
        scaled,
        labels,
        learning_rate=as_float(settings.get("learning_rate"), 0.3),
        epochs=as_int(settings.get("epochs"), 400),
        l2=as_float(settings.get("l2"), 0.01),
        class_weight=as_bool(settings.get("class_weight"), True),
    )

    threshold = as_float(settings.get("threshold"), 0.5)
    scores = logistic_scores(scaled, weights)
    model = {
        "model_type": "logistic_regression",
        "target": target,
        "columns": feature_columns(rows, target),
        "features": names,
        "weights": weights,
        "means": means,
        "deviations": deviations,
        "threshold": threshold,
        "hyperparameters": {
            "learning_rate": as_float(settings.get("learning_rate"), 0.3),
            "epochs": as_int(settings.get("epochs"), 400),
            "l2": as_float(settings.get("l2"), 0.01),
            "class_weight": as_bool(settings.get("class_weight"), True),
        },
        "trained_on": shown(resolve(data_source, TRAIN_DATA)),
        "training_rows": len(rows),
    }

    return {
        "model_path": save_model(model, model_path, "classification_model.json"),
        "model_type": "logistic_regression",
        "features": len(names),
        "training_metrics": classification_metrics(labels, scores, threshold),
        "top_coefficients": sorted(
            [{"feature": name, "weight": round(weight, 4)} for name, weight in zip(names, weights[1:])],
            key=lambda item: -abs(item["weight"]),
        )[:8],
    }


def regression_caller(
    data_source: str | None = None,
    target: str = "total_charges",
    hyperparameters: dict[str, Any] | None = None,
    model_path: str | None = None,
) -> dict[str, Any]:
    """Fit a linear regression on a continuous target, for the parts of the problem that aren't binary.

    Defaults to predicting `total_charges`, which is the one continuous column worth modelling here.
    Reports RMSE, MAE and R² on the data it was fitted to.
    """
    settings = hyperparameters or {}
    rows = [row for row in read_csv(data_source) if number(row.get(target)) is not None]
    matrix, _, names = design_matrix(rows, target)
    values = [number(row[target]) or 0.0 for row in rows]

    scaled, means, deviations = standardize(matrix)
    value_mean = statistics.fmean(values)
    value_deviation = statistics.pstdev(values, value_mean) or 1.0
    normalised = [(value - value_mean) / value_deviation for value in values]

    weights = fit_linear(
        scaled,
        normalised,
        learning_rate=as_float(settings.get("learning_rate"), 0.1),
        epochs=as_int(settings.get("epochs"), 500),
        l2=as_float(settings.get("l2"), 0.0),
    )

    predictions = [
        (weights[0] + sum(w * x for w, x in zip(weights[1:], row))) * value_deviation + value_mean
        for row in scaled
    ]
    errors = [prediction - actual for prediction, actual in zip(predictions, values)]
    total_variance = sum((value - value_mean) ** 2 for value in values) or 1.0

    model = {
        "model_type": "linear_regression",
        "target": target,
        "features": names,
        "weights": weights,
        "means": means,
        "deviations": deviations,
        "target_mean": value_mean,
        "target_deviation": value_deviation,
    }

    return {
        "model_path": save_model(model, model_path, "regression_model.json"),
        "model_type": "linear_regression",
        "target": target,
        "rows": len(rows),
        "rmse": round(math.sqrt(statistics.fmean([error ** 2 for error in errors])), 4),
        "mae": round(statistics.fmean([abs(error) for error in errors]), 4),
        "r2": round(1 - sum(error ** 2 for error in errors) / total_variance, 4),
    }


def ranking_caller(
    data_source: str | None = None,
    model_path: str | None = None,
    top_k: int = 20,
) -> dict[str, Any]:
    """Rank customers by predicted churn risk, and report how much of the churn the top slice captures.

    This is what turns a classifier into something a retention team can act on: the lift over random
    targeting at `top_k` says whether working the list beats working the base rate.
    """
    rows = read_csv(data_source, HOLDOUT_DATA)
    model = load_model(model_path)
    scores, labels = score_rows(model, rows)

    ranked = sorted(
        [
            {"customer_id": row.get(ID_COLUMN), "risk": round(score, 4), "actual": label}
            for row, score, label in zip(rows, scores, labels)
        ],
        key=lambda item: -item["risk"],
    )

    size = min(as_int(top_k, 20), len(ranked))
    captured = sum(item["actual"] for item in ranked[:size])
    total_positives = sum(labels) or 1
    base_rate = total_positives / len(labels)

    return {
        "scored_rows": len(ranked),
        "top_k": size,
        "top_customers": ranked[:size],
        "churners_in_top_k": captured,
        "capture_rate": round(captured / total_positives, 4),
        "precision_at_k": round(captured / size, 4) if size else 0.0,
        "lift_over_random": round((captured / size) / base_rate, 3) if size and base_rate else 0.0,
    }


def clustering_caller(
    data_source: str | None = None,
    n_clusters: int = 4,
    iterations: int = 25,
    seed: int = 7,
) -> dict[str, Any]:
    """Segment customers with k-means, and report the churn rate inside each segment.

    The segments are unsupervised, but reporting the target inside them is what makes the result useful:
    a cluster with a churn rate far above the base rate is a retention segment, not just a cluster.
    """
    rows = read_csv(data_source)
    matrix, labels, names = design_matrix(rows)
    scaled, _, _ = standardize(matrix)
    if not scaled:
        return {"clusters": []}

    count = max(2, as_int(n_clusters, 4))
    rng = random.Random(as_int(seed, 7))
    centroids = [list(row) for row in rng.sample(scaled, count)]
    assignment = [0] * len(scaled)

    for _ in range(max(1, as_int(iterations, 25))):
        moved = False
        for position, row in enumerate(scaled):
            distances = [sum((value - centre) ** 2 for value, centre in zip(row, centroid)) for centroid in centroids]
            nearest = distances.index(min(distances))
            if nearest != assignment[position]:
                assignment[position] = nearest
                moved = True

        for index in range(count):
            members = [row for row, cluster in zip(scaled, assignment) if cluster == index]
            if members:
                centroids[index] = [statistics.fmean(column) for column in zip(*members)]

        if not moved:
            break

    clusters = []
    base_rate = statistics.fmean(labels) if labels else 0.0
    for index in range(count):
        members = [position for position, cluster in enumerate(assignment) if cluster == index]
        if not members:
            continue
        churn = statistics.fmean([labels[position] for position in members])
        distinguishing = sorted(
            [{"feature": name, "z": round(centroids[index][position], 3)} for position, name in enumerate(names)],
            key=lambda item: -abs(item["z"]),
        )[:4]
        clusters.append({
            "cluster": index,
            "size": len(members),
            "share": round(len(members) / len(scaled), 4),
            "churn_rate": round(churn, 4),
            "lift_over_base": round(churn / base_rate, 3) if base_rate else 0.0,
            "distinguishing_features": distinguishing,
        })

    return {
        "n_clusters": count,
        "rows": len(scaled),
        "base_churn_rate": round(base_rate, 4),
        "clusters": sorted(clusters, key=lambda cluster: -cluster["churn_rate"]),
    }


def dimensionality_reduction_caller(
    data_source: str | None = None,
    n_components: int = 3,
    iterations: int = 200,
) -> dict[str, Any]:
    """Reduce the feature space with PCA, computed by power iteration with deflation.

    Reports the variance each component explains and the features loading onto it, which is what says
    whether the width of this dataset is real or mostly repetition.
    """
    rows = read_csv(data_source)
    matrix, _, names = design_matrix(rows)
    scaled, _, _ = standardize(matrix)
    if not scaled:
        return {"components": []}

    width = len(scaled[0])
    covariance = [[0.0] * width for _ in range(width)]
    for row in scaled:
        for i in range(width):
            for j in range(i, width):
                covariance[i][j] += row[i] * row[j]
    for i in range(width):
        for j in range(i, width):
            covariance[i][j] /= len(scaled)
            covariance[j][i] = covariance[i][j]

    total_variance = sum(covariance[index][index] for index in range(width)) or 1.0
    rng = random.Random(11)
    components = []

    for component in range(max(1, min(as_int(n_components, 3), width))):
        vector = [rng.gauss(0, 1) for _ in range(width)]
        eigenvalue = 0.0
        for _ in range(max(10, as_int(iterations, 200))):
            product = [sum(covariance[i][j] * vector[j] for j in range(width)) for i in range(width)]
            norm = math.sqrt(sum(value ** 2 for value in product)) or 1.0
            vector = [value / norm for value in product]
            eigenvalue = norm

        loadings = sorted(
            [{"feature": names[index], "loading": round(vector[index], 4)} for index in range(width)],
            key=lambda item: -abs(item["loading"]),
        )[:5]
        components.append({
            "component": component + 1,
            "explained_variance": round(eigenvalue, 4),
            "explained_variance_ratio": round(eigenvalue / total_variance, 4),
            "top_loadings": loadings,
        })

        for i in range(width):  # deflate, so the next iteration finds the next component
            for j in range(width):
                covariance[i][j] -= eigenvalue * vector[i] * vector[j]

    return {
        "features_in": width,
        "components": components,
        "cumulative_variance_explained": round(
            sum(component["explained_variance_ratio"] for component in components), 4
        ),
    }


def neural_network_caller(
    data_source: str | None = None,
    target: str = TARGET,
    hidden_units: int = 8,
    epochs: int = 120,
    learning_rate: float = 0.15,
    seed: int = 3,
) -> dict[str, Any]:
    """Train a one-hidden-layer neural network on the same problem, as the non-linear comparison.

    Whether it beats the logistic regression is the interesting part: on a dataset whose labels come
    from a linear model, it usually shouldn't, and an agent claiming otherwise is worth checking.
    """
    rows = read_csv(data_source)
    matrix, labels, names = design_matrix(rows, target)
    scaled, means, deviations = standardize(matrix)
    if not scaled:
        return {"error": "no data to train on"}

    width = len(scaled[0])
    units = max(2, as_int(hidden_units, 8))
    rng = random.Random(as_int(seed, 3))
    limit = math.sqrt(6 / (width + units))
    hidden = [[rng.uniform(-limit, limit) for _ in range(width)] for _ in range(units)]
    hidden_bias = [0.0] * units
    output = [rng.uniform(-limit, limit) for _ in range(units)]
    output_bias = 0.0
    rate = as_float(learning_rate, 0.15)

    for _ in range(max(1, as_int(epochs, 120))):
        for row, label in zip(scaled, labels):
            activations = [max(0.0, bias + sum(w * x for w, x in zip(weights, row)))
                           for weights, bias in zip(hidden, hidden_bias)]
            prediction = sigmoid(output_bias + sum(w * a for w, a in zip(output, activations)))
            error = prediction - label

            for index in range(units):
                gradient = error * output[index] * (1.0 if activations[index] > 0 else 0.0)
                output[index] -= rate * error * activations[index]
                hidden_bias[index] -= rate * gradient
                for position in range(width):
                    hidden[index][position] -= rate * gradient * row[position]
            output_bias -= rate * error

    scores = []
    for row in scaled:
        activations = [max(0.0, bias + sum(w * x for w, x in zip(weights, row)))
                       for weights, bias in zip(hidden, hidden_bias)]
        scores.append(sigmoid(output_bias + sum(w * a for w, a in zip(output, activations))))

    return {
        "model_type": "mlp",
        "architecture": f"{width}-{units}-1 (relu, sigmoid)",
        "epochs": as_int(epochs, 120),
        "training_metrics": classification_metrics(labels, scores, 0.5),
        "note": "trained with SGD on the training split; compare against classification_tool before choosing it.",
    }


def hyperparameter_tuning_caller(
    data_source: str | None = None,
    target: str = TARGET,
    grid: dict[str, list[Any]] | None = None,
    folds: int = 4,
) -> dict[str, Any]:
    """Search the logistic regression's hyperparameters by cross-validated ROC AUC, and return the best.

    The grid defaults to a small sweep over learning rate, L2 and epochs. Every combination's mean AUC
    is reported, not just the winner, so a marginal win over a simpler setting stays visible.
    """
    rows = read_csv(data_source)
    settings = grid or {"learning_rate": [0.1, 0.3], "l2": [0.001, 0.01, 0.1], "epochs": [200, 400]}

    results = []
    for learning_rate in as_list(settings.get("learning_rate")) or [0.3]:
        for l2 in as_list(settings.get("l2")) or [0.01]:
            for epochs in as_list(settings.get("epochs")) or [400]:
                hyperparameters = {
                    "learning_rate": as_float(learning_rate, 0.3),
                    "l2": as_float(l2, 0.01),
                    "epochs": as_int(epochs, 400),
                }
                validated = cross_validation_caller(data_source, target, folds, hyperparameters)
                results.append({**hyperparameters, "mean_roc_auc": validated["mean_roc_auc"],
                                "std_roc_auc": validated["std_roc_auc"]})

    ranked = sorted(results, key=lambda result: -result["mean_roc_auc"])
    return {
        "rows": len(rows),
        "folds": as_int(folds, 4),
        "combinations": len(results),
        "best": ranked[0] if ranked else {},
        "results": ranked,
    }


def model_development_caller(
    model_type: str = "classification",
    hyperparameters: dict[str, Any] | None = None,
    data_source: str | None = None,
    target: str = TARGET,
) -> dict[str, Any]:
    """Develop a model of the requested kind, dispatching to the tool that trains it.

    `model_type` is "classification", "regression", "neural_network" or "clustering" — the entry point an
    agent uses when it has decided what kind of problem it is looking at but not which tool trains it.
    """
    settings = hyperparameters or {}
    kind = str(model_type).strip().lower()

    if kind in ("classification", "classifier", "binary_classification", "logistic_regression"):
        return classification_caller(data_source, target, settings)
    if kind in ("regression", "linear_regression"):
        return regression_caller(data_source, settings.get("target", "total_charges"), settings)
    if kind in ("neural_network", "mlp", "deep_learning"):
        return neural_network_caller(
            data_source,
            target,
            hidden_units=settings.get("hidden_units", 8),
            epochs=settings.get("epochs", 120),
            learning_rate=settings.get("learning_rate", 0.15),
        )
    if kind in ("clustering", "kmeans", "segmentation"):
        return clustering_caller(data_source, settings.get("n_clusters", 4))

    return {
        "error": f"unknown model_type {model_type!r}",
        "supported": ["classification", "regression", "neural_network", "clustering"],
    }


# --------------------------------------------------------------------------------------------------
# evaluation, mlops and benchmarking tools
# --------------------------------------------------------------------------------------------------


def model_evaluation_caller(
    model_path: str | None = None,
    data_source: str | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Score a saved model against held-out data, at the threshold it was saved with or one you pass."""
    model = load_model(model_path)
    rows = read_csv(data_source, HOLDOUT_DATA)
    scores, labels = score_rows(model, rows)
    cut = as_float(threshold, model.get("threshold", 0.5))

    return {
        "model_type": model.get("model_type"),
        "evaluated_on": shown(resolve(data_source, HOLDOUT_DATA)),
        "rows": len(rows),
        "metrics": classification_metrics(labels, scores, cut),
        "baseline_majority_class": round(1 - statistics.fmean(labels), 4) if labels else 0.0,
    }


def performance_metrics_caller(
    model_path: str | None = None,
    data_source: str | None = None,
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Sweep the decision threshold and report the metrics at each one, plus the best F1 threshold.

    A single threshold hides the trade-off this problem is really about — catching churners versus
    bothering customers who were never going to leave — so the sweep is the honest answer.
    """
    model = load_model(model_path)
    rows = read_csv(data_source, HOLDOUT_DATA)
    scores, labels = score_rows(model, rows)
    wanted = as_list(metrics) or ["accuracy", "precision", "recall", "f1", "roc_auc"]

    sweep = [
        {key: value for key, value in classification_metrics(labels, scores, threshold / 20).items()
         if key in (*wanted, "threshold")}
        for threshold in range(4, 17)
    ]
    best = max(sweep, key=lambda entry: entry.get("f1", 0.0))

    return {
        "rows": len(rows),
        "requested_metrics": wanted,
        "at_default_threshold": classification_metrics(labels, scores, model.get("threshold", 0.5)),
        "best_f1_threshold": best.get("threshold"),
        "best_f1": best.get("f1"),
        "threshold_sweep": sweep,
    }


def cross_validation_caller(
    data_source: str | None = None,
    target: str = TARGET,
    folds: int = 5,
    hyperparameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run stratified k-fold cross-validation, returning the AUC per fold and its spread.

    The spread is the part that matters: a mean AUC with a wide spread across folds is a model whose
    single-split score was luck.
    """
    settings = hyperparameters or {}
    rows = read_csv(data_source)
    matrix, labels, _ = design_matrix(rows, target)
    count = max(2, as_int(folds, 5))

    # stratified: positives and negatives are dealt into the folds separately, so every fold keeps the
    # base rate — with an 18% positive class, an unstratified fold can end up almost pure.
    positives = [index for index, label in enumerate(labels) if label == 1]
    negatives = [index for index, label in enumerate(labels) if label == 0]
    rng = random.Random(17)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    assignments: list[list[int]] = [[] for _ in range(count)]
    for position, index in enumerate([*positives, *negatives]):
        assignments[position % count].append(index)

    scores = []
    for fold, holdout in enumerate(assignments):
        held = set(holdout)
        train_matrix = [row for index, row in enumerate(matrix) if index not in held]
        train_labels = [label for index, label in enumerate(labels) if index not in held]
        test_matrix = [matrix[index] for index in holdout]
        test_labels = [labels[index] for index in holdout]
        if not train_matrix or len(set(test_labels)) < 2:
            continue

        scaled, means, deviations = standardize(train_matrix)
        weights = fit_logistic(
            scaled,
            train_labels,
            learning_rate=as_float(settings.get("learning_rate"), 0.3),
            epochs=as_int(settings.get("epochs"), 400),
            l2=as_float(settings.get("l2"), 0.01),
        )
        predictions = logistic_scores(apply_scaling(test_matrix, means, deviations), weights)
        metrics = classification_metrics(test_labels, predictions, as_float(settings.get("threshold"), 0.5))
        scores.append({"fold": fold + 1, "rows": len(holdout), "roc_auc": metrics["roc_auc"],
                       "f1": metrics["f1"], "recall": metrics["recall"]})

    aucs = [score["roc_auc"] for score in scores]
    return {
        "folds": len(scores),
        "hyperparameters": settings,
        "mean_roc_auc": round(statistics.fmean(aucs), 4) if aucs else 0.0,
        "std_roc_auc": round(statistics.pstdev(aucs), 4) if len(aucs) > 1 else 0.0,
        "mean_f1": round(statistics.fmean([score["f1"] for score in scores]), 4) if scores else 0.0,
        "per_fold": scores,
    }


def mlops_integration_caller(
    deployment_environment: str = "cloud",
    monitoring_metrics: list[str] | None = None,
    model_path: str | None = None,
) -> dict[str, Any]:
    """Write the deployment manifest for a trained model: how it is served, watched, and rolled back.

    This is the step that turns a fitted model into something operable — it records the features the
    model needs at serving time, the metrics to watch, and the conditions under which it should be
    pulled. The manifest is written to `benchmarks/artifacts/deployment_manifest.json`.
    """
    model = load_model(model_path)
    watched = as_list(monitoring_metrics) or ["roc_auc", "precision", "recall", "prediction_drift", "null_rate"]

    manifest = {
        "model_type": model.get("model_type"),
        "target": model.get("target"),
        "environment": deployment_environment,
        "serving": {
            "input_columns": model.get("columns", []),
            "expanded_features": len(model.get("features", [])),
            "decision_threshold": model.get("threshold", 0.5),
            "scaling": "standardised with the training means and deviations stored in the model file",
        },
        "monitoring": {
            "metrics": watched,
            "cadence": "daily on scored traffic, weekly on labelled outcomes once churn is observable",
            "label_delay": "labels arrive one quarter after scoring, so live metrics lag by that much",
        },
        "rollback_criteria": [
            "ROC AUC on labelled traffic falls more than 0.05 below the holdout figure",
            "the share of rows scored above the threshold doubles week over week",
            "any serving feature exceeds a 5% null rate",
        ],
        "artifact": shown(resolve(model_path, ARTIFACTS_DIR / "classification_model.json")),
    }

    path = ARTIFACTS_DIR / "deployment_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return {"manifest_path": shown(path), "manifest": manifest}


def benchmarking_caller(
    benchmark_dataset: str | None = None,
    performance_metrics: list[str] | None = None,
    model_path: str | None = None,
) -> dict[str, Any]:
    """Measure the trained model against the baselines it has to beat to be worth deploying.

    The baselines are the ones that actually compete with a model in practice: predicting the majority
    class, and a single-rule heuristic (month-to-month contract) that a business already knows.
    """
    model = load_model(model_path)
    rows = read_csv(benchmark_dataset, HOLDOUT_DATA)
    scores, labels = score_rows(model, rows)
    wanted = as_list(performance_metrics) or ["accuracy", "precision", "recall", "f1", "roc_auc"]

    def keep(metrics: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in metrics.items() if key in wanted}

    heuristic = [1.0 if row.get("contract_type") == "month-to-month" else 0.0 for row in rows]
    majority = [0.0] * len(rows)

    model_metrics = classification_metrics(labels, scores, model.get("threshold", 0.5))
    comparisons = [
        {"approach": "trained model", **keep(model_metrics)},
        {"approach": "month-to-month rule", **keep(classification_metrics(labels, heuristic, 0.5))},
        {"approach": "majority class", **keep(classification_metrics(labels, majority, 0.5))},
    ]

    best_baseline = max(comparisons[1:], key=lambda entry: entry.get("f1", 0.0))
    return {
        "benchmark_dataset": shown(resolve(benchmark_dataset, HOLDOUT_DATA)),
        "rows": len(rows),
        "comparisons": comparisons,
        "beats_best_baseline": model_metrics["f1"] > best_baseline.get("f1", 0.0),
        "f1_gain_over_best_baseline": round(model_metrics["f1"] - best_baseline.get("f1", 0.0), 4),
    }


if __name__ == "__main__":
    # Runs the workflow the agents orchestrate, so the tools can be checked without any model in the loop.
    print(json.dumps({
        "profile": data_reader_caller(limit=2)["class_balance"],
        "validation": data_validation_caller()["failed_checks"],
        "cleaning": data_cleaning_caller()["duplicates_removed"],
        "training": classification_caller()["training_metrics"],
        "evaluation": model_evaluation_caller()["metrics"],
        "benchmark": benchmarking_caller()["comparisons"],
    }, indent=2))
