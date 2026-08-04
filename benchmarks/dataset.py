"""Generates the benchmark's dataset: a binary classification problem, subscription churn.

Each row is a customer of a fictional subscription business, and the target `churned` says whether they
cancelled in the following quarter. The labels are drawn from a logistic model over the features, so the
signal is real and known up front — a model trained on this data should land around 0.85 ROC AUC, which
is what makes the workflow's evaluation step meaningful rather than decorative.

The data is deliberately not clean. Some `total_charges` and `avg_session_minutes` are missing, a few
`monthly_charges` are billing-system outliers, and a handful of rows are duplicated — so the cleaning and
validation tools in `tools.py` have something to actually find and report.

Everything is seeded, so the CSVs regenerate byte for byte:

    python benchmarks/dataset.py
"""

import csv
import math
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
REPO_ROOT = DATA_DIR.parent.parent
TRAIN_PATH = DATA_DIR / "churn_train.csv"
HOLDOUT_PATH = DATA_DIR / "churn_holdout.csv"

TARGET = "churned"  # the column every model in this benchmark predicts
ID_COLUMN = "customer_id"
SEED = 20260802

TRAIN_ROWS = 2400
HOLDOUT_ROWS = 600

CONTRACTS = {"month-to-month": 0.55, "one-year": 0.28, "two-year": 0.17}
PAYMENTS = {"credit_card": 0.38, "bank_transfer": 0.27, "electronic_check": 0.23, "mailed_check": 0.12}

# Weights of the logistic model the labels are drawn from — the ground truth a trained model recovers.
WEIGHTS: dict[str, float] = {
    "intercept": -1.15,
    "tenure_months": -0.045,          # the longer they have stayed, the less likely they are to leave
    "monthly_charges_over_65": 0.018,  # price sensitivity, above a $65 anchor
    "support_tickets": 0.22,           # friction with the product
    "avg_session_minutes_over_20": -0.012,  # engagement
    "num_products": -0.28,             # every extra product is another reason to stay
    "is_senior": 0.25,
    "has_premium_support": -0.50,
    "contract:month-to-month": 1.10,   # nothing holding them in
    "contract:one-year": 0.15,
    "contract:two-year": -0.60,
    "payment:electronic_check": 0.55,  # the classic churn signal in this kind of dataset
    "payment:mailed_check": 0.20,
    "payment:credit_card": -0.10,
    "payment:bank_transfer": -0.15,
}

LABEL_NOISE = 0.35  # spread of the noise added to the logit, so the problem isn't perfectly separable

MISSING_TOTAL_CHARGES = 0.03   # share of rows whose `total_charges` never made it out of billing
MISSING_SESSION_MINUTES = 0.02
OUTLIER_CHARGES = 0.01         # share of rows with a billing-system spike in `monthly_charges`
DUPLICATE_ROWS = 15            # rows repeated verbatim in the training file

COLUMNS = [
    ID_COLUMN,
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "contract_type",
    "payment_method",
    "support_tickets",
    "avg_session_minutes",
    "num_products",
    "has_premium_support",
    "is_senior",
    TARGET,
]


def pick(rng: random.Random, weighted: dict[str, float]) -> str:
    """Draw one category from a `{value: probability}` mapping."""
    return rng.choices(list(weighted), weights=list(weighted.values()))[0]


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def churn_logit(row: dict[str, object]) -> float:
    """Apply `WEIGHTS` to one customer, returning the log-odds of them churning."""
    logit = WEIGHTS["intercept"]
    logit += WEIGHTS["tenure_months"] * float(row["tenure_months"])
    logit += WEIGHTS["monthly_charges_over_65"] * (float(row["monthly_charges"]) - 65.0)
    logit += WEIGHTS["support_tickets"] * float(row["support_tickets"])
    logit += WEIGHTS["avg_session_minutes_over_20"] * (float(row["avg_session_minutes"]) - 20.0)
    logit += WEIGHTS["num_products"] * float(row["num_products"])
    logit += WEIGHTS["is_senior"] * float(row["is_senior"])
    logit += WEIGHTS["has_premium_support"] * (1.0 if row["has_premium_support"] == "yes" else 0.0)
    logit += WEIGHTS[f"contract:{row['contract_type']}"]
    logit += WEIGHTS[f"payment:{row['payment_method']}"]
    return logit


def customer(rng: random.Random, number: int) -> dict[str, object]:
    """Draw one customer, label included."""
    contract = pick(rng, CONTRACTS)

    # Customers on longer contracts have usually been around longer, which is what makes tenure and
    # contract type correlated here — the kind of collinearity a feature-selection step has to handle.
    tenure_ceiling = {"month-to-month": 40, "one-year": 58, "two-year": 72}[contract]
    tenure = rng.randint(1, tenure_ceiling)

    products = rng.choices([1, 2, 3, 4], weights=[0.44, 0.32, 0.17, 0.07])[0]
    monthly = round(rng.gauss(48 + 14 * products, 12), 2)
    monthly = max(19.5, min(140.0, monthly))

    row: dict[str, object] = {
        ID_COLUMN: f"C{number:06d}",
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "total_charges": round(monthly * tenure * rng.uniform(0.94, 1.04), 2),
        "contract_type": contract,
        "payment_method": pick(rng, PAYMENTS),
        "support_tickets": min(12, int(rng.expovariate(1 / 1.6))),
        "avg_session_minutes": round(max(1.5, rng.gauss(26 - 1.6 * rng.random() * 6, 9)), 1),
        "num_products": products,
        "has_premium_support": "yes" if rng.random() < 0.31 else "no",
        "is_senior": 1 if rng.random() < 0.16 else 0,
    }

    probability = sigmoid(churn_logit(row) + rng.gauss(0, LABEL_NOISE))
    row[TARGET] = 1 if rng.random() < probability else 0
    return row


def dirty(rng: random.Random, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Put the mess back in: missing values, billing outliers, and repeated rows."""
    for row in rows:
        if rng.random() < MISSING_TOTAL_CHARGES:
            row["total_charges"] = ''
        if rng.random() < MISSING_SESSION_MINUTES:
            row["avg_session_minutes"] = ''
        if rng.random() < OUTLIER_CHARGES:
            row["monthly_charges"] = round(float(row["monthly_charges"]) * rng.uniform(5.0, 9.0), 2)

    return rows


def write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def generate() -> dict[str, object]:
    """Write the train and holdout CSVs, and return what was written."""
    rng = random.Random(SEED)
    rows = [customer(rng, number) for number in range(TRAIN_ROWS + HOLDOUT_ROWS)]

    train = dirty(rng, rows[:TRAIN_ROWS])
    train += [dict(row) for row in rng.sample(train, DUPLICATE_ROWS)]  # the duplicates cleaning finds
    holdout = rows[TRAIN_ROWS:]  # left clean, so evaluation measures the model rather than the mess

    write(TRAIN_PATH, train)
    write(HOLDOUT_PATH, holdout)

    churn_rate = sum(int(row[TARGET]) for row in train) / len(train)
    return {
        # repo-relative, so what this prints is the same on every machine
        "train_path": f"./{TRAIN_PATH.resolve().relative_to(REPO_ROOT)}",
        "holdout_path": f"./{HOLDOUT_PATH.resolve().relative_to(REPO_ROOT)}",
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "churn_rate": round(churn_rate, 4),
        "columns": COLUMNS,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(generate(), indent=2))
