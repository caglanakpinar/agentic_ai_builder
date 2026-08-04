You are classifying a data science request so the right specialists pick it up.

## The request

Our subscription business is losing customers and we don't know which ones to work on. Using the customer table in benchmarks/data/churn_train.csv, build something that tells the retention team who is about to churn, and tell us whether it is good enough to act on. The team can contact about 50 customers a week.

## What the data looks like

Dataset: benchmarks/data/churn_train.csv (training), benchmarks/data/churn_holdout.csv (holdout)

Shape: 2415 rows x 12 columns, one row per customer.

Target: `churned` — 1 if the customer cancelled in the following quarter.
  positive: 444 rows (18.4%)
  negative: 1971 rows

Columns:
  customer_id          identifier, not a feature
  tenure_months        numeric, months since signup
  monthly_charges      numeric, current monthly bill
  total_charges        numeric, billed to date  (has missing values)
  contract_type        categorical: month-to-month, one-year, two-year
  payment_method       categorical: credit_card, bank_transfer, electronic_check, mailed_check
  support_tickets      numeric, tickets raised
  avg_session_minutes  numeric, mean session length  (has missing values)
  num_products         numeric, products held
  has_premium_support  categorical: yes, no
  is_senior            binary flag
  churned              the target

Known defects (measured, not assumed):
  validation checks failing: unique_ids
  duplicate rows: 15
  columns with missing values: avg_session_minutes, total_charges
  columns with outliers beyond 1.5x IQR: avg_session_minutes, monthly_charges, total_charges

Strongest measured relationships with the target:
  - tenure_months: correlation -0.252, mutual information 0.0331
  - tenure_bucket=established: correlation -0.220, mutual information 0.0250
  - contract_type=month-to-month: correlation +0.218, mutual information 0.0253
  - is_month_to_month: correlation +0.218, mutual information 0.0253
  - total_charges: correlation -0.215, mutual information 0.0236
  - tickets_per_year: correlation +0.190, mutual information 0.0262
  - support_tickets: correlation +0.170, mutual information 0.0113
  - contract_type=two-year: correlation -0.153, mutual information 0.0151

Operational constraint: the retention team can contact about 50 customers a week, so what matters is
who is at the top of the list, not only how well the model separates overall.

## What the agents before you produced

### rag_problem_thinker_agent

<the output rag_problem_thinker_agent produces earlier in the run>

## What to produce

Answer with exactly these six lines, and nothing else:

```
problem_type: <binary_classification | multiclass_classification | regression | ranking | clustering | forecasting | anomaly_detection>
target: <the column being predicted>
learning: <supervised | unsupervised | semi_supervised>
imbalance: <balanced | mild | severe>   # severe when the positive class is under 10%
primary_metric: <roc_auc | pr_auc | f1 | recall | precision | rmse | mae>
confidence: <high | medium | low>
```

## Rules

- One value per line, no explanation, no code fences in your answer.
- `problem_type` must follow from the target's values in the profile, not from the wording of the
  request.
- `imbalance` must follow from the positive rate given in the profile. State it from the number, not
  from the impression.
- If the profile and the problem definition disagree, follow the profile — it is measured, the
  definition is argued.
- Answer `confidence: low` rather than guessing when the profile does not contain what you need.