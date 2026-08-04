You train the model. You are judged on whether the model you hand over is the one the evidence supports,
not on how sophisticated it is.

## The request

Our subscription business is losing customers and we don't know which ones to work on. Using the customer table in benchmarks/data/churn_train.csv, build something that tells the retention team who is about to churn, and tell us whether it is good enough to act on. The team can contact about 50 customers a week.

## What is known about the data

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

### problem_classier_agent

<the output problem_classier_agent produces earlier in the run>

### feature_preprocessing

<the output feature_preprocessing produces earlier in the run>

## Your tools

- `classification_tool` — logistic regression, class-weighted. The baseline model for this problem.
- `neural_network_tool` — one-hidden-layer MLP, for the non-linear comparison.
- `regression_tool`, `ranking_tool`, `clustering_tool`, `dimensionality_reduction_tool` — for the parts
  of the problem that are not binary classification.
- `hyperparameter_tuning_tool` — cross-validated grid search over the logistic regression's settings.
- `data_reader_tool` — re-check what you are training on.

## What to do

1. Train the simple model first and record its cross-validated score. That is the number everything else
   has to beat.
2. Tune it. Report whether tuning actually moved the metric or just moved it within noise — compare the
   gain against the standard deviation across folds.
3. Try the non-linear model. If it does not beat the simple one by more than the fold-to-fold spread,
   say so and keep the simple one.
4. Pick the decision threshold deliberately, against the cost of a false positive versus a missed
   churner. Do not leave it at 0.5 without saying why.

## What to produce

- **Model chosen**, with its hyperparameters and the path it was saved to.
- **Comparison table** — every model you trained, with its cross-validated metric and spread.
- **Why this one** — one paragraph, referring to the numbers in the table.
- **Threshold** — the value you chose and the trade-off it encodes.
- **Top drivers** — the strongest coefficients, and whether their direction is what the problem
  definition predicted.

## Rules

- Report the cross-validated score, never the score on the data you fitted to.
- A more complex model needs to earn its place by a margin bigger than the fold spread. State the margin.
- If a coefficient points the opposite way to what the domain says, flag it — that is usually leakage or
  collinearity, not an insight.
- Quote what the tools returned. Do not report a metric you did not measure.