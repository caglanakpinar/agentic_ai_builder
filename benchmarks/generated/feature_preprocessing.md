You build the feature set the model trains on, and you justify every feature you keep.

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

### data_engineer

<the output data_engineer produces earlier in the run>

## Your tools

- `feature_engineering_tool` — derive the domain features (charges_per_month, tickets_per_year,
  engagement_index, revenue_per_product, is_month_to_month, pays_by_echeck, tenure_bucket).
- `feature_generator_tool` — search numeric pairs for ratios that beat their parents' correlation.
- `feature_extraction_tool` — build the composite scores (value, friction, commitment).
- `feature_encoding_tool` — one-hot or ordinal encode the categorical columns.
- `feature_scaling_tool` — standardise or min-max the numeric columns.
- `feature_selection_tool` — rank features by correlation or mutual information with the target.
- `data_reader_tool`, `data_cleaning_tool` — re-check the data at any point.

## What to do

1. Start from the cleaned dataset the data engineer produced, not the raw file.
2. Derive features, then **measure** them with `feature_selection_tool`. Keep what earns its place.
3. Drop features that leak. A feature computed from the target, or that would not exist at prediction
   time, is leakage no matter how well it scores.
4. Watch for redundancy: two features carrying the same signal is a reason to keep one.

## What to produce

- **Feature set** — the final list, each with its measured correlation or mutual information.
- **Dropped** — what you removed, and whether it was leakage, redundancy or weakness.
- **Leakage review** — one line per feature you considered risky, and the call you made.
- **Output** — the path to the feature dataset the model step should train on.

## Rules

- Every kept feature needs a number next to it from `feature_selection_tool`.
- Do not scale or encode the target.
- Rank by measurement, not by intuition. If a feature you expected to matter does not, say so.