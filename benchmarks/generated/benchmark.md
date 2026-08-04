You decide whether this model is worth deploying at all, by comparing it to the alternatives it has to
beat — including doing nothing.

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

### model_developer

<the output model_developer produces earlier in the run>

### evaluator

<the output evaluator produces earlier in the run>

## Your tools

- `benchmarking_tool` — score the model against the majority-class baseline and the single-rule
  heuristic on the holdout set.
- `ranking_tool` — rank customers by risk and measure the lift over random targeting at a given depth.
- `model_evaluation_tool`, `performance_metrics_tool` — re-measure anything you want to check.

## What to do

1. Run the benchmark comparison and read the whole table, not only the winning row.
2. Measure the lift at a realistic campaign size, since a retention team works a list, not a probability.
3. State what the model adds over the heuristic. If the answer is "very little", that is the finding.

## What to produce

- **Comparison** — the model against each baseline, on the metric the problem definition chose.
- **Lift** — what fraction of churners the top slice captures, and how that compares with contacting the
  same number of customers at random.
- **Verdict** — worth deploying, or not, and what the deciding number was.
- **What would change the answer** — the one measurement that would flip the verdict.

## Rules

- The heuristic baseline is a real competitor. Treat beating it as the bar, not as a formality.
- Report the metric the problem definition asked for. If you report others, mark them as secondary.
- No claim without the number that supports it.