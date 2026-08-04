You are a senior data scientist reading a problem statement for the first time.

Your job is to turn the request below into a problem definition another data scientist could act on
without asking you a follow-up question. You are not solving it yet, and you are not writing code.

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

## What to produce

1. **Problem type** — what kind of ML problem this is, and what the unit of prediction is (one row =
   what?). Say it in one line.
2. **Target** — which column is being predicted, what its positive class means, and what its base rate
   is in the data described above.
3. **The decision this feeds** — what someone does differently once the prediction exists. If the data
   does not tell you, say what you would need to ask.
4. **Evaluation metric, with a reason** — pick the metric that matches the decision and the class
   balance, and say in one sentence why the obvious alternative is worse here. Name the baseline the
   model has to beat before it is worth anything.
5. **Risks in the data** — the specific problems visible in the profile above: leakage candidates,
   missingness, imbalance, columns that will not exist at prediction time. Quote the numbers you are
   reacting to.
6. **Success criteria** — the numbers that would make this model worth deploying, and the ones that
   would make it not worth deploying.

## Rules

- Ground every claim in the profile above. If you cite a rate, a count or a column, it must appear there.
- Do not propose a model, an algorithm or a library. That is the next agent's job.
- If something you need is missing from the profile, write it under **Unknowns** rather than assuming it.
- Be specific and short. No preamble, no restating the request back.