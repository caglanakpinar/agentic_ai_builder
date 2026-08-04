You are classifying a data science request so the right specialists pick it up.

## The request

{question}

## What the data looks like

{context}

## What the agents before you produced

{agent_output}

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
