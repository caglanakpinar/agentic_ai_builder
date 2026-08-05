"""Builds the knowledge base `rag_problem_thinker_agent` retrieves from before it frames the problem.

    python benchmarks/knowledge.py               # write both stores
    python benchmarks/knowledge.py --text-only   # just the documents, when no embeddings key is set

Retrieval here is split across two stores, and both have to exist for it to work:

  - the **text db** (`ds_knowledge_text_db`, Chroma under `benchmarks/artifacts/ds_knowledge_text`)
    holds the documents themselves, keyed by id. Writing it needs no API key.
  - the **vector db** (`ds_knowledge_db`, a FAISS index at `benchmarks/artifacts/ds_knowledge.index`)
    holds one embedding per document, and answers a question with the ids of the nearest ones. Writing
    it embeds every document, so it needs the key the `embeddings:` block names.

At run time the agent embeds its question, asks the vector db which documents are closest, and reads
those documents out of the text db. A missing vector index means nothing is ever matched; a missing text
db means ids come back with no documents behind them. Either way the agent answers from the dataset
profile alone, which is why this script exists as a step of its own.

The notes below are reference material — how to frame a churn question, which metric survives an
imbalanced target, what a fixed weekly call budget does to the decision threshold. Deliberately none of
them contains a figure about this particular dataset: every number the agents report has to come from a
tool that measured it, and a knowledge base that carried plausible-looking numbers would be the easiest
way for an unmeasured claim to reach a judge.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ''):  # run as a script rather than `python -m benchmarks.knowledge`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_builder import build_embeddings, build_text_db, build_vector_db, load_configs
from benchmarks import tools
from uilts.logger import logger

CONFIG_DIR = Path(__file__).parent


# One note per id. `id` is what the vector search returns and the text db is keyed by, so it is also
# what the agent sees as the heading above each retrieved block — worth keeping readable.
DOCUMENTS: list[dict[str, str]] = [
    {
        "id": "framing-a-retention-request",
        "topic": "problem definition",
        "text": (
            "A retention request ('tell us who is about to leave') becomes a prediction problem only "
            "once three things are fixed: the unit of prediction (usually one row per customer), the "
            "outcome being predicted (cancellation within a stated window), and the moment the "
            "prediction is made. Features must describe the customer as of that moment and no later. "
            "State the window explicitly — 'churns in the next quarter' is a different target from "
            "'has churned', and a model trained on one answers the other badly. If the request does not "
            "name a window, the target column's definition is what settles it."
        ),
    },
    {
        "id": "binary-classification-basics",
        "topic": "problem type",
        "text": (
            "The problem type follows from the target's values, not from the wording of the request. A "
            "column taking two values is binary classification; more than two unordered values is "
            "multiclass; a continuous column is regression. 'Predict who will churn' sounds like "
            "ranking and is usually served as one, but it is trained as binary classification and then "
            "scored by predicted probability. Say which one you mean: the learning problem is "
            "classification, the delivery is a ranked list."
        ),
    },
    {
        "id": "class-imbalance",
        "topic": "imbalance",
        "text": (
            "Imbalance is a property of the measured positive rate, not an impression. A common "
            "convention: above roughly 40% positives is balanced, between 10% and 40% is mild, and "
            "under 10% is severe. Read the rate off the data profile before calling it. Imbalance "
            "matters because it decides which metrics are meaningful, whether class weighting or "
            "resampling is worth it, and how a decision threshold should be picked. A severe imbalance "
            "also shrinks the effective sample: the number of positive rows, not the number of rows, is "
            "what limits how much a model can learn."
        ),
    },
    {
        "id": "metric-choice-on-imbalanced-targets",
        "topic": "evaluation",
        "text": (
            "Accuracy is the wrong primary metric on an imbalanced target: a model predicting the "
            "majority class for everyone scores exactly the negative rate and is worth nothing. Report "
            "accuracy only beside the majority-class baseline, so the comparison is visible. ROC AUC "
            "measures separation across all thresholds and is stable under imbalance, which makes it a "
            "good model-selection metric. PR AUC (average precision) is more sensitive to performance "
            "on the positive class and is the better headline when positives are rare. F1 balances "
            "precision and recall at one chosen threshold, so it only means something once the "
            "threshold has been chosen deliberately. Recall matters when a missed positive is expensive; "
            "precision matters when acting on a false positive is."
        ),
    },
    {
        "id": "capacity-constrained-targeting",
        "topic": "delivery",
        "text": (
            "When the team can only act on a fixed number of customers per cycle, the model is a "
            "ranking device and the metrics that matter are measured at that depth. Precision@k is the "
            "share of the k contacted who really churn — the hit rate of the call list. Capture rate "
            "(recall@k) is the share of all churners the list caught. Lift@k is precision@k divided by "
            "the base rate: how many times better the list is than calling at random, and the number "
            "that answers 'is this worth doing'. A lift near 1 means the model adds nothing at that "
            "depth, however good its overall AUC looks. Report all three against the stated capacity, "
            "not against a round number."
        ),
    },
    {
        "id": "decision-threshold",
        "topic": "delivery",
        "text": (
            "0.5 is a default, not a decision. The threshold should come from something outside the "
            "model: the capacity available (take the top k and let the threshold be whatever score "
            "lands there), or the relative cost of a wasted contact against a missed churner, or a "
            "sweep that maximises F1 when neither is known. Whichever is chosen, say why. On an "
            "imbalanced target a class-weighted model's probabilities are shifted towards the positive "
            "class, so a threshold carried over from an unweighted model will not mean what it used to."
        ),
    },
    {
        "id": "leakage",
        "topic": "data quality",
        "text": (
            "Leakage is any feature that could only be known after the outcome. In subscription data "
            "the usual sources are cumulative totals that keep accruing past the prediction moment, "
            "cancellation or refund fields, support activity logged during the cancellation itself, and "
            "identifiers that encode signup cohort. The symptom is a score that looks too good, or a "
            "coefficient whose sign contradicts the domain. An identifier column is never a feature. "
            "When a model performs implausibly well, suspect leakage before celebrating."
        ),
    },
    {
        "id": "validation-discipline",
        "topic": "evaluation",
        "text": (
            "A score measured on the data the model was fitted to is not evidence it generalises, and "
            "quoting one as though it were is the most common failure in a modelling write-up. Use "
            "stratified k-fold cross-validation for model selection — stratified so every fold keeps "
            "the positive rate — and report the mean with its spread across folds. The spread is what "
            "makes a comparison honest: a difference between two models smaller than the fold-to-fold "
            "standard deviation is noise, and picking the more complex model on that basis is not "
            "justified. Keep a holdout set that is touched once, at the end, and label every number "
            "with the dataset it came from."
        ),
    },
    {
        "id": "overfitting-gap",
        "topic": "evaluation",
        "text": (
            "The distance between the training score and the held-out score is the overfitting check. A "
            "small gap means the model learned signal; a large one means it learned the training set. "
            "What counts as large depends on the metric and the sample size, but a gap wider than the "
            "cross-validation spread deserves an explanation. Regularisation, fewer features, or a "
            "simpler model are the usual answers — not a larger training run."
        ),
    },
    {
        "id": "baselines",
        "topic": "benchmarking",
        "text": (
            "A model is only worth deploying if it beats what the business could do without it. Two "
            "baselines are always available: predicting the majority class, which fixes the accuracy "
            "floor, and the single-rule heuristic someone is already using — in subscriptions, usually "
            "'everyone on a month-to-month contract'. A single rule is often a surprisingly strong "
            "baseline, and beating it by a margin smaller than the fold-to-fold spread is not beating "
            "it. Report the model and the baselines on the same dataset, at the same threshold, with "
            "the same metric."
        ),
    },
    {
        "id": "cleaning-numeric-columns",
        "topic": "data quality",
        "text": (
            "Impute missing numeric values with the median rather than the mean when the column is "
            "skewed, and record how many values were filled rather than silently filling them. Clipping "
            "outliers at 1.5x IQR is a reasonable default for continuous measurements, but it is wrong "
            "for two kinds of column: a binary or near-constant flag has an IQR of zero, so every value "
            "outside the majority gets clipped away, and a small integer count gets its tail flattened "
            "into a fraction. Skip both, and say which columns were skipped. Duplicate rows should be "
            "removed before any split, otherwise the same customer appears on both sides of it."
        ),
    },
    {
        "id": "subscription-features",
        "topic": "feature engineering",
        "text": (
            "Features that usually carry signal in subscription churn: tenure (short tenure churns "
            "most, and bucketing it captures the non-linearity a linear model would miss), contract "
            "type as an explicit flag for the shortest commitment, payment method — manual methods "
            "correlate with churn more than the amount does — support contact rate normalised by "
            "tenure rather than raw counts, spend per product or per month rather than lifetime totals, "
            "and an engagement measure relative to the customer's own history. Ratios of two columns "
            "often beat both parents; check that a derived feature actually correlates better before "
            "keeping it."
        ),
    },
    {
        "id": "reading-coefficients",
        "topic": "interpretation",
        "text": (
            "In a linear model the coefficients are readable, but only after scaling — an unscaled "
            "coefficient's size mostly reflects its column's units. Check the direction against the "
            "domain: longer tenure should lower churn risk, more support tickets should raise it. A "
            "reversed sign is usually collinearity between two features carrying the same information, "
            "or leakage, and is a finding rather than an insight. Strength of a driver is not evidence "
            "of causation, and a retention action justified by a coefficient needs an experiment."
        ),
    },
    {
        "id": "reporting-results",
        "topic": "reporting",
        "text": (
            "Every performance figure needs three labels: which dataset it was measured on, at which "
            "threshold, and by which metric. Without them a reader cannot tell a cross-validated score "
            "from a training one. Quote the primary metric first, then the operating-point metrics at "
            "the threshold that will actually be used, then the baselines. State what would make the "
            "result wrong — the assumption most likely to fail — rather than only what it shows. A "
            "claim with no measurement behind it should be marked as an assumption, not written as a "
            "result."
        ),
    },
]


def documents() -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Return the knowledge base as the `(ids, texts, metadatas)` both stores are written with."""
    return (
        [document["id"] for document in DOCUMENTS],
        [document["text"] for document in DOCUMENTS],
        [{"topic": document["topic"], "id": document["id"]} for document in DOCUMENTS],
    )


def retrieval_names(configs) -> tuple[str | None, str | None, str | None]:
    """Read which vector db, text db and embeddings caller the agents retrieve through.

    Taken from the config rather than hardcoded, so this script fills whatever the YAML points the
    agents at — renaming a db there does not leave the knowledge base behind under the old name.
    """
    for config in configs.agent_configs.values():
        if config.db_vector or config.db_text:
            return config.db_vector, config.db_text, config.embedding

    return None, None, None


def fill_text_db(configs, name: str) -> Any:
    """Write every document into the text db, and return the connector."""
    connector = build_text_db(name, configs)
    ids, texts, metadatas = documents()
    connector.upsert(ids=ids, documents=texts, metadatas=metadatas)
    print(f"  text db    {name:<22} {connector.count()} document(s) at {tools.shown(connector.path)}")
    return connector


def fill_vector_db(configs, name: str, embeddings_name: str) -> Any:
    """Embed every document and write the vectors into the vector db, and return the connector.

    This is the half that needs a key: one embedding call per document. The index is saved to disk
    afterwards when the engine keeps its own file — FAISS holds the index in memory and would otherwise
    lose it when the process ends.
    """
    embeddings = build_embeddings(embeddings_name, configs)
    ids, texts, metadatas = documents()

    print(f"  embedding  {len(texts)} document(s) with {embeddings.model_name} ...")
    vectors = embeddings.embed_texts(texts)
    width = len(vectors[0]) if vectors else 0

    connector = build_vector_db(name, configs)
    if connector.dimension and width and connector.dimension != width:
        raise ValueError(
            f"{name}: configured for {connector.dimension}-wide vectors but {embeddings.model_name} "
            f"returned {width}. Set `dimension: {width}` on the db, or ask the model for "
            f"{connector.dimension} dimensions."
        )

    # The documents are written into the vector db as well: FAISS keeps them in its sidecar, so a
    # retrieval still has text behind its ids if the text store is ever unavailable.
    connector.upsert(ids=ids, vectors=vectors, metadatas=metadatas, documents=texts)
    if hasattr(connector, "save"):
        connector.save()

    print(f"  vector db  {name:<22} {connector.count()} vector(s) of width {width} "
          f"at {tools.shown(connector.path)}")
    return connector


def build(config_dir: str | Path = CONFIG_DIR, text_only: bool = False) -> None:
    """Fill both stores from the config's `dbs:` block, reporting what each one now holds."""
    configs = load_configs(str(config_dir))
    vector_name, text_name, embeddings_name = retrieval_names(configs)

    print(f"\n{'=' * 96}\nKNOWLEDGE BASE — {len(DOCUMENTS)} documents, from {tools.shown(config_dir)}"
          f"\n{'=' * 96}")

    if text_name:
        fill_text_db(configs, text_name)
    else:
        logger.warning("No agent names a `db_text`, so there is no text db to write the documents into.")

    if text_only:
        print("\n  Vector index skipped (--text-only): nothing was embedded, so retrieval will match "
              "nothing until it is built.")
        return

    if not vector_name:
        logger.warning("No agent names a `db_vector`, so there is no index to embed the documents into.")
        return

    if not embeddings_name:
        logger.warning(
            f"No agent names an `embedding`, so there is nothing to embed the documents for "
            f"{vector_name} with. Add `embedding:` to the agent that retrieves."
        )
        return

    fill_vector_db(configs, vector_name, embeddings_name)
    print("\n  Both stores written — the retrieving agent will find documents on its next run.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config-dir", default=None,
                        help=f"Directory holding the benchmark YAML. [default: {tools.shown(CONFIG_DIR)}]")
    parser.add_argument("--text-only", action="store_true",
                        help="Write the documents but skip the embeddings, for when no key is set.")
    arguments = parser.parse_args()

    build(arguments.config_dir or CONFIG_DIR, text_only=arguments.text_only)


if __name__ == "__main__":
    main()
