"""Runs the benchmark: builds the agentic workflow from the YAML and puts it to work on the churn problem.

    python benchmarks/dataset.py            # write the dataset (once)
    python benchmarks/knowledge.py          # fill the knowledge base the first agent retrieves from (once)
    python benchmarks/run_benchmark.py      # build the workflow, run the tool chain, render the prompts
    python benchmarks/run_benchmark.py --live   # ...and actually call the models

There are two modes because they check different things.

Without `--live` nothing is sent to a provider, and no API key is needed. The workflow is still built for
real — every agent constructed, every tool function imported, every prompt read — and the tool chain is
run end to end over the dataset, so the numbers printed are measured. What each agent *would* be sent is
rendered and written out. This is the mode that can be verified.

With `--live` the pipeline in the YAML is walked step by step: each step's agent runs, its output is
passed to the next by name, and a step that declares `needs_judger` is reviewed before the run moves on.

The tool chain is the ground truth the agents are measured against — it is the same set of calls a
competent data scientist would make on this problem, run without a model in the loop.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ''):  # run as a script rather than `python -m benchmarks.run_benchmark`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_builder import (
    build_agent,
    build_embeddings,
    build_text_db,
    build_vector_db,
    load_configs,
)
from benchmarks import tools
from uilts.configs import ENV_VAR_NAME
from uilts.logger import logger

CONFIG_DIR = Path(__file__).parent
GENERATED_DIR = CONFIG_DIR / "generated"

VERDICT_LINE = re.compile(r"^\s*(verdict|gates_failed|confidence)\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE)


class Transcript:
    """Mirrors everything the run prints into a plain-text log beside the artifacts.

    The terminal transcript of a live run is the record of what happened, but it is not something you
    can read afterwards: the provider SDK's response objects carry kilobytes of base64 thinking
    signature each, and they bury the part that matters. What lands here is text — the section headers,
    each agent's answer, each judge's verdict, the gate arithmetic — plus the warnings, in the order
    they happened. Nothing is reformatted; it is the same structure the run prints.
    """

    def __init__(self, stream: Any, path: Path) -> None:
        self.stream = stream
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, 'w')
        self.path = path

        # Warnings belong in the transcript too, in place — a truncated answer or a failed tool is part
        # of the record, and reading it later out of a separate stream loses the ordering. So does each
        # tool call, which is the run's actual work. Startup chatter (tools binding to their functions,
        # every API response's token count) is not, and would bury both.
        self.handler = logging.StreamHandler(self.file)
        self.handler.setFormatter(logging.Formatter("  [%(levelname)s] %(message)s"))
        self.handler.addFilter(lambda record: (
            record.levelno >= logging.WARNING or record.getMessage().startswith("Calling tool ")
        ))
        logger.addHandler(self.handler)

    def write(self, text: str) -> int:
        self.stream.write(text)
        self.file.write(text)
        return len(text)

    def flush(self) -> None:
        self.stream.flush()
        self.file.flush()

    def close(self) -> None:
        logger.removeHandler(self.handler)
        self.file.close()


def parsed_verdict(output: str) -> dict[str, str]:
    """Pull the `verdict:` / `gates_failed:` / `confidence:` lines out of a judge's answer."""
    return {key.lower(): value.strip() for key, value in VERDICT_LINE.findall(output or '')}

QUESTION = (
    "Our subscription business is losing customers and we don't know which ones to work on. "
    "Using the customer table in benchmarks/data/churn_train.csv, build something that tells the "
    "retention team who is about to churn, and tell us whether it is good enough to act on. "
    "The team can contact about 50 customers a week."
)


def ground_truth() -> dict[str, object]:
    """Run the data science workflow with the tools alone, and return what they measured.

    This is what the agents are supposed to orchestrate, so it doubles as the answer key: profile the
    data, validate it, clean it, build features, select them, train, cross-validate, evaluate on the
    holdout, and compare against the baselines.
    """
    profile = tools.data_reader_caller(limit=3)
    validation = tools.data_validation_caller()
    cleaning = tools.data_cleaning_caller()
    cleaned = tools.data_validation_caller(cleaning["output_path"])  # did the cleaning actually fix it?
    features = tools.feature_engineering_caller(data_source=cleaning["output_path"])
    selection = tools.feature_selection_caller(features["output_path"], top_k=8)
    training = tools.classification_caller(features["output_path"])
    validated = tools.cross_validation_caller(features["output_path"], folds=5)
    evaluation = tools.model_evaluation_caller()
    thresholds = tools.performance_metrics_caller()
    targeting = tools.ranking_caller(top_k=50)
    comparison = tools.benchmarking_caller()

    duplicates = next(
        (check["duplicates"] for check in cleaned["checks"] if check["check"] == "unique_ids"), 0
    )
    missing_rates = [
        column["missing"] / max(1, profile["rows"]) for column in profile["schema"].values()
    ]

    return {
        "rows": profile["rows"],
        "class_balance": profile["class_balance"],
        "validation_failures": validation["failed_checks"],
        "cleaning": {
            "duplicates_removed": cleaning["duplicates_removed"],
            "imputed_columns": sorted(cleaning["imputed"]),
            "clipped_columns": sorted(cleaning["clipped"]),
            "clipping_skipped": cleaning["clipping_skipped"],
        },
        "top_features": selection["scores"][:8],
        "training_metrics": training["training_metrics"],
        "cross_validation": {
            "mean_roc_auc": validated["mean_roc_auc"],
            "std_roc_auc": validated["std_roc_auc"],
            "folds": validated["folds"],
        },
        "holdout_metrics": evaluation["metrics"],
        "majority_class_accuracy": evaluation["baseline_majority_class"],
        "best_f1_threshold": thresholds["best_f1_threshold"],
        "targeting_50": {
            "capture_rate": targeting["capture_rate"],
            "precision_at_k": targeting["precision_at_k"],
            "lift_over_random": targeting["lift_over_random"],
        },
        "baselines": comparison["comparisons"],
        "beats_best_baseline": comparison["beats_best_baseline"],
        "model_path": training["model_path"],
        # The measurements the judges' thresholds are evaluated against. Every key here is the stem of a
        # `min_`/`max_` threshold in the YAML, which is what lets a bar be checked without any bespoke
        # wiring: `min_holdout_f1` is compared against `holdout_f1` below.
        "measurements": {
            "rows": profile["rows"],
            "positive_rate": profile["class_balance"]["positive_rate"],
            "missing_rate": round(max(missing_rates, default=0.0), 4),
            "duplicate_rows": duplicates,
            "rows_after_cleaning": cleaning["rows"],
            "feature_correlation": round(
                max((abs(score["correlation"]) for score in selection["scores"]), default=0.0), 4
            ),
            "cv_roc_auc": validated["mean_roc_auc"],
            "cv_spread": validated["std_roc_auc"],
            "train_holdout_auc_gap": round(
                abs(training["training_metrics"]["roc_auc"] - evaluation["metrics"]["roc_auc"]), 4
            ),
            "holdout_f1": evaluation["metrics"]["f1"],
            "lift_at_50": targeting["lift_over_random"],
            "precision_at_50": targeting["precision_at_k"],
            "f1_gain_over_baseline": comparison["f1_gain_over_best_baseline"],
        },
    }


# Where each gated measurement can be read from an agent's own tool result. This is what closes the
# loop: with the tools actually running, a gate is evaluated against the number the agent produced, not
# against the runner's parallel chain — and any disagreement between the two is itself a finding.
FROM_TOOL_RESULTS: dict[str, tuple[str, Any]] = {
    "rows": ("data_reader_tool", lambda r: r.get("rows")),
    "positive_rate": ("data_reader_tool", lambda r: r["class_balance"]["positive_rate"]),
    "missing_rate": ("data_reader_tool", lambda r: round(
        max((column["missing"] for column in r["schema"].values()), default=0) / max(1, r["rows"]), 4
    )),
    "duplicate_rows": ("data_validation_tool", lambda r: next(
        (check["duplicates"] for check in r["checks"] if check["check"] == "unique_ids"), None
    )),
    "rows_after_cleaning": ("data_cleaning_tool", lambda r: r.get("rows")),
    "feature_correlation": ("feature_selection_tool", lambda r: round(
        max((abs(score["correlation"]) for score in r["scores"]), default=0.0), 4
    )),
    "cv_roc_auc": ("cross_validation_tool", lambda r: r.get("mean_roc_auc")),
    "cv_spread": ("cross_validation_tool", lambda r: r.get("std_roc_auc")),
    "holdout_f1": ("model_evaluation_tool", lambda r: r["metrics"]["f1"]),
    "lift_at_50": ("ranking_tool", lambda r: r.get("lift_over_random")),
    "precision_at_50": ("ranking_tool", lambda r: r.get("precision_at_k")),
    "f1_gain_over_baseline": ("benchmarking_tool", lambda r: r.get("f1_gain_over_best_baseline")),
}


def measurements_from(agents: dict[str, object]) -> dict[str, float]:
    """Read the gated measurements back out of what the agents' tools actually returned.

    Every agent records each call it made and what came back, so the numbers a gate needs are already
    there — no agent has to be trusted to report them correctly in prose. Where an agent called a tool
    more than once the latest result wins, which is the one it worked from.
    """
    produced: dict[str, float] = {}
    scores: dict[str, float] = {}
    for agent in agents.values():
        for call in getattr(agent, "tool_calls", []):
            if call["failed"] or not isinstance(call["result"], dict):
                continue

            # the overfitting gate spans two tools, so its halves are collected as they go by
            if call["tool"] == "classification_tool":
                scores["train"] = call["result"].get("training_metrics", {}).get("roc_auc")
            if call["tool"] == "model_evaluation_tool":
                scores["holdout"] = call["result"].get("metrics", {}).get("roc_auc")

            for stem, (tool, read) in FROM_TOOL_RESULTS.items():
                if call["tool"] != tool:
                    continue
                try:
                    value = read(call["result"])
                except (KeyError, TypeError, IndexError):
                    continue
                if value is not None:
                    produced[stem] = value

    if scores.get("train") is not None and scores.get("holdout") is not None:
        produced["train_holdout_auc_gap"] = round(abs(scores["train"] - scores["holdout"]), 4)

    return produced


def check_thresholds(
    name: str,
    thresholds: dict[str, object],
    measurements: dict[str, float],
    sources: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    """Evaluate one judge's thresholds against what the tools measured.

    The threshold's name carries everything needed: `min_`/`max_` is the direction and the rest is the
    measurement to compare against, so a new bar in the YAML is checked here the moment its stem exists
    in `measurements` — no per-threshold code. A bar naming a measurement that doesn't exist is reported
    as unmeasured rather than quietly passing.
    """
    results = []
    for threshold, bound in (thresholds or {}).items():
        stem = threshold.split("_", 1)[1] if threshold.startswith(("min_", "max_")) else threshold
        measured = measurements.get(stem)
        if measured is None:
            results.append({
                "judge": name, "threshold": threshold, "required": bound, "measured": None,
                "source": "unmeasured", "passed": False, "note": f"no measurement named {stem!r}",
            })
            continue

        passed = measured >= bound if threshold.startswith("min_") else measured <= bound
        results.append({
            "judge": name, "threshold": threshold, "required": bound, "measured": measured,
            "source": (sources or {}).get(stem, "tools"), "passed": bool(passed),
        })

    return results


def gate_report(
    configs,
    measurements: dict[str, float],
    sources: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    """Run every judge's thresholds over the measurements, in the order the pipeline gates them.

    The final judge's `max_failed_gates` is settled last, against the count of everything that failed
    before it — so the arbiter's own bar is measured, not asserted.
    """
    judges = [
        name for name, config in configs.agent_configs.items()
        if config.type == "judger" and config.thresholds
    ]
    order = {judger: position for position, judger in enumerate(pipeline_judges(configs))}
    judges.sort(key=lambda name: order.get(name, len(order)))

    measurements = dict(measurements)
    results: list[dict[str, object]] = []
    for judge in judges:
        measurements["failed_gates"] = sum(1 for result in results if not result["passed"])
        results.extend(
            check_thresholds(judge, configs.agent_configs[judge].thresholds, measurements, sources)
        )

    return results


def reproduction_report(agents: dict[str, object], baseline: dict[str, float]) -> dict[str, object]:
    """Compare what the agents measured against the reference chain — the benchmark's headline score.

    This is only answerable because the agents run their tools for real: each side computed the same
    quantity independently, so a difference means the agents did something different — trained on the
    raw file instead of the cleaned one, validated before cleaning rather than after, skipped a step. A
    measurement the agents never produced is its own result: nobody measured it.
    """
    produced = measurements_from(agents)
    rows = []
    for stem, reference in sorted(baseline.items()):
        agent_value = produced.get(stem)
        rows.append({
            "measurement": stem,
            "reference": reference,
            "agents": agent_value,
            "matches": agent_value is not None and abs(float(agent_value) - float(reference)) < 1e-9,
        })

    calls = [call for agent in agents.values() for call in getattr(agent, "tool_calls", [])]
    return {
        "tool_calls": len(calls),
        "tools_failed": sum(1 for call in calls if call["failed"]),
        "measurements_reproduced": sum(1 for row in rows if row["matches"]),
        "measurements_produced": sum(1 for row in rows if row["agents"] is not None),
        "measurements_total": len(rows),
        "rows": rows,
        "calls": [
            {"agent": call["agent"], "tool": call["tool"], "arguments": call["arguments"],
             "failed": call["failed"]}
            for call in calls
        ],
    }


def run_record(
    configs,
    agents: dict[str, object],
    outputs: dict[str, str],
    measured: dict[str, object],
    gates: list[dict[str, object]],
) -> dict[str, object]:
    """Assemble the whole run as one structured record, alongside the plain-text transcript.

    The log is for reading; this is for checking. Every agent appears with the tools it ran and the
    answer it gave, every judge with its parsed verdict next to the arithmetic on its own thresholds, so
    a run can be compared against another one without re-reading either transcript.
    """
    judges = {name for name, config in configs.agent_configs.items() if config.type == "judger"}
    record: dict[str, object] = {
        "question": QUESTION,
        "measurements": measured.get("measurements") if measured else {},
        "ground_truth": {
            key: measured.get(key) for key in ("cross_validation", "holdout_metrics", "targeting_50")
        } if measured else {},
        "agents": [],
        "gates": gates,
    }

    for name, agent in agents.items():
        output = outputs.get(name)
        if output is None and not getattr(agent, "tool_calls", []):
            continue  # never reached in this run

        entry: dict[str, object] = {
            "agent": name,
            "type": getattr(agent, "type", None),
            "class": type(agent).__name__,
            "llm": f"{type(agent.llm).__name__}({agent.llm.model_name})",
            "depends_on": dependencies_of(configs, name),
            "tool_calls": [
                {"tool": call["tool"], "arguments": call["arguments"], "failed": call["failed"]}
                for call in getattr(agent, "tool_calls", [])
            ],
            "output": output,
        }
        if name in judges:
            entry["verdict"] = parsed_verdict(output)
            entry["thresholds"] = [gate for gate in gates if gate["judge"] == name]

        record["agents"].append(entry)

    record["summary"] = {
        "agents_run": len(record["agents"]),
        "tool_calls": sum(len(entry["tool_calls"]) for entry in record["agents"]),
        "tools_failed": sum(
            1 for entry in record["agents"] for call in entry["tool_calls"] if call["failed"]
        ),
        "gates_passed": sum(1 for gate in gates if gate["passed"]),
        "gates_total": len(gates),
        "verdicts": {
            entry["agent"]: entry.get("verdict", {}).get("verdict")
            for entry in record["agents"] if entry["agent"] in judges
        },
    }
    return record


def gates_now(configs, agents: dict[str, object], baseline: dict[str, float]) -> list[dict[str, object]]:
    """Evaluate every gate against what the agents have produced so far, falling back to the tool chain.

    Called before each judge runs, so a gate reflects the numbers the agents had actually measured by
    that point. A measurement no agent has produced yet falls back to the runner's own chain and is
    labelled `tools`, so a judge can see which of its bars the work in front of it earned.
    """
    produced = measurements_from(agents)
    merged = {**baseline, **produced}
    sources = {stem: ("agent" if stem in produced else "tools") for stem in merged}
    return gate_report(configs, merged, sources)


def dependencies_of(configs, name: str) -> list[str]:
    """The agents whose output this one works from, as declared by `dependency_agent` in the config."""
    declared = getattr(configs.agent_configs.get(name), "dependency_agent", None)
    return [declared] if isinstance(declared, str) else list(declared or [])


def pipeline_judges(configs) -> list[str]:
    """The judges the pipeline gates on, in the order its steps reach them."""
    ordered = []
    for pipeline in getattr(configs, "pipeline", None) or []:
        for step in pipeline.get("steps", []):
            for judge in step.get("needs_judger", []) or []:
                if judge not in ordered:
                    ordered.append(judge)
            # A judge can also be a step's own processor or next_step — the final arbiter is a step of
            # its own, so that it runs after the gates it reviews.
            for processor in (step.get("processor"), step.get("next_step")):
                name = (processor or {}).get("name")
                if (processor or {}).get("type") == "agent" and name not in ordered:
                    if getattr(configs.agent_configs.get(name), "type", None) == "judger":
                        ordered.append(name)

    return ordered


def data_context(measured: dict[str, object]) -> str:
    """Turn what the tools measured into the `{context}` every prompt is rendered with.

    This is the step that grounds the workflow in the dataset: the agents are not told about churn in the
    abstract, they are told this dataset's shape, balance, defects and measured feature strengths.
    """
    balance = measured["class_balance"]
    features = "\n".join(
        f"  - {score['feature']}: correlation {score['correlation']:+.3f}, "
        f"mutual information {score['mutual_information']:.4f}"
        for score in measured["top_features"]
    )
    return f"""Dataset: benchmarks/data/churn_train.csv (training), benchmarks/data/churn_holdout.csv (holdout)

Shape: {measured['rows']} rows x 12 columns, one row per customer.

Target: `churned` — 1 if the customer cancelled in the following quarter.
  positive: {balance['positive']} rows ({balance['positive_rate']:.1%})
  negative: {balance['negative']} rows

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
  validation checks failing: {', '.join(measured['validation_failures']) or 'none'}
  duplicate rows: {measured['cleaning']['duplicates_removed']}
  columns with missing values: {', '.join(measured['cleaning']['imputed_columns']) or 'none'}
  columns with outliers beyond 1.5x IQR: {', '.join(measured['cleaning']['clipped_columns']) or 'none'}

Strongest measured relationships with the target:
{features}

Operational constraint: the retention team can contact about 50 customers a week, so what matters is
who is at the top of the list, not only how well the model separates overall."""


def stand_in_for_missing_keys(configs) -> list[str]:
    """Put a placeholder behind any API key the config names but the environment doesn't hold.

    Constructing an agent constructs its provider client, which needs a key to exist — but a dry run
    never calls one. Rather than demand real credentials to inspect a workflow, the unset variables get a
    placeholder and are reported. Keys already in the environment are left alone, and `--live` never
    comes through here, so a real run still fails loudly on a key that isn't set.
    """
    stood_in = []
    for config in (*configs.llm_configs.values(), *configs.embeddings_configs.values()):
        name = config.api_key
        if name and ENV_VAR_NAME.match(name) and not os.getenv(name):
            os.environ[name] = "dry-run-placeholder-no-calls-are-made"
            stood_in.append(name)

    return sorted(set(stood_in))


def retrieval_connectors(configs) -> dict[str, dict[str, object]]:
    """Connect the embeddings caller and the two dbs each retrieving agent needs, before any is built.

    Retrieval here is a three-part chain — embed the question, ask the vector db which documents are
    nearest, read those documents out of the text db — and a missing part is silent: the agent answers
    from the context it was given, and the run looks the same as one where retrieval worked. So each
    part is connected here, up front, where a failure can be reported against the agent that needed it.

    Each db is connected once and shared by every agent naming it. Chroma and FAISS are embedded engines
    holding a directory, and opening the same one per agent is at best wasted work — Chroma's persistent
    client will not open a store twice in a process at all.
    """
    connectors: dict[tuple[str, str], object] = {}

    def connect(kind: str, build, name: str | None) -> object | None:
        if not name:
            return None
        if (kind, name) not in connectors:
            try:
                connectors[(kind, name)] = build(name, configs)
            except Exception as error:
                logger.warning(f"Could not connect {kind} {name!r}: {error}")
                connectors[(kind, name)] = None
        return connectors[(kind, name)]

    wiring: dict[str, dict[str, object]] = {}
    for name, config in configs.agent_configs.items():
        if not (config.db_vector or config.db_text or config.embedding):
            continue

        wiring[name] = {
            "db_vector_connector": connect("vector db", build_vector_db, config.db_vector),
            "db_text_connector": connect("text db", build_text_db, config.db_text),
            "embeddings_connector": connect("embeddings", build_embeddings, config.embedding),
        }

    return wiring


def workflow(configs) -> dict[str, object]:
    """Build every agent the pipeline references, with any retrieval connectors it needs already open."""
    wiring = retrieval_connectors(configs)
    return {
        name: build_agent(name, configs, **wiring.get(name, {}))
        for name in configs.agent_configs
    }


def retrieval_report(agent: object) -> str:
    """One line saying what an agent retrieves through, and whether there is anything there to find."""
    if not any((
        getattr(agent, "db_vector_connector", None),
        getattr(agent, "db_text_connector", None),
        getattr(agent, "embeddings_connector", None),
    )):
        return ''

    vector = getattr(agent, "db_vector_connector", None)
    text = getattr(agent, "db_text_connector", None)
    embeddings = getattr(agent, "embeddings_connector", None)

    def holding(connector: object, label: str) -> str:
        if not connector:
            return f"{label}: (not connected)"
        try:
            return f"{label}: {connector.name} [{connector.count()}]"
        except Exception as error:  # a store that connected but cannot be read is worth naming too
            return f"{label}: {connector.name} (unreadable: {error})"

    embedder = f"{embeddings.model_name}" if embeddings else "(no embeddings)"
    ready = "" if getattr(agent, "retrieves", lambda: False)() else "  — retrieval off, context only"
    return (f"  {'':<28} retrieves:  {holding(vector, 'vectors')}, {holding(text, 'documents')}"
            f", via {embedder}{ready}")


def render(agents: dict[str, object], context: str, outputs: dict[str, str] | None = None) -> dict[str, str]:
    """Render each agent's prompt as it would be sent, with the dataset context and the prior outputs.

    Nothing has run yet in a dry run, so each agent an agent depends on is stood in for by a placeholder —
    that way the rendered prompt shows the whole shape of what gets sent, including which earlier outputs
    land where.
    """
    stand_ins = {
        name: f"<the output {name} produces earlier in the run>" for name in agents
    }
    return {
        name: agent.build_prompt(QUESTION, context, {**stand_ins, **(outputs or {})})
        for name, agent in agents.items()
    }


def judge_context(context: str, measured: dict[str, object], gates: list[dict[str, object]], judge: str) -> str:
    """The context a judge is given: the dataset, plus what the tools measured and how its gates land.

    A worker agent is told what the data looks like; a judge additionally needs the outcomes it is
    holding to a bar, and the arithmetic on its own thresholds — that is what makes its verdict checkable
    against the numbers rather than against its own reading of an earlier agent's summary.
    """
    measurements = "\n".join(
        f"  {name:<24} {value}" for name, value in (measured.get("measurements") or {}).items()
    )
    mine = [gate for gate in gates if gate["judge"] == judge]
    evaluated = "\n".join(
        f"  {'PASS' if gate['passed'] else 'FAIL'}  {gate['threshold']:<28} "
        f"required {gate['required']}, measured {gate['measured']}"
        f"  [from: {gate.get('source', 'tools')}]"
        f"{'  — ' + gate['note'] if gate.get('note') else ''}"
        for gate in mine
    ) or "  (this judge has no configured thresholds)"

    return f"""{context}

## Measured outcomes — read out of tool results, not claimed in prose

{measurements}

## Your thresholds, evaluated against those measurements

{evaluated}

`from: agent` means the number came out of a tool the agent under review actually ran. `from: tools`
means no agent produced it and it fell back to the reference run — which is itself worth noting, since
it means the work you are reviewing never measured that. Treat this arithmetic as ground truth: if an
agent's write-up disagrees with a number here, the number is right and the write-up is a finding."""


def run_pipeline(
    configs,
    agents: dict[str, object],
    context: str,
    measured: dict[str, object] | None = None,
    gates: list[dict[str, object]] | None = None,
) -> dict[str, str]:
    """Walk the `pipeline:` block: run each step's agent, then its next step, then any judger it needs."""
    outputs: dict[str, str] = {}
    orchestrators = getattr(configs, "orchestrators", None) or {}
    baseline = dict((measured or {}).get("measurements") or {})
    judges = {
        name for name, config in configs.agent_configs.items() if config.type == "judger"
    }

    def context_for(name: str) -> str:
        """Judges get the measured outcomes and their gate arithmetic; workers get the dataset alone."""
        if name in judges and measured:
            return judge_context(context, measured, gates or [], name)
        return context

    def feed(configs, name: str, produced: dict[str, str], step: str | None) -> list[str]:
        """Report what feeds this agent, and complain when the pipeline reaches it out of order.

        Every agent is handed the full `outputs` dict; `dependency_agent` is what decides which of it
        lands in the agent's `{agent_output}`. So a dependency the pipeline has not run yet is not a
        crash — the prompt says the output is missing — but it is a config error worth naming, because
        the agent is about to reason from a hole.
        """
        upstream = dependencies_of(configs, name)
        missing = [dependency for dependency in upstream if dependency not in produced]
        if missing:
            logger.warning(
                f"Step {step}: {name} depends on {missing}, which has not run yet — "
                "check the pipeline order against the agent's `dependency_agent`."
            )
        return upstream

    for pipeline in getattr(configs, "pipeline", None) or []:
        for _ in range(int(pipeline.get("loops", 1))):
            for step in pipeline.get("steps", []):
                for processor in (step.get("processor"), step.get("next_step")):
                    if not processor:
                        continue

                    if processor.get("type") == "orchestrator":
                        sub_agents = orchestrators.get(processor["name"], {}).get("sub_agents", [])
                    else:
                        sub_agents = [processor["name"]]

                    for name in sub_agents:
                        agent = agents.get(name)
                        if not agent:
                            logger.warning(f"Pipeline step {step.get('name')} references unknown agent {name!r}.")
                            continue

                        upstream = feed(configs, name, outputs, step.get("name"))
                        print(f"\n{'=' * 96}\n{step.get('name')} -> {name}"
                              f"{'  <- ' + ', '.join(upstream) if upstream else ''}\n{'=' * 96}")
                        outputs[name] = agent.run(
                            question=QUESTION, context=context_for(name), agent_outputs=outputs
                        )
                        print(outputs[name])

                for judger in step.get("needs_judger", []) or []:
                    if judger not in agents:
                        logger.warning(f"Pipeline step {step.get('name')} needs unknown judger {judger!r}.")
                        continue

                    # Re-measure first: the agents in this step have just run their tools, so the gate is
                    # evaluated against what they produced rather than against the reference chain.
                    gates = gates_now(configs, agents, baseline) if baseline else (gates or [])
                    mine = [gate for gate in gates if gate["judge"] == judger]
                    failed = [gate for gate in mine if not gate["passed"]]
                    from_agents = sum(1 for gate in mine if gate.get("source") == "agent")
                    upstream = feed(configs, judger, outputs, step.get("name"))
                    print(f"\n{'-' * 96}\n{step.get('name')} gate: {judger}"
                          f"{'  <- ' + ', '.join(upstream) if upstream else ''}"
                          f"\n  [{len(failed)}/{len(mine)} failing by the numbers, "
                          f"{from_agents} measured by the agents themselves]\n{'-' * 96}")
                    outputs[judger] = agents[judger].run(
                        question=QUESTION, context=context_for(judger), agent_outputs=outputs
                    )
                    print(outputs[judger])

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true", help="Call the models and run the pipeline for real.")
    # Defaults resolve to absolute paths after parsing rather than in the parser, so `--help` shows the
    # repo-relative form and the run still works from any working directory.
    parser.add_argument("--config-dir", default=None,
                        help=f"Directory holding the benchmark YAML. [default: {tools.shown(CONFIG_DIR)}]")
    parser.add_argument("--skip-tools", action="store_true", help="Skip the tool chain and use a bare context.")
    parser.add_argument("--log", default=None,
                        help="Where to write the plain-text transcript of this run. "
                             f"[default: {tools.shown(GENERATED_DIR / 'run.log')}]")
    arguments = parser.parse_args()
    arguments.config_dir = arguments.config_dir or str(CONFIG_DIR)
    arguments.log = arguments.log or str(GENERATED_DIR / "run.log")

    transcript = Transcript(sys.stdout, Path(arguments.log))
    sys.stdout = transcript  # every print below lands in the terminal and in the log
    try:
        run(arguments)
    finally:
        sys.stdout = transcript.stream
        transcript.close()
        print(f"\n  Transcript written to {tools.shown(arguments.log)}")


def run(arguments) -> None:
    """The benchmark itself. Everything it prints is mirrored into the transcript by `main`."""
    if not tools.TRAIN_DATA.exists():
        print("Dataset missing — generating it first.")
        from benchmarks.dataset import generate

        generate()

    print(f"\n{'=' * 96}\nGROUND TRUTH — what the tools measure, with no model in the loop\n{'=' * 96}")
    measured = {} if arguments.skip_tools else ground_truth()
    if measured:
        print(json.dumps(measured, indent=2))

    context = data_context(measured) if measured else "See benchmarks/data/churn_train.csv."

    configs = load_configs(arguments.config_dir)
    if not arguments.live:
        stood_in = stand_in_for_missing_keys(configs)
        if stood_in:
            print(f"\nNo key set for {', '.join(stood_in)} — standing in, since a dry run makes no calls.")

    agents = workflow(configs)
    print(f"\n{'=' * 96}\nWORKFLOW — built from {tools.shown(arguments.config_dir)}\n{'=' * 96}")
    for name, agent in agents.items():
        upstream = dependencies_of(configs, name)
        lines = [
            f"  {name:<28} {type(agent).__name__:<18} {type(agent.llm).__name__}({agent.llm.model_name})",
            f"  {'':<28} works from: {', '.join(upstream) if upstream else '(starts the run)'}",
            f"  {'':<28} tools:      {', '.join(agent.toolbox.agent_tools) if agent.toolbox else '-'}",
        ]
        retrieval = retrieval_report(agent)
        if retrieval:
            lines.append(retrieval)
        print("\n".join(lines))

    gates = gate_report(configs, measured.get("measurements") or {}) if measured else []
    if gates:
        print(f"\n{'=' * 96}\nGATES — every judge's thresholds, evaluated against the measurements\n{'=' * 96}")
        judge = None
        for gate in gates:
            if gate["judge"] != judge:
                judge = gate["judge"]
                print(f"\n  {judge}")
            print(
                f"    {'PASS' if gate['passed'] else 'FAIL'}  {gate['threshold']:<28}"
                f" required {str(gate['required']):<8} measured {gate['measured']}"
                f"{'  — ' + gate['note'] if gate.get('note') else ''}"
            )
        failed = [gate for gate in gates if not gate["passed"]]
        print(f"\n  {len(gates) - len(failed)}/{len(gates)} thresholds pass"
              f"{' — failing: ' + ', '.join(gate['threshold'] for gate in failed) if failed else ''}")

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    (GENERATED_DIR / "ground_truth.json").write_text(json.dumps(measured, indent=2))
    (GENERATED_DIR / "gates.json").write_text(json.dumps(gates, indent=2))

    rendered = render(agents, context, {})
    for name, prompt in rendered.items():
        (GENERATED_DIR / f"{name}.md").write_text(prompt)

    # The prompt that answers the problem: the first step's, grounded in what the tools measured.
    problem_prompt = rendered.get("rag_problem_thinker_agent", '')
    (GENERATED_DIR / "problem_prompt.md").write_text(problem_prompt)
    print(f"\n{'=' * 96}\nPROMPT — what the first agent is sent, grounded in the dataset\n{'=' * 96}")
    print(problem_prompt)
    print(f"\nRendered prompts written to {tools.shown(GENERATED_DIR)}/")

    if arguments.live:
        outputs = run_pipeline(configs, agents, context, measured, gates)

        gates = gates_now(configs, agents, measured.get("measurements") or {}) if measured else gates
        record = run_record(configs, agents, outputs, measured, gates)
        (GENERATED_DIR / "run.json").write_text(json.dumps(record, indent=2, default=str))

        print(f"\n{'=' * 96}\nVERDICTS\n{'=' * 96}")
        for judge, verdict in record["summary"]["verdicts"].items():
            thresholds = [gate for gate in gates if gate["judge"] == judge]
            failed = [gate["threshold"] for gate in thresholds if not gate["passed"]]
            print(f"  {judge:<20} {verdict or '(no verdict line)':<24}"
                  f" gates {len(thresholds) - len(failed)}/{len(thresholds)}"
                  f"{'  failing: ' + ', '.join(failed) if failed else ''}")

        summary = record["summary"]
        print(f"\n  {summary['agents_run']} agents ran, {summary['tool_calls']} tool calls "
              f"({summary['tools_failed']} failed), {summary['gates_passed']}/{summary['gates_total']} gates pass.")

        if measured:
            report = reproduction_report(agents, measured["measurements"])
            print(f"\n{'=' * 96}\nREPRODUCTION — what the agents measured themselves, "
                  f"against the reference chain\n{'=' * 96}")
            for row in report["rows"]:
                mark = "  ok " if row["matches"] else ("DIFF" if row["agents"] is not None else "none")
                print(f"  {mark}  {row['measurement']:<24} reference {str(row['reference']):<10}"
                      f" agents {row['agents']}")
            print(
                f"\n  {report['tool_calls']} tool call(s) across the run, {report['tools_failed']} failed."
                f"\n  {report['measurements_produced']}/{report['measurements_total']} measurements "
                f"produced by the agents, {report['measurements_reproduced']} matching the reference."
            )
            (GENERATED_DIR / "reproduction.json").write_text(json.dumps(report, indent=2, default=str))

        print(f"\n  Structured record written to {tools.shown(GENERATED_DIR / 'run.json')}"
              f"\n  Reproduction written to {tools.shown(GENERATED_DIR / 'reproduction.json')}")
    else:
        print("\nNothing was sent to a provider. Re-run with --live to execute the pipeline.")


if __name__ == "__main__":
    main()
