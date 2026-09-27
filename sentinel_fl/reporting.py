"""Summary tables and a self-describing RESULTS_SUMMARY.md for a finished leave-one-subject-out run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import VERSION
from .metrics import bootstrap_mean
from .utils import write_json

HEADLINE_METRICS = [
    "macro_f1",
    "accuracy",
    "balanced_acc",
    "auroc",
    "kappa",
    "sensitivity",
    "specificity",
    "f1_class0",
    "f1_class1",
    "minority_ap",
    "brier",
    "nll",
    "probability_ece",
    "inference_ms_median",
]

METRIC_GLOSSARY = {
    "macro_f1": "Unweighted mean of the two per-class F1 scores at the selected threshold (primary metric; higher is better).",
    "accuracy": "Fraction of epochs/windows classified correctly.",
    "balanced_acc": "Mean of sensitivity and specificity.",
    "auroc": "Area under the ROC curve (threshold free).",
    "kappa": "Cohen's kappa between predictions and labels.",
    "sensitivity": "Recall of class 1 (SLEEP: sleep; WESAD: stress).",
    "specificity": "Recall of class 0 (SLEEP: wake; WESAD: non-stress).",
    "f1_class0": "F1 of class 0 (SLEEP: wake, the minority class; WESAD: non-stress).",
    "f1_class1": "F1 of class 1 (SLEEP: sleep; WESAD: stress).",
    "minority_ap": "Average precision of the minority class (SLEEP: wake; WESAD: stress).",
    "brier": "Mean squared error of the predicted probability (lower is better).",
    "nll": "Negative log-likelihood of the predicted probability (lower is better).",
    "probability_ece": "Expected calibration error of the predicted probability (lower is better).",
    "inference_ms_median": "Median single-row forward-pass time in milliseconds on the machine that ran the experiment.",
}


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return f"{x:.{nd}f}"
    return str(x)


def md_table(df, nd=4):
    """Render a DataFrame as a GitHub-flavoured Markdown table without extra dependencies."""
    if df is None or not len(df):
        return "_(no rows)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c], nd) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def per_subject_table(metrics):
    """Seed-averaged metrics of every held-out subject (one row per dataset and subject)."""
    cols = [m for m in HEADLINE_METRICS if m in metrics]
    extra = [c for c in ("n", "prevalence", "threshold", "sequence_strength") if c in metrics]
    g = metrics.groupby(["dataset", "client_id"], sort=True)
    out = g[cols + extra].mean().reset_index()
    out.insert(2, "n_seeds", g.seed.nunique().to_numpy())
    return out


def summary_table(metrics, min_ci_subjects=6, reps=2000):
    """Participant-level summary (mean, sd, median, p10, min, max, bootstrap CI) per dataset and metric."""
    rows = []
    for dataset, g in metrics.groupby("dataset", sort=True):
        for metric in HEADLINE_METRICS:
            if metric not in g:
                continue
            per = g.groupby("client_id")[metric].mean()
            x = per.dropna().to_numpy()
            lo, hi = bootstrap_mean(x, reps, f"{dataset}:{metric}", min_ci_subjects)
            rows.append(
                dict(
                    dataset=dataset,
                    metric=metric,
                    n_subjects=len(per),
                    n_valid_subjects=len(x),
                    mean=float(np.mean(x)) if len(x) else np.nan,
                    sd=float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
                    median=float(np.median(x)) if len(x) else np.nan,
                    p10=float(np.quantile(x, 0.1)) if len(x) else np.nan,
                    minimum=float(np.min(x)) if len(x) else np.nan,
                    maximum=float(np.max(x)) if len(x) else np.nan,
                    ci_low=lo,
                    ci_high=hi,
                    ci_reportable=bool(len(x) >= min_ci_subjects),
                    n_seeds_min=int(g.groupby("client_id").seed.nunique().min()),
                    degenerate_subject_runs=int(g["degenerate_subject"].sum()) if "degenerate_subject" in g else 0,
                )
            )
    return pd.DataFrame(rows)


def communication_table(systems):
    """Uplink/downlink bytes, distortion and compute cost per dataset (mean over runs)."""
    if not len(systems):
        return pd.DataFrame()
    cols = [
        "uplink_bytes",
        "downlink_bytes",
        "selected_uplink_bytes",
        "selected_round",
        "completed_rounds",
        "client_rounds",
        "model_params",
        "model_packet_fp32_bytes",
        "relative_distortion",
        "client_train_ms",
        "client_encode_ms",
        "server_screen_ms",
        "server_audit_ms",
        "server_aggregate_ms",
        "audit_excluded_client_rounds",
        "low_consensus_client_rounds",
        "infeasible_client_rounds",
        "screen_tpr",
        "screen_fpr",
    ]
    cols = [c for c in cols if c in systems]
    return systems.groupby("dataset", sort=True)[cols].mean().reset_index()


def personalization_table(personal):
    """Mean query-suffix macro-F1 of the shared and the personalised model per dataset."""
    if not len(personal):
        return pd.DataFrame()
    g = personal.groupby(["dataset", "arm"], sort=True)
    out = g.agg(
        n_subject_runs=("client_id", "size"),
        macro_f1=("macro_f1", "mean"),
        balanced_acc=("balanced_acc", "mean"),
        gate_passed_share=("adaptation_gate_passed", "mean"),
        eligible_share=("adaptation_eligible", "mean"),
        mean_personal_bias=("personal_bias", "mean"),
    ).reset_index()
    return out


def write_summary(root, min_ci_subjects=6, reps=2000):
    """Read the canonical CSVs in ``root`` and write summary tables plus ``RESULTS_SUMMARY.md``."""
    root = Path(root)
    metrics = pd.read_csv(root / "participant_metrics.csv")
    systems = pd.read_csv(root / "system_metrics.csv") if (root / "system_metrics.csv").exists() else pd.DataFrame()
    personal = pd.read_csv(root / "personalization.csv") if (root / "personalization.csv").exists() else pd.DataFrame()
    ledger = (
        pd.read_csv(root / "completion_ledger.csv") if (root / "completion_ledger.csv").exists() else pd.DataFrame()
    )
    manifest_path = root / "run_manifest.json"
    if not manifest_path.exists():
        cands = sorted(root.glob("run_manifest*.json"))
        manifest_path = cands[0] if cands else None
    manifest = json.loads(manifest_path.read_text()) if manifest_path else {}

    summary = summary_table(metrics, min_ci_subjects, reps)
    subjects = per_subject_table(metrics)
    comm = communication_table(systems)
    pers = personalization_table(personal)
    summary.to_csv(root / "summary_by_dataset.csv", index=False)
    subjects.to_csv(root / "per_subject_results.csv", index=False)
    comm.to_csv(root / "communication_summary.csv", index=False)
    pers.to_csv(root / "personalization_summary.csv", index=False)

    failed = int((ledger.status != "completed").sum()) if len(ledger) else 0
    completed = int((ledger.status == "completed").sum()) if len(ledger) else len(metrics)
    planned = manifest.get("total_planned_runs")
    complete = failed == 0 and (planned is None or completed >= planned)
    cfg = manifest.get("config", {})
    synthetic = bool(metrics["synthetic"].any()) if "synthetic" in metrics else False
    status = dict(
        version=VERSION,
        runs_completed=completed,
        runs_failed=failed,
        runs_planned=planned,
        complete=bool(complete),
        synthetic_data=synthetic,
        datasets={d: int(g.client_id.nunique()) for d, g in metrics.groupby("dataset")},
    )
    write_json(root / "run_status.json", status)

    lines = [f"# SENTINEL-FL results summary ({VERSION})", ""]
    lines += [
        "This file is generated by `sentinel-fl summarize` from the CSVs in the same folder. It is written so that a person "
        "or an AI tool can understand the run without opening the code.",
        "",
        "## 1. How to read this run",
        "",
        "- Evaluation is leave-one-subject-out: each held-out subject is never used for fitting, calibration, threshold "
        "selection, decoder tuning or learning-rate selection. The statistical unit is the held-out participant; results "
        "of several seeds are averaged within a participant before any summary statistic is computed.",
        "- Datasets: SLEEP (class 0 = wake, the minority class; class 1 = sleep) and WESAD (class 0 = non-stress; "
        "class 1 = stress). Each dataset is trained and evaluated independently.",
        "- Intervals are percentile bootstrap intervals over held-out participants and are reported only when "
        f"at least {min_ci_subjects} participants were evaluated (`ci_reportable`).",
        "- Communication figures are application-level bytes of the serialized protocol; they are not network-layer "
        "measurements and they do not include transport encryption.",
        "",
        "## 2. Run status",
        "",
        f"- Runs completed: {completed}" + (f" of {planned} planned" if planned else "") + f"; failed: {failed}.",
        f"- Complete: {'yes' if complete else 'NO - do not report this run as a full result'}.",
        f"- Synthetic fixture data: {'YES - software test only' if synthetic else 'no (real data)'}.",
        "",
    ]
    if cfg:
        keep = [
            "rounds",
            "local_steps",
            "batch_size",
            "hidden",
            "clients_per_round",
            "byte_budget",
            "sketch_dim",
            "seeds",
            "max_folds",
            "val_subjects",
            "hp_lrs",
            "hpo_rounds",
            "sequence_decode",
            "adapt_shots",
        ]
        lines += [
            "Configuration:",
            "",
            md_table(pd.DataFrame([dict(parameter=k, value=json.dumps(cfg.get(k))) for k in keep if k in cfg])),
        ]
    sc = manifest.get("scenario", {})
    if sc:
        lines += [
            f"Scenario: `{sc.get('name')}` (attack `{sc.get('attack')}`, malicious fraction {sc.get('bad_fraction')}).",
            "",
        ]
    lines += [
        "## 3. Held-out performance (participant-level)",
        "",
        md_table(summary[summary.metric.isin(HEADLINE_METRICS)]),
        "",
    ]
    lines += ["## 4. Per-subject results (seed-averaged)", "", md_table(subjects), ""]
    lines += ["## 5. Communication, compute and protocol diagnostics (mean per run)", "", md_table(comm, 5), ""]
    if len(pers):
        lines += [
            "## 6. Personalisation on the query suffix",
            "",
            "`shared` is the model after calibration and sequence decoding; `personalised` additionally applies the gated, "
            "shrunk personal bias fitted on the first labelled epochs. Both are scored on the same later query epochs.",
            "",
            md_table(pers),
            "",
        ]
    lines += ["## 7. Metric glossary", ""]
    lines += [f"- `{k}`: {v}" for k, v in METRIC_GLOSSARY.items()]
    lines += [
        "",
        "## 8. Files in this folder",
        "",
        "- `participant_metrics.csv`: one row per held-out subject and seed with every metric and run identifier.",
        "- `system_metrics.csv`: bytes, timing, screening and audit diagnostics per run.",
        "- `personalization.csv`: shared vs. personalised metrics per held-out subject.",
        "- `completion_ledger.csv`: status of every requested run (completed / failed).",
        "- `summary_by_dataset.csv`, `per_subject_results.csv`, `communication_summary.csv`, `personalization_summary.csv`: "
        "the tables shown above.",
        "- `run_manifest.json`, `run_status.json`, `data_provenance.json`: configuration, splits, environment and data fingerprints.",
        "- `runs/<key>/`: per-run `artifact.pkl` (deployable model bundle), `predictions.csv`, `rounds.csv`, "
        "`client_events.csv`, `result.json`.",
        "",
    ]
    (root / "RESULTS_SUMMARY.md").write_text("\n".join(lines))
    return summary, status
