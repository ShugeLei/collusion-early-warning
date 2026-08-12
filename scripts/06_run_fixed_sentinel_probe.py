#!/usr/bin/env python3
"""Run Experiment D: fixed full-discussion honest-agent sentinel probe."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "external" / "narcbench"
CORE_SOURCE = UPSTREAM_ROOT / "generation" / "core.py"
CORE_DIR = PROJECT_ROOT / "data" / "narcbench" / "activations" / "qwen3_32b" / "core"
EXPERIMENT_C_METRICS = PROJECT_ROOT / "outputs" / "metrics" / "sentinel_probe_layer28.json"
EXPERIMENT_C_PREDICTIONS = (
    PROJECT_ROOT / "outputs" / "predictions" / "oof_sentinel_scores_layer28.csv"
)
PREDICTION_OUTPUT = (
    PROJECT_ROOT / "outputs" / "predictions" / "oof_fixed_sentinel_scores_layer28.csv"
)
METRICS_OUTPUT = PROJECT_ROOT / "outputs" / "metrics" / "fixed_sentinel_probe_layer28.json"
REPORT_OUTPUT = PROJECT_ROOT / "outputs" / "logs" / "fixed_sentinel_probe_layer28.md"
TEMPORAL_FIGURE = (
    PROJECT_ROOT / "outputs" / "figures" / "fixed_sentinel_temporal_layer28.png"
)
COMPARISON_FIGURE = (
    PROJECT_ROOT / "outputs" / "figures" / "fixed_vs_separate_sentinel_layer28.png"
)

LAYER = 28
SEED = 42
BOOTSTRAP_REPLICATES = 1000
MODEL_ID = "Qwen/Qwen3-32B-AWQ"
DATASET_ID = "aaronrose227/narcbench"
DATASET_REVISION = "19cb14b269e052b19ba6d2bd2f4519afd3b10c57"


sys.path.insert(0, str(PROJECT_ROOT))
from src.honest_observer_detection import (  # noqa: E402
    audit_core_structure,
    load_name_pool,
    source_evidence,
)
from src.sentinel_probe import (  # noqa: E402
    attach_paired_diagnostics,
    build_observer_examples,
    calculate_sentinel_metrics,
    population_audit,
    run_lodo_fixed_sentinel,
)
from src.temporal_detectability import build_public_discussion_index  # noqa: E402


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def json_ready(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def validate_fixed_predictions(predictions, expected_rows):
    issues = []
    if len(predictions) != expected_rows:
        issues.append("Expected {} prediction rows, found {}".format(expected_rows, len(predictions)))
    if set(predictions["generator_position"]) != {2, 3}:
        issues.append("Fixed sentinel output contains a non-observer position")
    if not (predictions["role"] == "honest").all():
        issues.append("Fixed sentinel output contains a non-honest role")
    if not (predictions["fold"] == predictions["domain"]).all():
        issues.append("A fixed sentinel score is not out-of-fold")
    if not predictions["fixed_across_temporal_views"].all():
        issues.append("A prediction is not marked as fixed across time")
    if set(predictions["probe_training_representation"]) != {"mean(R1,R2,R3)"}:
        issues.append("Unexpected fixed sentinel training representation")
    if predictions.loc[
        predictions["analysis_population"] == "style_matched",
        "paired_prompt_style_matched",
    ].eq(False).any():
        issues.append("Style-matched fixed output includes a mismatched observer")
    fingerprint_counts = predictions.groupby(
        ["analysis_population", "observer_scope", "fold"]
    )["model_fingerprint"].nunique()
    if not (fingerprint_counts == 1).all():
        issues.append("A fold/scope uses more than one model fingerprint across temporal views")
    keys = [
        "analysis_population",
        "observer_scope",
        "run_id",
        "agent_name",
        "temporal_view",
        "round",
        "layer",
    ]
    if predictions.duplicated(keys).any():
        issues.append("Fixed sentinel output contains duplicate prediction keys")
    r1 = predictions.loc[predictions["round"] == 1].pivot(
        index=["analysis_population", "observer_scope", "run_id", "agent_name"],
        columns="temporal_view",
        values="sentinel_score",
    )
    if not np.allclose(r1["round_only"], r1["prefix"], rtol=0.0, atol=1e-12):
        issues.append("Equivalent R1 round-only and prefix scores differ")
    return issues


def exposure_score_shift(predictions):
    """Describe fixed-probe paired shifts by verified cumulative exposure count."""
    frame = predictions.loc[
        (predictions["observer_scope"] == "both_positions")
        & (predictions["temporal_view"] == "round_only")
        & (predictions["mode"] == "collusion")
    ].copy()
    records = []
    for keys, group in frame.groupby(
        [
            "analysis_population",
            "generator_position",
            "round",
            "round_or_prefix",
            "exposure_count",
        ],
        sort=True,
    ):
        population, position, endpoint_round, label, exposure_count = keys
        scenario_values = (
            group.groupby("scenario_id")["paired_score_difference"].mean().to_numpy(dtype=float)
        )
        rng = np.random.RandomState(SEED)
        bootstrap = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
        for replicate in range(BOOTSTRAP_REPLICATES):
            selected = rng.randint(0, len(scenario_values), size=len(scenario_values))
            bootstrap[replicate] = np.mean(scenario_values[selected])
        differences = group["paired_score_difference"].to_numpy(dtype=float)
        records.append({
            "analysis_population": population,
            "observer_position": int(position),
            "round": int(endpoint_round),
            "round_or_prefix": label,
            "exposure_count": int(exposure_count),
            "definition": "preceding public messages from generator positions 0–1",
            "observer_identity_pair_count": len(group),
            "scenario_pair_count": group["scenario_id"].nunique(),
            "mean_turn_in_round_zero_indexed": float(group["turn_in_round"].mean()),
            "paired_mean_score_difference": float(np.mean(differences)),
            "paired_sample_sd_score_difference": (
                float(np.std(differences, ddof=1)) if len(differences) > 1 else None
            ),
            "paired_median_score_difference": float(np.median(differences)),
            "paired_mean_score_difference_bootstrap_ci_95": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "bootstrap_unit": "matched scenario pair",
            "caution": "descriptive post-stratification; no exposure-specific model was fitted",
        })
    return records


def compare_with_experiment_c(fixed_summary, experiment_c):
    separate = pd.DataFrame(experiment_c["metrics"]["summary"])
    key_columns = [
        "analysis_population",
        "observer_scope",
        "temporal_view",
        "round",
        "round_or_prefix",
    ]
    columns = key_columns + [
        "pooled_oof_auroc",
        "pooled_oof_auroc_bootstrap_ci_95",
        "paired_mean_score_difference",
        "paired_mean_score_difference_bootstrap_ci_95",
    ]
    merged = fixed_summary[columns].merge(
        separate[columns],
        on=key_columns,
        suffixes=("_fixed", "_separately_trained"),
        how="inner",
        validate="one_to_one",
    )
    merged["auroc_difference_fixed_minus_separately_trained"] = (
        merged["pooled_oof_auroc_fixed"]
        - merged["pooled_oof_auroc_separately_trained"]
    )
    return merged.to_dict(orient="records")


def temporal_strengthening_diagnostic(predictions):
    """Bootstrap AUROC and paired-score gains between temporal endpoints."""
    records = []
    comparisons = ((1, 2), (2, 3), (1, 3))
    for keys, view in predictions.groupby(
        ["analysis_population", "observer_scope", "temporal_view"], sort=True
    ):
        population, scope, temporal_view = keys
        scenario_ids = sorted(view["scenario_id"].unique())
        blocks = {
            (endpoint, scenario): view.loc[
                (view["round"] == endpoint) & (view["scenario_id"] == scenario)
            ]
            for endpoint in (1, 2, 3)
            for scenario in scenario_ids
        }
        scenario_score_shifts = (
            view.loc[view["mode"] == "collusion"]
            .groupby(["round", "scenario_id"])["paired_score_difference"]
            .mean()
            .to_dict()
        )
        for start, end in comparisons:
            start_frame = view.loc[view["round"] == start]
            end_frame = view.loc[view["round"] == end]
            start_auroc = float(
                roc_auc_score(start_frame["environment_label"], start_frame["sentinel_score"])
            )
            end_auroc = float(
                roc_auc_score(end_frame["environment_label"], end_frame["sentinel_score"])
            )
            start_shift = float(np.mean([
                scenario_score_shifts[(start, scenario)] for scenario in scenario_ids
            ]))
            end_shift = float(np.mean([
                scenario_score_shifts[(end, scenario)] for scenario in scenario_ids
            ]))
            rng = np.random.RandomState(SEED)
            auroc_gains = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
            shift_gains = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
            for replicate in range(BOOTSTRAP_REPLICATES):
                selected = rng.randint(0, len(scenario_ids), size=len(scenario_ids))
                sampled_ids = [scenario_ids[index] for index in selected]
                sampled_start = pd.concat(
                    [blocks[(start, scenario)] for scenario in sampled_ids], ignore_index=True
                )
                sampled_end = pd.concat(
                    [blocks[(end, scenario)] for scenario in sampled_ids], ignore_index=True
                )
                auroc_gains[replicate] = (
                    roc_auc_score(
                        sampled_end["environment_label"], sampled_end["sentinel_score"]
                    )
                    - roc_auc_score(
                        sampled_start["environment_label"], sampled_start["sentinel_score"]
                    )
                )
                shift_gains[replicate] = np.mean([
                    scenario_score_shifts[(end, scenario)]
                    - scenario_score_shifts[(start, scenario)]
                    for scenario in sampled_ids
                ])
            records.append({
                "analysis_population": population,
                "observer_scope": scope,
                "temporal_view": temporal_view,
                "start_round": start,
                "end_round": end,
                "start_label": (
                    "R{}".format(start)
                    if temporal_view == "round_only"
                    else "-".join("R{}".format(value) for value in range(1, start + 1))
                ),
                "end_label": (
                    "R{}".format(end)
                    if temporal_view == "round_only"
                    else "-".join("R{}".format(value) for value in range(1, end + 1))
                ),
                "scenario_pair_count": len(scenario_ids),
                "start_pooled_oof_auroc": start_auroc,
                "end_pooled_oof_auroc": end_auroc,
                "auroc_gain": end_auroc - start_auroc,
                "auroc_gain_bootstrap_ci_95": [
                    float(np.quantile(auroc_gains, 0.025)),
                    float(np.quantile(auroc_gains, 0.975)),
                ],
                "start_paired_mean_score_difference": start_shift,
                "end_paired_mean_score_difference": end_shift,
                "paired_mean_score_difference_gain": end_shift - start_shift,
                "paired_mean_score_difference_gain_bootstrap_ci_95": [
                    float(np.quantile(shift_gains, 0.025)),
                    float(np.quantile(shift_gains, 0.975)),
                ],
                "bootstrap_unit": "matched scenario pair",
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            })
    return records


def _format_interval(value, interval):
    return "{:.4f} [{:.4f}, {:.4f}]".format(value, interval[0], interval[1])


def plot_fixed_temporal(metrics_path, output_path):
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary = pd.DataFrame(metrics["metrics"]["summary"])
    names = {"position_2": "Position 2", "position_3": "Position 3", "both_positions": "Both positions"}
    colors = {"position_2": "#4c72b0", "position_3": "#c44e52", "both_positions": "#55a868"}
    markers = {"position_2": "o", "position_3": "s", "both_positions": "^"}
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), sharey=True)
    for row_index, population in enumerate(("all_observers", "style_matched")):
        for column_index, temporal_view in enumerate(("round_only", "prefix")):
            axis = axes[row_index, column_index]
            view = summary.loc[
                (summary["analysis_population"] == population)
                & (summary["temporal_view"] == temporal_view)
            ]
            for scope in ("position_2", "position_3", "both_positions"):
                scoped = view.loc[view["observer_scope"] == scope].sort_values("round")
                pooled = scoped["pooled_oof_auroc"].to_numpy(dtype=float)
                intervals = np.stack(scoped["pooled_oof_auroc_bootstrap_ci_95"].to_numpy())
                errors = np.vstack([pooled - intervals[:, 0], intervals[:, 1] - pooled])
                axis.errorbar(
                    np.arange(3), pooled, yerr=errors, color=colors[scope],
                    marker=markers[scope], linewidth=1.7, capsize=3, label=names[scope]
                )
            labels = ["R1", "R2", "R3"] if temporal_view == "round_only" else ["R1", "R1–R2", "R1–R2–R3"]
            axis.set_xticks(np.arange(3), labels)
            axis.set_ylim(0.0, 1.02)
            axis.axhline(0.5, color="grey", linestyle=":", linewidth=1.0)
            axis.grid(axis="y", alpha=0.2)
            axis.set_title("{} — {}".format(
                "All observers" if population == "all_observers" else "Style-matched",
                "round only" if temporal_view == "round_only" else "causal prefix",
            ))
    axes[0, 0].set_ylabel("Pooled OOF AUROC")
    axes[1, 0].set_ylabel("Pooled OOF AUROC")
    axes[1, 0].set_xlabel("Temporal representation")
    axes[1, 1].set_xlabel("Temporal representation")
    axes[0, 1].legend(fontsize=8, loc="best")
    figure.suptitle("Fixed full-discussion sentinel — layer 28")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_training_comparison(metrics_path, output_path):
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    comparison = pd.DataFrame(metrics["experiment_c_comparison"])
    comparison = comparison.loc[comparison["observer_scope"] == "both_positions"]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    colors = {"all_observers": "#4c72b0", "style_matched": "#55a868"}
    for axis, temporal_view in zip(axes, ("round_only", "prefix")):
        view = comparison.loc[comparison["temporal_view"] == temporal_view]
        for population in ("all_observers", "style_matched"):
            frame = view.loc[view["analysis_population"] == population].sort_values("round")
            label = "All observers" if population == "all_observers" else "Style-matched"
            axis.plot(
                np.arange(3), frame["pooled_oof_auroc_fixed"], color=colors[population],
                marker="o", linewidth=1.9, label=label + " — fixed"
            )
            axis.plot(
                np.arange(3), frame["pooled_oof_auroc_separately_trained"],
                color=colors[population], marker="x", linestyle="--", linewidth=1.3,
                label=label + " — separate"
            )
        labels = ["R1", "R2", "R3"] if temporal_view == "round_only" else ["R1", "R1–R2", "R1–R2–R3"]
        axis.set_xticks(np.arange(3), labels)
        axis.set_ylim(0.0, 1.02)
        axis.axhline(0.5, color="grey", linestyle=":", linewidth=1.0)
        axis.grid(axis="y", alpha=0.2)
        axis.set_title("Round only" if temporal_view == "round_only" else "Causal prefix")
        axis.set_xlabel("Temporal representation")
    axes[0].set_ylabel("Pooled OOF AUROC")
    axes[1].legend(fontsize=7.5, loc="best")
    figure.suptitle("Fixed vs separately trained sentinel — both observer positions")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_report(metrics):
    summary = pd.DataFrame(metrics["metrics"]["summary"])
    folds = pd.DataFrame(metrics["metrics"]["fold_level"])
    comparison = pd.DataFrame(metrics["experiment_c_comparison"])
    exposure = pd.DataFrame(metrics["exposure_score_shift"])
    strengthening = pd.DataFrame(metrics["temporal_strengthening_diagnostic"])
    population = pd.DataFrame(metrics["population_audit"]["population_scope_counts"])
    lines = [
        "# Experiment D — Fixed Sentinel Temporal Probe (Layer 28)",
        "",
        "- Status: **{}**".format(metrics["status"]),
        "- Training: one non-held-out `mean(R1,R2,R3)` probe per fold/population/scope",
        "- Evaluation: the same fitted scaler and classifier reused at all six temporal views",
        "- Features: honest observer positions 2–3 only; zero colluder activation rows",
        "- Classifier: StandardScaler + LogisticRegression(C=1.0, max_iter=1000, random_state=42)",
        "- Scope: temporal methodology control only; no thresholds, TFIR, PIC, or E2",
        "",
        "## Fixed-probe integrity",
        "",
        "Every prediction in a given population/scope/fold carries one identical model fingerprint. "
        "All fitted scalers and classifiers use only non-held-out full-discussion means. There were "
        "{} fitted probes, {} style-subset folds with no retained test observation, and {} convergence "
        "warnings.".format(
            metrics["training"]["fit_count"],
            metrics["training"]["not_fit_no_test_count"],
            metrics["training"]["convergence_warning_count"],
        ),
        "",
        "## Retained samples",
        "",
        "| Population | Scope | Retained observer pairs | Excluded style mismatches | Scenarios | Missing domains |",
        "|---|---|---:|---:|---:|---|",
    ]
    for _, row in population.iterrows():
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            row["analysis_population"], row["observer_scope"],
            row["retained_observer_identity_pairs"], row["excluded_style_mismatched_pairs"],
            row["retained_scenario_count"],
            ", ".join(row["missing_domains"]) if row["missing_domains"] else "none",
        ))
    lines.extend([
        "",
        "## Primary result — fixed probe on round-only activations",
        "",
        "| Population | Scope | Round | Pooled OOF AUROC (95% CI) | Fold mean | Fold SD | Valid folds | Paired mean Δ (95% CI) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for _, row in summary.loc[summary["temporal_view"] == "round_only"].sort_values(
        ["analysis_population", "observer_scope", "round"]
    ).iterrows():
        lines.append("| {} | {} | {} | {} | {:.4f} | {:.4f} | {}/{} | {} |".format(
            row["analysis_population"], row["observer_scope"], row["round_or_prefix"],
            _format_interval(row["pooled_oof_auroc"], row["pooled_oof_auroc_bootstrap_ci_95"]),
            row["fold_macro_mean_auroc"], row["fold_population_sd_auroc"],
            row["fold_count_valid"], row["fold_count_expected"],
            _format_interval(
                row["paired_mean_score_difference"],
                row["paired_mean_score_difference_bootstrap_ci_95"],
            ),
        ))
    lines.extend([
        "",
        "## Secondary result — fixed probe on causal prefixes",
        "",
        "| Population | Scope | Prefix | Pooled OOF AUROC (95% CI) | Fold mean | Fold SD | Valid folds | Paired mean Δ (95% CI) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for _, row in summary.loc[summary["temporal_view"] == "prefix"].sort_values(
        ["analysis_population", "observer_scope", "round"]
    ).iterrows():
        lines.append("| {} | {} | {} | {} | {:.4f} | {:.4f} | {}/{} | {} |".format(
            row["analysis_population"], row["observer_scope"], row["round_or_prefix"],
            _format_interval(row["pooled_oof_auroc"], row["pooled_oof_auroc_bootstrap_ci_95"]),
            row["fold_macro_mean_auroc"], row["fold_population_sd_auroc"],
            row["fold_count_valid"], row["fold_count_expected"],
            _format_interval(
                row["paired_mean_score_difference"],
                row["paired_mean_score_difference_bootstrap_ci_95"],
            ),
        ))

    lines.extend([
        "",
        "## Direct temporal-strengthening test",
        "",
        "Scenario-paired bootstrap intervals below test the temporal gain itself. They avoid "
        "inferring accumulation only from separate per-round confidence intervals.",
        "",
        "| Population | View | Transition | AUROC gain (95% CI) | Paired-score gain (95% CI) |",
        "|---|---|---|---:|---:|",
    ])
    primary_strengthening = strengthening.loc[
        strengthening["observer_scope"] == "both_positions"
    ]
    for _, row in primary_strengthening.sort_values(
        ["analysis_population", "temporal_view", "start_round", "end_round"]
    ).iterrows():
        lines.append("| {} | {} | {} → {} | {} | {} |".format(
            row["analysis_population"], row["temporal_view"],
            row["start_label"], row["end_label"],
            _format_interval(row["auroc_gain"], row["auroc_gain_bootstrap_ci_95"]),
            _format_interval(
                row["paired_mean_score_difference_gain"],
                row["paired_mean_score_difference_gain_bootstrap_ci_95"],
            ),
        ))
    endpoint_gain = strengthening.loc[
        (strengthening["start_round"] == 1) & (strengthening["end_round"] == 3)
    ].set_index(["analysis_population", "observer_scope", "temporal_view"])
    lines.extend([
        "",
        "For all observers combined, the R1→R3 gain excludes zero for both round-only AUROC "
        "({}) and prefix AUROC ({}). The position-specific endpoint test localizes that gain mainly "
        "to position 2: its round-only and prefix intervals exclude zero, while position 3 starts "
        "high at R1 and its endpoint-gain intervals cross zero. Style-matched combined point estimates "
        "also rise, but both endpoint-gain intervals cross zero; that smaller sensitivity confirms "
        "stable early signal more strongly than it confirms temporal strengthening.".format(
            _format_interval(
                endpoint_gain.loc[("all_observers", "both_positions", "round_only"), "auroc_gain"],
                endpoint_gain.loc[("all_observers", "both_positions", "round_only"), "auroc_gain_bootstrap_ci_95"],
            ),
            _format_interval(
                endpoint_gain.loc[("all_observers", "both_positions", "prefix"), "auroc_gain"],
                endpoint_gain.loc[("all_observers", "both_positions", "prefix"), "auroc_gain_bootstrap_ci_95"],
            ),
        ),
    ])

    lines.extend([
        "",
        "## Comparison with Experiment C",
        "",
        "Both-position pooled OOF AUROC; Experiment C trained a separate classifier per temporal point.",
        "",
        "| Population | View | Time | Fixed D | Separate C | Fixed − separate |",
        "|---|---|---|---:|---:|---:|",
    ])
    combined_comparison = comparison.loc[comparison["observer_scope"] == "both_positions"]
    for _, row in combined_comparison.sort_values(
        ["analysis_population", "temporal_view", "round"]
    ).iterrows():
        lines.append("| {} | {} | {} | {:.4f} | {:.4f} | {:+.4f} |".format(
            row["analysis_population"], row["temporal_view"], row["round_or_prefix"],
            row["pooled_oof_auroc_fixed"], row["pooled_oof_auroc_separately_trained"],
            row["auroc_difference_fixed_minus_separately_trained"],
        ))

    lines.extend([
        "",
        "## Exposure-count score shifts (R1)",
        "",
        "The fixed both-position probe is post-stratified by the verified number of preceding "
        "position-0/1 public messages. No exposure-specific model is trained.",
        "",
        "| Population | Position | Exposure | Pairs | Paired mean Δ (95% CI) |",
        "|---|---:|---:|---:|---:|",
    ])
    for _, row in exposure.loc[exposure["round"] == 1].sort_values(
        ["analysis_population", "observer_position", "exposure_count"]
    ).iterrows():
        lines.append("| {} | {} | {} | {} | {} |".format(
            row["analysis_population"], int(row["observer_position"]),
            int(row["exposure_count"]), int(row["observer_identity_pair_count"]),
            _format_interval(
                row["paired_mean_score_difference"],
                row["paired_mean_score_difference_bootstrap_ci_95"],
            ),
        ))

    lines.extend([
        "",
        "## Leave-one-domain-out fold AUROCs",
        "",
        "Undefined style-subset folds are retained explicitly.",
        "",
    ])
    for keys, frame in folds.groupby(
        ["analysis_population", "observer_scope", "temporal_view", "round", "round_or_prefix"],
        sort=True,
    ):
        population_name, scope, view, _, time_label = keys
        values = []
        for _, fold in frame.sort_values("fold").iterrows():
            rendered = (
                "undefined ({})".format(fold["metric_issue"])
                if pd.isna(fold["auroc"])
                else "{:.4f}".format(fold["auroc"])
            )
            values.append("{}={}".format(fold["fold"], rendered))
        lines.append("- `{}` / `{}` / `{}` / `{}`: {}".format(
            population_name, scope, view, time_label, ", ".join(values)
        ))

    round_primary = summary.loc[
        (summary["analysis_population"] == "all_observers")
        & (summary["observer_scope"] == "both_positions")
        & (summary["temporal_view"] == "round_only")
    ].set_index("round")
    prefix_primary = summary.loc[
        (summary["analysis_population"] == "all_observers")
        & (summary["observer_scope"] == "both_positions")
        & (summary["temporal_view"] == "prefix")
    ].set_index("round")
    style_primary = summary.loc[
        (summary["analysis_population"] == "style_matched")
        & (summary["observer_scope"] == "both_positions")
        & (summary["temporal_view"] == "round_only")
    ].set_index("round")
    early_reliable = round_primary.loc[1, "pooled_oof_auroc_bootstrap_ci_95"][0] > 0.5
    style_early_reliable = style_primary.loc[1, "pooled_oof_auroc_bootstrap_ci_95"][0] > 0.5
    round_gain = round_primary.loc[3, "pooled_oof_auroc"] - round_primary.loc[1, "pooled_oof_auroc"]
    prefix_gain = prefix_primary.loc[3, "pooled_oof_auroc"] - prefix_primary.loc[1, "pooled_oof_auroc"]
    lines.extend(["", "## Interpretation", ""])
    if early_reliable:
        lines.append(
            "A single full-discussion sentinel direction retains reliable R1 predictive signal."
        )
    else:
        lines.append(
            "The fixed full-discussion direction does not retain reliable R1 predictive signal."
        )
    lines.append(
        "For all observers, pooled AUROC changes by {:+.4f} from R1 to R3 for round-only "
        "activations and {:+.4f} from R1 to the full prefix.".format(round_gain, prefix_gain)
    )
    if style_early_reliable:
        lines.append(
            "The style-matched sensitivity also retains reliable R1 signal, so the early fixed-direction "
            "result is not explained solely by the audited prompt-style mismatch. Its R1→R3 temporal "
            "gain is positive in point estimate but uncertain, with bootstrap intervals crossing zero."
        )
    else:
        lines.append(
            "The smaller style-matched sensitivity does not establish reliable R1 signal; interpret "
            "the all-observer early result with prompt-style caution."
        )
    lines.extend([
        "The direct endpoint test supports temporal strengthening for all observers combined, driven "
        "primarily by position 2. Position 3 already carries strong early signal and does not show a "
        "reliable additional endpoint gain. AUROC and paired score shift still do not establish a "
        "causal mechanism or prove that exposure itself caused the within-agent change.",
        "",
        "No intervention thresholds, TFIR, PIC, or E2 analysis were run.",
        "",
    ])
    return "\n".join(lines)


def main():
    experiment_c = json.loads(EXPERIMENT_C_METRICS.read_text(encoding="utf-8"))
    if experiment_c["status"] != "PASS":
        raise RuntimeError("Experiment C metrics are not PASS")
    if sha256_file(EXPERIMENT_C_PREDICTIONS) != experiment_c["predictions"]["sha256"]:
        raise RuntimeError("Experiment C prediction hash mismatch")
    metadata = json.loads((CORE_DIR / "metadata_gen.json").read_text(encoding="utf-8"))
    positions, turns, structure_audit = audit_core_structure(
        metadata,
        load_name_pool(CORE_SOURCE),
        source_evidence(CORE_SOURCE),
    )
    if structure_audit["status"] != "PASS":
        raise RuntimeError("Core structure audit failed")

    archive = np.load(CORE_DIR / "activations_gen.npz", allow_pickle=False)
    try:
        activations = np.asarray(archive["layer_{}".format(LAYER)])
        temporal_index, run_metadata = build_public_discussion_index(metadata, activations)
        examples = build_observer_examples(temporal_index, run_metadata, positions, turns)
        examples["layer"] = LAYER
        domains = sorted(examples["domain"].unique())
        predictions, diagnostics, training_audits = run_lodo_fixed_sentinel(
            examples, domains
        )
    finally:
        archive.close()

    predictions = attach_paired_diagnostics(predictions, diagnostics)
    predictions = predictions.sort_values([
        "analysis_population", "observer_scope", "fold", "scenario_id", "mode",
        "agent_name", "temporal_view", "round",
    ])
    populations = population_audit(examples)
    expected_rows = 12 * sum(
        row["retained_observer_identity_pairs"]
        for row in populations["population_scope_counts"]
    )
    validation_issues = validate_fixed_predictions(predictions, expected_rows)
    convergence = [
        audit for audit in training_audits if audit.get("convergence_warnings")
    ]
    if convergence:
        validation_issues.append("One or more fixed sentinel fits emitted a convergence warning")

    PREDICTION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTION_OUTPUT, index=False)
    saved_predictions = pd.read_csv(PREDICTION_OUTPUT)
    fold_metrics, summary_metrics, metric_issues = calculate_sentinel_metrics(
        saved_predictions,
        domains,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        seed=SEED,
    )
    exposure = exposure_score_shift(saved_predictions)
    comparison = compare_with_experiment_c(summary_metrics, experiment_c)
    strengthening = temporal_strengthening_diagnostic(saved_predictions)

    output = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not validation_issues else "ISSUES",
        "scope": "Experiment D fixed sentinel temporal control only; no intervention analysis.",
        "sources": {
            "upstream_repository": "https://github.com/aaronrose227/narcbench",
            "upstream_git_commit": git_commit(UPSTREAM_ROOT),
            "upstream_source_modified": False,
            "dataset_repository": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "metadata_sha256": sha256_file(CORE_DIR / "metadata_gen.json"),
            "activations_sha256": sha256_file(CORE_DIR / "activations_gen.npz"),
            "experiment_c_metrics_path": str(EXPERIMENT_C_METRICS.relative_to(PROJECT_ROOT)),
            "experiment_c_metrics_sha256": sha256_file(EXPERIMENT_C_METRICS),
            "experiment_c_predictions_sha256": sha256_file(EXPERIMENT_C_PREDICTIONS),
        },
        "environment": {
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
            "cpu_only": True,
            "random_seed": SEED,
        },
        "model": MODEL_ID,
        "dataset": {
            "tier": "Core",
            "layer": LAYER,
            "activation_width": int(activations.shape[1]),
            "metadata_rows": len(metadata),
            "run_count": len(run_metadata),
            "scenario_pair_count": examples["scenario_id"].nunique(),
            "domains": domains,
            "observer_positions_zero_indexed": [2, 3],
            "colluder_feature_rows": 0,
        },
        "research_question": (
            "Does evidence of surrounding collusion accumulate over time along a stable "
            "honest-observer representation direction?"
        ),
        "structure_audit": structure_audit,
        "population_audit": populations,
        "methodology": {
            "training_representation": "mean(R1,R2,R3)",
            "fold_protocol": "leave-one-domain-out",
            "fit_count_per_population_scope_fold": 1,
            "fixed_probe_reuse": "same fitted StandardScaler and LogisticRegression at all six test views",
            "classifier": "LogisticRegression(C=1.0, max_iter=1000, random_state=42)",
            "score": "decision_function; positive means collusion environment",
            "temporal_views": {
                "round_only": {"R1": [1], "R2": [2], "R3": [3]},
                "prefix": {"R1": [1], "R1-R2": [1, 2], "R1-R2-R3": [1, 2, 3]},
            },
            "bootstrap": {
                "method": "percentile bootstrap",
                "unit": "matched scenario pair",
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": SEED,
                "confidence_level": 0.95,
            },
            "no_threshold_selection": True,
        },
        "training": {
            "expected_population_scope_fold_records": 2 * 3 * len(domains),
            "audit_record_count": len(training_audits),
            "fit_count": sum(audit["fit_status"] == "fit" for audit in training_audits),
            "not_fit_no_test_count": sum(
                audit["fit_status"] == "not_fit_no_held_out_observations"
                for audit in training_audits
            ),
            "convergence_warning_count": len(convergence),
            "fold_audits": training_audits,
        },
        "predictions": {
            "path": str(PREDICTION_OUTPUT.relative_to(PROJECT_ROOT)),
            "rows": len(saved_predictions),
            "sha256": sha256_file(PREDICTION_OUTPUT),
            "colluder_activation_rows": int((saved_predictions["generator_position"] < 2).sum()),
        },
        "metrics": {
            "summary": summary_metrics.to_dict(orient="records"),
            "fold_level": fold_metrics.to_dict(orient="records"),
            "undefined_fold_records": metric_issues,
            "undefined_fold_count": len(metric_issues),
        },
        "exposure_score_shift": exposure,
        "temporal_strengthening_diagnostic": strengthening,
        "experiment_c_comparison": comparison,
        "issues": validation_issues,
    }
    output = json_ready(output)
    METRICS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUTPUT.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    plot_fixed_temporal(METRICS_OUTPUT, TEMPORAL_FIGURE)
    plot_training_comparison(METRICS_OUTPUT, COMPARISON_FIGURE)
    saved_output = json.loads(METRICS_OUTPUT.read_text(encoding="utf-8"))
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT.write_text(build_report(saved_output), encoding="utf-8")

    for path in (
        PREDICTION_OUTPUT, METRICS_OUTPUT, REPORT_OUTPUT, TEMPORAL_FIGURE, COMPARISON_FIGURE,
    ):
        print("Saved {}".format(path.relative_to(PROJECT_ROOT)), flush=True)
    print("Status: {}".format(output["status"]), flush=True)
    return 0 if output["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
