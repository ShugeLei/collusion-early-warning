#!/usr/bin/env python3
"""Run Experiment C: honest-agent sentinel probe on NARCBench-Core."""

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
PHASE2_AGENT_SCORES = (
    PROJECT_ROOT / "outputs" / "predictions" / "oof_agent_temporal_scores_layer28.csv"
)
PHASE2_METRICS = PROJECT_ROOT / "outputs" / "metrics" / "temporal_detectability_layer28.json"
PIVOT_METRICS = (
    PROJECT_ROOT / "outputs" / "metrics" / "honest_observer_detection_layer28.json"
)
PREDICTION_OUTPUT = (
    PROJECT_ROOT / "outputs" / "predictions" / "oof_sentinel_scores_layer28.csv"
)
METRICS_OUTPUT = PROJECT_ROOT / "outputs" / "metrics" / "sentinel_probe_layer28.json"
REPORT_OUTPUT = PROJECT_ROOT / "outputs" / "logs" / "sentinel_probe_layer28.md"
COMPARISON_FIGURE = (
    PROJECT_ROOT / "outputs" / "figures" / "observability_comparison_layer28.png"
)
TEMPORAL_FIGURE = PROJECT_ROOT / "outputs" / "figures" / "sentinel_temporal_layer28.png"

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
    run_lodo_sentinel,
    summarize_representation_shifts,
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


def _bootstrap_surface(frame, score_column, seed=42, replicates=1000):
    scenario_ids = sorted(frame["scenario_id"].unique())
    blocks = {scenario: frame.loc[frame["scenario_id"] == scenario] for scenario in scenario_ids}
    rng = np.random.RandomState(seed)
    values = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        selected = rng.randint(0, len(scenario_ids), size=len(scenario_ids))
        sample = pd.concat(
            [blocks[scenario_ids[index]] for index in selected], ignore_index=True
        )
        values[replicate] = roc_auc_score(sample["environment_label"], sample[score_column])
    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def build_observability_comparison(phase2_scores, positions, sentinel_predictions):
    position_columns = [
        "scenario_id",
        "run_id",
        "mode",
        "agent_name",
        "generator_position",
    ]
    official = phase2_scores.merge(
        positions[position_columns],
        on=["scenario_id", "run_id", "mode", "agent_name"],
        how="left",
        validate="many_to_one",
    )
    official = official.loc[official["temporal_view"] == "round_only"].copy()
    official["environment_label"] = (official["mode"] == "collusion").astype(int)
    sentinel = sentinel_predictions.loc[
        (sentinel_predictions["analysis_population"] == "all_observers")
        & (sentinel_predictions["observer_scope"] == "both_positions")
        & (sentinel_predictions["temporal_view"] == "round_only")
    ].copy()

    records = []
    for round_number in (1, 2, 3):
        surfaces = [
            (
                "Colluders visible",
                "Official deception direction (positions 0–1)",
                official.loc[
                    (official["round"] == round_number)
                    & (official["generator_position"] < 2)
                ],
                "deception_score",
            ),
            (
                "Honest observers only",
                "Official deception direction (positions 2–3)",
                official.loc[
                    (official["round"] == round_number)
                    & (official["generator_position"] >= 2)
                ],
                "deception_score",
            ),
            (
                "Honest observers only",
                "New sentinel probe (both-position training)",
                sentinel.loc[sentinel["round"] == round_number],
                "sentinel_score",
            ),
        ]
        for surface, detector, frame, score_column in surfaces:
            auroc = float(roc_auc_score(frame["environment_label"], frame[score_column]))
            records.append({
                "observation_surface": surface,
                "detector": detector,
                "round": round_number,
                "round_or_prefix": "R{}".format(round_number),
                "observation_unit": "paired agent-position activation",
                "pooled_oof_auroc": auroc,
                "pooled_oof_auroc_bootstrap_ci_95": _bootstrap_surface(
                    frame,
                    score_column,
                    seed=SEED,
                    replicates=BOOTSTRAP_REPLICATES,
                ),
                "observation_count": len(frame),
                "scenario_pair_count": frame["scenario_id"].nunique(),
            })
    return records


def build_turn_order_diagnostic(sentinel_predictions):
    """Stratify the combined sentinel by exact within-prompt colluder exposure."""
    frame = sentinel_predictions.loc[
        (sentinel_predictions["observer_scope"] == "both_positions")
        & (sentinel_predictions["temporal_view"] == "round_only")
    ].copy()
    records = []
    for keys, stratum in frame.groupby(
        [
            "analysis_population",
            "generator_position",
            "round",
            "round_or_prefix",
            "exposure_count",
        ],
        sort=True,
    ):
        population, position, round_number, label, exposure_count = keys
        pairs = stratum.loc[stratum["mode"] == "collusion"]
        differences = pairs["paired_score_difference"].to_numpy(dtype=float)
        scenario_ids = sorted(pairs["scenario_id"].unique())
        difference_by_scenario = (
            pairs.groupby("scenario_id")["paired_score_difference"].mean().to_dict()
        )
        rng = np.random.RandomState(SEED)
        difference_bootstrap = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
        for replicate in range(BOOTSTRAP_REPLICATES):
            selected = rng.randint(0, len(scenario_ids), size=len(scenario_ids))
            difference_bootstrap[replicate] = np.mean(
                [difference_by_scenario[scenario_ids[index]] for index in selected]
            )
        records.append({
            "analysis_population": population,
            "observer_position": int(position),
            "round": int(round_number),
            "round_or_prefix": label,
            "exposure_count": int(exposure_count),
            "definition": "number of preceding public messages from generator positions 0–1",
            "observer_identity_pair_count": len(pairs),
            "scenario_pair_count": len(scenario_ids),
            "mean_turn_in_round_zero_indexed": float(pairs["turn_in_round"].mean()),
            "turn_in_round_values_zero_indexed": sorted(
                int(value) for value in pairs["turn_in_round"].unique()
            ),
            "pooled_oof_auroc": float(
                roc_auc_score(stratum["environment_label"], stratum["sentinel_score"])
            ),
            "pooled_oof_auroc_bootstrap_ci_95": _bootstrap_surface(
                stratum,
                "sentinel_score",
                seed=SEED,
                replicates=BOOTSTRAP_REPLICATES,
            ),
            "paired_mean_score_difference": float(np.mean(differences)),
            "paired_mean_score_difference_bootstrap_ci_95": [
                float(np.quantile(difference_bootstrap, 0.025)),
                float(np.quantile(difference_bootstrap, 0.975)),
            ],
            "caution": "descriptive post-stratification with small, non-random stratum sizes",
        })
    return records


def validate_predictions(predictions, expected_rows):
    issues = []
    if len(predictions) != expected_rows:
        issues.append(
            "Prediction row count {} differs from expected {}".format(
                len(predictions), expected_rows
            )
        )
    if set(predictions["generator_position"]) != {2, 3}:
        issues.append("Sentinel output contains an unexpected generator position")
    if not (predictions["role"] == "honest").all():
        issues.append("Sentinel output contains a non-honest role")
    if not (predictions["fold"] == predictions["domain"]).all():
        issues.append("A sentinel prediction is not from its held-out domain")
    if set(predictions["layer"]) != {LAYER}:
        issues.append("Sentinel output contains an unexpected activation layer")
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
        issues.append("Sentinel output contains duplicate prediction keys")
    if predictions.loc[
        predictions["analysis_population"] == "style_matched",
        "paired_prompt_style_matched",
    ].eq(False).any():
        issues.append("Style-matched predictions include a prompt-style mismatch")
    r1 = predictions.loc[predictions["round"] == 1]
    comparison = r1.pivot(
        index=[
            "analysis_population",
            "observer_scope",
            "run_id",
            "agent_name",
        ],
        columns="temporal_view",
        values="sentinel_score",
    )
    if not np.allclose(comparison["prefix"], comparison["round_only"], rtol=0.0, atol=1e-12):
        issues.append("Equivalent R1 prefix and round-only sentinel scores differ")
    return issues


def plot_sentinel_temporal(metrics_path, output_path):
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary = pd.DataFrame(metrics["metrics"]["summary"])
    scope_names = {
        "position_2": "Position 2",
        "position_3": "Position 3",
        "both_positions": "Both positions",
    }
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
                    np.arange(3),
                    pooled,
                    yerr=errors,
                    color=colors[scope],
                    marker=markers[scope],
                    linewidth=1.7,
                    capsize=3,
                    label=scope_names[scope],
                )
            labels = (
                ["R1", "R2", "R3"]
                if temporal_view == "round_only"
                else ["R1", "R1–R2", "R1–R2–R3"]
            )
            axis.set_xticks(np.arange(3), labels)
            axis.set_ylim(0.0, 1.02)
            axis.axhline(0.5, color="grey", linestyle=":", linewidth=1.0)
            axis.grid(axis="y", alpha=0.2)
            axis.set_title(
                "{} — {}".format(
                    "All observers" if population == "all_observers" else "Style-matched",
                    "round only" if temporal_view == "round_only" else "causal prefix",
                )
            )
    axes[0, 0].set_ylabel("Pooled OOF AUROC")
    axes[1, 0].set_ylabel("Pooled OOF AUROC")
    axes[1, 0].set_xlabel("Temporal representation")
    axes[1, 1].set_xlabel("Temporal representation")
    axes[0, 1].legend(fontsize=8, loc="best")
    figure.suptitle("Honest-agent sentinel temporal performance — layer 28")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_observability_comparison(metrics_path, output_path):
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(metrics["observability_comparison"])
    styles = {
        "Official deception direction (positions 0–1)": ("#c44e52", "o"),
        "Official deception direction (positions 2–3)": ("#4c72b0", "s"),
        "New sentinel probe (both-position training)": ("#55a868", "^"),
    }
    figure, axis = plt.subplots(figsize=(7.5, 4.6))
    for detector, (color, marker) in styles.items():
        detector_frame = frame.loc[frame["detector"] == detector].sort_values("round")
        pooled = detector_frame["pooled_oof_auroc"].to_numpy(dtype=float)
        intervals = np.stack(detector_frame["pooled_oof_auroc_bootstrap_ci_95"].to_numpy())
        errors = np.vstack([pooled - intervals[:, 0], intervals[:, 1] - pooled])
        axis.errorbar(
            np.arange(3),
            pooled,
            yerr=errors,
            color=color,
            marker=marker,
            linewidth=1.9,
            capsize=3,
            label=detector,
        )
    axis.set_xticks(np.arange(3), ["R1", "R2", "R3"])
    axis.set_ylim(0.0, 1.02)
    axis.axhline(0.5, color="grey", linestyle=":", linewidth=1.0)
    axis.set_ylabel("Pooled OOF AUROC")
    axis.set_xlabel("Round-only activation")
    axis.set_title("Observability comparison — NARCBench Core, layer 28")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(fontsize=8, loc="center right")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _format_interval(value, interval):
    return "{:.4f} [{:.4f}, {:.4f}]".format(value, interval[0], interval[1])


def build_report(metrics):
    population = pd.DataFrame(metrics["population_audit"]["population_scope_counts"])
    summary = pd.DataFrame(metrics["metrics"]["summary"])
    folds = pd.DataFrame(metrics["metrics"]["fold_level"])
    comparison = pd.DataFrame(metrics["observability_comparison"])
    turn_order = pd.DataFrame(metrics["turn_order_exposure_diagnostic"])
    representation = metrics["representation_shift_diagnostic"]["summary"]
    lines = [
        "# Experiment C — Honest-Agent Sentinel Probe (Layer 28)",
        "",
        "- Status: **{}**".format(metrics["status"]),
        "- Model/data: {} / NARCBench Core".format(metrics["model"]),
        "- Evaluation: leave-one-domain-out; 14 expected folds",
        "- Labels: control honest observer = 0; collusion-environment honest observer = 1",
        "- Features: positions 2–3 only; no colluder activation enters any fit or prediction",
        "- Classifier: training-fold StandardScaler + LogisticRegression(C=1.0, max_iter=1000, random_state=42)",
        "- Scope: diagnostic only; no thresholds or E2 intervention analysis",
        "",
        "## Explicit deviation from the original experiment plan",
        "",
        "The original temporal experiment kept the colluder-vs-honest deception direction fixed. "
        "Experiment C is explicitly authorized to train a new honest-environment sentinel separately "
        "for each round/prefix, position scope, and style population. Domain holdout remains unchanged; "
        "held-out data are never used for scaling, training, model selection, or thresholds.",
        "",
        "## Retained samples",
        "",
        "| Population | Scope | Available observer pairs | Retained | Style-mismatch exclusions | Scenarios | Missing domains |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for _, row in population.iterrows():
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                row["analysis_population"],
                row["observer_scope"],
                row["available_observer_identity_pairs"],
                row["retained_observer_identity_pairs"],
                row["excluded_style_mismatched_pairs"],
                row["retained_scenario_count"],
                ", ".join(row["missing_domains"]) if row["missing_domains"] else "none",
            )
        )
    lines.extend([
        "",
        "The style-matched analysis excludes every mismatched identity/position pair explicitly; "
        "it does not require both observers in a scenario to match. Scenario-level bootstrap samples "
        "all retained observer pairs for the selected scenario together.",
        "",
        "The sensitivity subset is small: position 2 retains 15 pairs across 8 domains, position 3 "
        "retains 15 across 11, and the combined model covers 22 scenarios across 12 domains. Across "
        "all temporal configurations this creates 66 explicitly undefined fold records, and several "
        "defined folds contain only one pair. Treat pooled paired-bootstrap uncertainty as primary and "
        "do not over-interpret the style-matched fold macro means.",
        "",
        "## Primary result — round-only activations",
        "",
        "| Population | Scope | Round | Pooled OOF AUROC (95% CI) | Fold mean | Fold SD | Valid/expected folds | Paired mean Δ (95% CI) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    primary = summary.loc[summary["temporal_view"] == "round_only"]
    for _, row in primary.sort_values(
        ["analysis_population", "observer_scope", "round"]
    ).iterrows():
        lines.append(
            "| {} | {} | {} | {} | {:.4f} | {:.4f} | {}/{} | {} |".format(
                row["analysis_population"],
                row["observer_scope"],
                row["round_or_prefix"],
                _format_interval(
                    row["pooled_oof_auroc"], row["pooled_oof_auroc_bootstrap_ci_95"]
                ),
                row["fold_macro_mean_auroc"],
                row["fold_population_sd_auroc"],
                row["fold_count_valid"],
                row["fold_count_expected"],
                _format_interval(
                    row["paired_mean_score_difference"],
                    row["paired_mean_score_difference_bootstrap_ci_95"],
                ),
            )
        )

    lines.extend([
        "",
        "## Secondary result — causal prefixes",
        "",
        "| Population | Scope | Prefix | Pooled OOF AUROC (95% CI) | Fold mean | Fold SD | Valid/expected folds | Paired mean Δ (95% CI) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    prefixes = summary.loc[summary["temporal_view"] == "prefix"]
    for _, row in prefixes.sort_values(
        ["analysis_population", "observer_scope", "round"]
    ).iterrows():
        lines.append(
            "| {} | {} | {} | {} | {:.4f} | {:.4f} | {}/{} | {} |".format(
                row["analysis_population"],
                row["observer_scope"],
                row["round_or_prefix"],
                _format_interval(
                    row["pooled_oof_auroc"], row["pooled_oof_auroc_bootstrap_ci_95"]
                ),
                row["fold_macro_mean_auroc"],
                row["fold_population_sd_auroc"],
                row["fold_count_valid"],
                row["fold_count_expected"],
                _format_interval(
                    row["paired_mean_score_difference"],
                    row["paired_mean_score_difference_bootstrap_ci_95"],
                ),
            )
        )

    lines.extend([
        "",
        "## Leave-one-domain-out fold AUROCs",
        "",
        "Undefined folds are retained below rather than removed from the record.",
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
        lines.append(
            "- `{}` / `{}` / `{}` / `{}`: {}".format(
                population_name, scope, view, time_label, ", ".join(values)
            )
        )

    lines.extend([
        "",
        "## Exact turn-order / exposure diagnostic",
        "",
        "`exposure_count` is reconstructed from the generator and verified released row order. It "
        "counts preceding public messages from positions 0–1; those messages are collusive only in "
        "the collusion condition. The paired control has the same source positions and turn order. "
        "The table uses the both-position sentinel and is descriptive post-stratification, not a "
        "separately trained exposure-specific classifier.",
        "",
        "| Population | Observer position | R1 exposure | Pairs | Mean turn index | Pooled AUROC (95% CI) | Paired mean Δ (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    r1_turn_order = turn_order.loc[turn_order["round"] == 1]
    for _, row in r1_turn_order.sort_values(
        ["analysis_population", "observer_position", "exposure_count"]
    ).iterrows():
        lines.append(
            "| {} | {} | {} | {} | {:.2f} | {} | {} |".format(
                row["analysis_population"],
                int(row["observer_position"]),
                int(row["exposure_count"]),
                int(row["observer_identity_pair_count"]),
                row["mean_turn_in_round_zero_indexed"],
                _format_interval(
                    row["pooled_oof_auroc"], row["pooled_oof_auroc_bootstrap_ci_95"]
                ),
                _format_interval(
                    row["paired_mean_score_difference"],
                    row["paired_mean_score_difference_bootstrap_ci_95"],
                ),
            )
        )
    zero_exposure_counts = r1_turn_order.loc[
        r1_turn_order["exposure_count"] == 0
    ].set_index(["analysis_population", "observer_position"])[
        "observer_identity_pair_count"
    ]
    lines.extend([
        "",
        "For all observers at R1, position 2 contributes {} zero-exposure pairs versus {} for "
        "position 3; the style-matched subset contributes {} versus {}. Zero-exposure position-2 "
        "performance is near chance, while one- and two-exposure strata are markedly higher. This "
        "is consistent with an exposure/propagation contribution to the position gap, but the strata "
        "are too small and non-random for a causal claim.".format(
            int(zero_exposure_counts.loc[("all_observers", 2)]),
            int(zero_exposure_counts.loc[("all_observers", 3)]),
            int(zero_exposure_counts.loc[("style_matched", 2)]),
            int(zero_exposure_counts.loc[("style_matched", 3)]),
        ),
        "",
        "## Representation-shift diagnostic",
        "",
        "These are descriptive paired activation shifts, not deployment-classifier metrics. L2 and "
        "cosine use raw temporal representations. Projection uses `(collusion - control) / "
        "training-scaler.scale_` dotted with the held-out-fold position-specific unit coefficient.",
        "",
        "| Population | Position | Round | n | Mean L2 | Mean cosine | Mean sentinel projection (95% CI) |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for row in representation:
        projection = row["delta_projection_on_unit_sentinel_direction"]
        lines.append(
            "| {} | {} | {} | {} | {:.4f} | {:.4f} | {} |".format(
                row["analysis_population"],
                row["observer_scope"],
                row["round_or_prefix"],
                row["observer_identity_pair_count"],
                row["delta_activation_l2"]["mean"],
                row["paired_activation_cosine_similarity"]["mean"],
                _format_interval(projection["mean"], projection["mean_bootstrap_ci_95"]),
            )
        )
    lines.extend([
        "",
        "Raw paired distance does not grow with detectability: for all observers, mean L2 decreases "
        "from R1 to R3 while cosine similarity rises slightly. The learned-direction projection, "
        "however, increases for both positions. Descriptively, the classifier is exploiting a "
        "directional change rather than a growing global representation distance; this still does "
        "not identify the underlying mechanism.",
    ])

    lines.extend([
        "",
        "## Observability comparison",
        "",
        "All three rows below use paired agent-position observations so the surfaces are directly "
        "comparable. The two official-direction rows reuse Phase 2 scores without refitting.",
        "",
        "| Observation Surface | Detector | R1 | R2 | R3 |",
        "|---|---|---:|---:|---:|",
    ])
    for (surface, detector), frame in comparison.groupby(
        ["observation_surface", "detector"], sort=False
    ):
        by_round = frame.set_index("round")
        lines.append(
            "| {} | {} | {:.4f} | {:.4f} | {:.4f} |".format(
                surface,
                detector,
                by_round.loc[1, "pooled_oof_auroc"],
                by_round.loc[2, "pooled_oof_auroc"],
                by_round.loc[3, "pooled_oof_auroc"],
            )
        )

    all_combined = primary.loc[
        (primary["analysis_population"] == "all_observers")
        & (primary["observer_scope"] == "both_positions")
    ].set_index("round")
    style_combined = primary.loc[
        (primary["analysis_population"] == "style_matched")
        & (primary["observer_scope"] == "both_positions")
    ].set_index("round")
    position_2 = primary.loc[
        (primary["analysis_population"] == "all_observers")
        & (primary["observer_scope"] == "position_2")
    ].set_index("round")
    position_3 = primary.loc[
        (primary["analysis_population"] == "all_observers")
        & (primary["observer_scope"] == "position_3")
    ].set_index("round")
    reliable_all = any(
        all_combined.loc[round_number, "pooled_oof_auroc_bootstrap_ci_95"][0] > 0.5
        for round_number in (1, 2, 3)
    )
    reliable_style = any(
        style_combined.loc[round_number, "pooled_oof_auroc_bootstrap_ci_95"][0] > 0.5
        for round_number in (1, 2, 3)
    )
    position_3_advantage = max(
        position_3.loc[round_number, "pooled_oof_auroc"]
        - position_2.loc[round_number, "pooled_oof_auroc"]
        for round_number in (1, 2, 3)
    )
    lines.extend(["", "## Interpretation", ""])
    if reliable_all and reliable_style:
        lines.append(
            "Outcome A is supported at least at one temporal point in both populations: honest-agent "
            "activations contain collusion-environment information not captured by the official "
            "colluder-trained deception direction. This is predictive evidence, not a mechanistic claim."
        )
    elif reliable_all and not reliable_style:
        lines.append(
            "The all-observer analysis is above chance at at least one temporal point, but the "
            "style-matched sensitivity is not reliably above chance. This matches Outcome C and is "
            "more consistent with prompt-style confounding than robust sentinel evidence."
        )
    elif not reliable_all and not reliable_style:
        lines.append(
            "Neither the all-observer nor style-matched combined analysis is reliably above chance. "
            "This matches Outcome B: on NARCBench Core, the reliable activation signal remains "
            "concentrated in agents carrying the collusive objective."
        )
    else:
        lines.append(
            "Only the smaller style-matched sensitivity is reliably above chance. Given its reduced "
            "fold coverage and sample size, treat this as an unstable diagnostic requiring replication."
        )
    if position_3_advantage >= 0.10:
        lines.append(
            "Position 3 exceeds position 2 by at least 0.10 at one round, which is consistent with "
            "Outcome D as a possible exposure/order effect. The exact turn-order diagnostic above "
            "supports that cautious interpretation, but does not establish propagation causally."
        )
    else:
        lines.append(
            "Position 3 does not show a ≥0.10 pooled-AUROC advantage over position 2 at any round, so "
            "there is no strong position-based propagation pattern by that descriptive criterion."
        )
    lines.extend([
        "",
        "AUROC and learned-direction projection do not identify a mechanism. Prompt construction, "
        "response content, scenario semantics, and other correlated representation shifts remain "
        "possible explanations.",
        "",
        "No intervention thresholds, TFIR, PIC, or E2 analysis were run.",
        "",
    ])
    return "\n".join(lines)


def main():
    metadata = json.loads((CORE_DIR / "metadata_gen.json").read_text(encoding="utf-8"))
    phase2_metrics = json.loads(PHASE2_METRICS.read_text(encoding="utf-8"))
    pivot_metrics = json.loads(PIVOT_METRICS.read_text(encoding="utf-8"))
    phase2_scores = pd.read_csv(PHASE2_AGENT_SCORES)
    if phase2_metrics["status"] != "PASS" or pivot_metrics["status"] != "PASS":
        raise RuntimeError("Required Phase 2 source artifacts are not PASS")
    if sha256_file(PHASE2_AGENT_SCORES) != phase2_metrics["predictions"]["agent_sha256"]:
        raise RuntimeError("Phase 2 fixed-probe source CSV hash mismatch")

    name_pool = load_name_pool(CORE_SOURCE)
    positions, turns, structure_audit = audit_core_structure(
        metadata, name_pool, source_evidence(CORE_SOURCE)
    )
    if structure_audit["status"] != "PASS":
        raise RuntimeError("Core structure audit failed: {}".format(structure_audit["issues"]))

    archive = np.load(CORE_DIR / "activations_gen.npz", allow_pickle=False)
    try:
        activations = np.asarray(archive["layer_{}".format(LAYER)])
        temporal_index, run_metadata = build_public_discussion_index(metadata, activations)
        examples = build_observer_examples(
            temporal_index, run_metadata, positions, turns
        )
        examples["layer"] = LAYER
        domains = sorted(examples["domain"].unique())
        predictions, diagnostics, training_audits = run_lodo_sentinel(examples, domains)
    finally:
        archive.close()

    predictions = attach_paired_diagnostics(predictions, diagnostics)
    prediction_columns = [column for column in predictions.columns if column != "activation"]
    predictions = predictions[prediction_columns].sort_values(
        [
            "analysis_population",
            "observer_scope",
            "fold",
            "scenario_id",
            "mode",
            "agent_name",
            "temporal_view",
            "round",
        ]
    )
    population_counts = population_audit(examples)
    expected_rows = 12 * sum(
        row["retained_observer_identity_pairs"]
        for row in population_counts["population_scope_counts"]
    )
    validation_issues = validate_predictions(predictions, expected_rows)
    convergence_issues = [
        {
            "analysis_population": audit["analysis_population"],
            "observer_scope": audit["observer_scope"],
            "temporal_view": audit["temporal_view"],
            "round_or_prefix": audit["round_or_prefix"],
            "held_out_domain": audit["held_out_domain"],
            "warnings": audit.get("convergence_warnings", []),
        }
        for audit in training_audits
        if audit.get("convergence_warnings")
    ]
    if convergence_issues:
        validation_issues.append("One or more sentinel fits emitted a convergence warning")

    # Raw predictions are persisted before any reported metric or figure is computed.
    PREDICTION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTION_OUTPUT, index=False)
    saved_predictions = pd.read_csv(PREDICTION_OUTPUT)
    fold_metrics, summary_metrics, metric_issues = calculate_sentinel_metrics(
        saved_predictions,
        domains,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        seed=SEED,
    )
    representation_summary = summarize_representation_shifts(
        diagnostics,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        seed=SEED,
    )
    comparison = build_observability_comparison(
        phase2_scores, positions, saved_predictions
    )
    turn_order_diagnostic = build_turn_order_diagnostic(saved_predictions)

    top_level_issues = validation_issues + convergence_issues
    output = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not top_level_issues else "ISSUES",
        "scope": "Experiment C honest-agent sentinel probe only; no threshold or E2 analysis.",
        "sources": {
            "upstream_repository": "https://github.com/aaronrose227/narcbench",
            "upstream_git_commit": git_commit(UPSTREAM_ROOT),
            "upstream_source_modified": False,
            "dataset_repository": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "metadata_sha256": sha256_file(CORE_DIR / "metadata_gen.json"),
            "activations_sha256": sha256_file(CORE_DIR / "activations_gen.npz"),
            "phase2_agent_scores_path": str(PHASE2_AGENT_SCORES.relative_to(PROJECT_ROOT)),
            "phase2_agent_scores_sha256": sha256_file(PHASE2_AGENT_SCORES),
            "phase2_metrics_sha256": sha256_file(PHASE2_METRICS),
            "pivot_metrics_sha256": sha256_file(PIVOT_METRICS),
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
        "plan_deviation": {
            "authorized": True,
            "description": (
                "Unlike the original fixed deception-direction temporal experiment, Experiment C "
                "trains a new environment-label probe separately for each temporal representation, "
                "observer scope, and style population, as explicitly requested."
            ),
            "unchanged_safeguards": [
                "leave-one-domain-out evaluation",
                "training-only scaling",
                "layer 28 fixed before evaluation",
                "seed 42",
                "no threshold selection",
                "no colluder activations",
            ],
        },
        "structure_audit": structure_audit,
        "population_audit": population_counts,
        "methodology": {
            "labels": {
                "0": "honest observer in matched control environment",
                "1": "same honest observer identity/position in collusion environment",
            },
            "fold_protocol": "leave-one-domain-out",
            "training_unit": "honest agent temporal representation",
            "features": "layer-28 final-token activation; round-only or causal mean prefix",
            "temporal_models": "separate classifier fit for each round/prefix",
            "classifier_pipeline": {
                "scaler": "StandardScaler() fit on non-held-out rows only",
                "classifier": "LogisticRegression(C=1.0, max_iter=1000, random_state=42)",
                "score": "decision_function; positive means collusion environment",
            },
            "observer_scopes": {
                "position_2": "seeded generator position 2 only",
                "position_3": "seeded generator position 3 only",
                "both_positions": "positions 2 and 3 pooled as agent observations",
            },
            "style_sensitivity": (
                "Retain an observer identity/position iff its standard/aggressive honest system-prompt "
                "style is identical in paired collusion/control runs."
            ),
            "bootstrap": {
                "method": "percentile bootstrap",
                "unit": "matched scenario pair",
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": SEED,
                "confidence_level": 0.95,
            },
            "no_hyperparameter_or_threshold_selection": True,
        },
        "training": {
            "expected_configuration_fold_records": 2 * 3 * 2 * 3 * len(domains),
            "audit_record_count": len(training_audits),
            "fit_count": sum(audit["fit_status"] == "fit" for audit in training_audits),
            "not_fit_no_test_count": sum(
                audit["fit_status"] == "not_fit_no_held_out_observations"
                for audit in training_audits
            ),
            "convergence_warning_count": len(convergence_issues),
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
        "representation_shift_diagnostic": {
            "raw_pair_record_count": len(diagnostics),
            "scope": "round-only summary uses position-specific OOF directions",
            "summary": representation_summary,
            "mechanistic_evidence": False,
        },
        "turn_order_exposure_diagnostic": turn_order_diagnostic,
        "observability_comparison": comparison,
        "issues": top_level_issues,
    }
    output = json_ready(output)
    METRICS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUTPUT.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    # Both figures reload the saved JSON; no transient fit result is plotted.
    plot_sentinel_temporal(METRICS_OUTPUT, TEMPORAL_FIGURE)
    plot_observability_comparison(METRICS_OUTPUT, COMPARISON_FIGURE)
    saved_output = json.loads(METRICS_OUTPUT.read_text(encoding="utf-8"))
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT.write_text(build_report(saved_output), encoding="utf-8")

    for path in (
        PREDICTION_OUTPUT,
        METRICS_OUTPUT,
        REPORT_OUTPUT,
        COMPARISON_FIGURE,
        TEMPORAL_FIGURE,
    ):
        print("Saved {}".format(path.relative_to(PROJECT_ROOT)), flush=True)
    print("Status: {}".format(output["status"]), flush=True)
    return 0 if output["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
