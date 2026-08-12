#!/usr/bin/env python3
"""Run the trusted-observer / partial-white-box NARCBench diagnostic."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "external" / "narcbench"
CORE_SOURCE = UPSTREAM_ROOT / "generation" / "core.py"
CORE_DIR = PROJECT_ROOT / "data" / "narcbench" / "activations" / "qwen3_32b" / "core"
PHASE2_AGENT_SCORES = (
    PROJECT_ROOT / "outputs" / "predictions" / "oof_agent_temporal_scores_layer28.csv"
)
PHASE2_METRICS = PROJECT_ROOT / "outputs" / "metrics" / "temporal_detectability_layer28.json"
PREDICTION_OUTPUT = (
    PROJECT_ROOT / "outputs" / "predictions" / "oof_honest_observer_scores_layer28.csv"
)
METRICS_OUTPUT = (
    PROJECT_ROOT / "outputs" / "metrics" / "honest_observer_detection_layer28.json"
)
ROLE_FIGURE = (
    PROJECT_ROOT / "outputs" / "figures" / "role_source_decomposition_layer28.png"
)
OBSERVER_FIGURE = (
    PROJECT_ROOT / "outputs" / "figures" / "honest_observer_detectability_layer28.png"
)
REPORT_OUTPUT = (
    PROJECT_ROOT / "outputs" / "logs" / "honest_observer_detection_layer28.md"
)

LAYER = 28
BOOTSTRAP_REPLICATES = 1000
SEED = 42
MODEL_ID = "Qwen/Qwen3-32B-AWQ"
DATASET_ID = "aaronrose227/narcbench"
DATASET_REVISION = "19cb14b269e052b19ba6d2bd2f4519afd3b10c57"


sys.path.insert(0, str(PROJECT_ROOT))
from src.honest_observer_detection import (  # noqa: E402
    OBSERVER_METHODS,
    add_run_risks_to_observers,
    aggregate_observer_risk,
    audit_core_structure,
    calculate_exposure_diagnostic,
    calculate_observer_metrics,
    enrich_fixed_probe_scores,
    load_name_pool,
    role_source_summary,
    select_honest_observer_scores,
    source_evidence,
)


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


def validate_phase2_scores(scores, phase2_metrics):
    issues = []
    expected_hash = phase2_metrics["predictions"]["agent_sha256"]
    actual_hash = sha256_file(PHASE2_AGENT_SCORES)
    if phase2_metrics["status"] != "PASS":
        issues.append("Phase 2 metrics do not have PASS status")
    if actual_hash != expected_hash:
        issues.append("Saved Phase 2 agent predictions do not match their recorded hash")
    if set(scores["layer"]) != {LAYER}:
        issues.append("Phase 2 scores contain an unexpected layer")
    if not (scores["fold"] == scores["domain"]).all():
        issues.append("Some Phase 2 scores are not out-of-fold")
    fingerprint_counts = scores.groupby("fold")["probe_fingerprint"].nunique()
    if (fingerprint_counts != 1).any():
        issues.append("A fold uses more than one probe fingerprint")
    expected_grid = scores.groupby(
        ["run_id", "temporal_view", "round"], dropna=False
    )["agent_name"].nunique()
    if not (expected_grid == 4).all():
        issues.append("Phase 2 scores do not have four agents per run/view/time")
    duplicate_keys = scores.duplicated(
        ["run_id", "agent_name", "temporal_view", "round", "layer"]
    )
    if duplicate_keys.any():
        issues.append("Phase 2 scores contain duplicate agent/view/time keys")
    r1 = scores.loc[scores["round"] == 1]
    r1_pivot = r1.pivot(
        index=["run_id", "agent_name"], columns="temporal_view", values="deception_score"
    )
    if not np.array_equal(
        r1_pivot["prefix"].to_numpy(), r1_pivot["round_only"].to_numpy()
    ):
        issues.append("R1 prefix and round-only scores are not identical")
    return {
        "status": "PASS" if not issues else "FAIL",
        "phase2_metrics_status": phase2_metrics["status"],
        "path": str(PHASE2_AGENT_SCORES.relative_to(PROJECT_ROOT)),
        "rows": len(scores),
        "recorded_sha256": expected_hash,
        "observed_sha256": actual_hash,
        "fold_count": scores["fold"].nunique(),
        "unique_probe_fingerprints": scores["probe_fingerprint"].nunique(),
        "one_fixed_probe_per_fold_across_all_views": bool((fingerprint_counts == 1).all()),
        "probe_retrained_for_this_experiment": False,
        "issues": issues,
    }


def paired_role_contrasts(round_frame):
    records = []
    frame = round_frame.copy()
    frame["position_set"] = np.where(
        frame["generator_position"] < 2, "positions_0_1", "positions_2_3"
    )
    paired = frame.pivot(
        index=["scenario_id", "domain", "agent_name", "generator_position", "round", "position_set"],
        columns="mode",
        values="deception_score",
    ).reset_index()
    paired["score_difference"] = paired["collusion"] - paired["control"]
    for (round_number, position_set), values in paired.groupby(["round", "position_set"]):
        differences = values["score_difference"].to_numpy(dtype=float)
        records.append({
            "round": int(round_number),
            "round_or_prefix": "R{}".format(round_number),
            "position_set": position_set,
            "paired_agent_observation_count": len(values),
            "scenario_count": values["scenario_id"].nunique(),
            "mean_collusion_minus_control": float(np.mean(differences)),
            "sample_sd": float(np.std(differences, ddof=1)),
            "median_collusion_minus_control": float(np.median(differences)),
            "minimum": float(np.min(differences)),
            "maximum": float(np.max(differences)),
        })
    return pd.DataFrame.from_records(records)


def plot_role_decomposition(enriched_scores, output_path):
    frame = enriched_scores.loc[enriched_scores["temporal_view"] == "round_only"].copy()
    group_rules = [
        ((frame["mode"] == "collusion") & (frame["generator_position"] < 2), "Colluders\nin collusion"),
        ((frame["mode"] == "collusion") & (frame["generator_position"] >= 2), "Observers\nin collusion"),
        ((frame["mode"] == "control") & (frame["generator_position"] < 2), "Control\npos. 0–1"),
        ((frame["mode"] == "control") & (frame["generator_position"] >= 2), "Control\npos. 2–3"),
    ]
    frame["plot_group"] = "unexpected"
    for mask, label in group_rules:
        frame.loc[mask, "plot_group"] = label
    labels = [label for _, label in group_rules]
    colors = ["#c44e52", "#4c72b0", "#dd8452", "#55a868"]
    rng = np.random.RandomState(SEED)
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), sharey=True)
    for axis, round_number in zip(axes, (1, 2, 3)):
        round_frame = frame.loc[frame["round"] == round_number]
        distributions = [
            round_frame.loc[round_frame["plot_group"] == label, "deception_score"].to_numpy()
            for label in labels
        ]
        boxes = axis.boxplot(
            distributions,
            tick_labels=labels,
            patch_artist=True,
            showfliers=False,
            widths=0.62,
        )
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.38)
        for index, (scores, color) in enumerate(zip(distributions, colors), start=1):
            jitter = rng.uniform(-0.14, 0.14, size=len(scores))
            axis.scatter(
                np.full(len(scores), index) + jitter,
                scores,
                s=9,
                alpha=0.48,
                color=color,
                linewidths=0,
            )
        axis.axhline(0.0, color="grey", linestyle=":", linewidth=0.9)
        axis.set_title("Round {}".format(round_number))
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("Fixed deception-probe score")
    figure.suptitle("Role-source decomposition — NARCBench Core, Qwen3-32B, layer 28")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_observer_detectability(summary, output_path):
    names = {
        "mean_observer_suspicion": "Mean Observer Suspicion",
        "peak_observer_suspicion": "Peak Observer Suspicion",
    }
    colors = {
        "mean_observer_suspicion": "#4c72b0",
        "peak_observer_suspicion": "#c44e52",
    }
    markers = {"mean_observer_suspicion": "o", "peak_observer_suspicion": "s"}
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    for axis, temporal_view in zip(axes, ("round_only", "prefix")):
        view = summary.loc[summary["temporal_view"] == temporal_view]
        labels = ["R1", "R2", "R3"] if temporal_view == "round_only" else ["R1", "R1–R2", "R1–R2–R3"]
        for method in OBSERVER_METHODS:
            method_frame = view.loc[view["method"] == method].sort_values("round")
            x = np.arange(3)
            pooled = method_frame["pooled_oof_auroc"].to_numpy(dtype=float)
            intervals = np.stack(method_frame["pooled_oof_auroc_bootstrap_ci_95"].to_numpy())
            errors = np.vstack([pooled - intervals[:, 0], intervals[:, 1] - pooled])
            axis.errorbar(
                x,
                pooled,
                yerr=errors,
                color=colors[method],
                marker=markers[method],
                linewidth=1.8,
                capsize=3,
                label=names[method] + " (pooled OOF, 95% CI)",
            )
            axis.plot(
                x,
                method_frame["fold_macro_mean_auroc"],
                color=colors[method],
                linestyle="--",
                linewidth=1.0,
                alpha=0.75,
                label=names[method] + " (fold macro mean)",
            )
        axis.axhline(0.5, color="grey", linestyle=":", linewidth=1.0)
        axis.set_xticks(np.arange(3), labels)
        axis.set_ylim(0.0, 1.02)
        axis.set_title("Round only" if temporal_view == "round_only" else "Causal prefix")
        axis.set_xlabel("Temporal view")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Collusion-environment AUROC")
    axes[1].legend(fontsize=7.2, loc="best")
    figure.suptitle("Honest-observer detectability — layer 28")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_report(output):
    summary = pd.DataFrame(output["experiment_b"]["summary"])
    fold_metrics = pd.DataFrame(output["experiment_b"]["fold_level"])
    role = pd.DataFrame(output["experiment_a"]["distribution_summary"])
    role_contrasts = pd.DataFrame(output["experiment_a"]["paired_position_contrasts"])
    exposure = pd.DataFrame(output["exposure_diagnostic"]["score_shift_summary"])
    assumptions = output["assumption_verification"]["assumptions"]
    lines = [
        "# Trusted Observer / Partial White-Box Collusion Detection",
        "",
        "- Status: **{}**".format(output["status"]),
        "- Scope: benchmark diagnostic only; no intervention threshold or E2 analysis",
        "- Model/data: {} / NARCBench Core, layer {}".format(output["model"], LAYER),
        "- Probe: saved Phase 2 leave-one-domain-out fixed deception probes; no retraining",
        "- Matched scenarios: {} across {} domains".format(
            output["dataset"]["scenario_pair_count"], len(output["dataset"]["domains"])
        ),
        "",
        "## Assumption verification",
        "",
        "No violations were found. Generator positions are zero-indexed.",
        "",
        "| Assumption | Result |",
        "|---|---|",
    ]
    assumption_labels = {
        "positions_0_1_are_colluders_in_collusion": "Collusion positions 0–1 are colluders",
        "positions_2_3_are_honest_in_collusion": "Collusion positions 2–3 are honest",
        "all_control_positions_are_honest": "All control positions are honest",
        "paired_conditions_use_same_identities": "Paired identities are identical",
        "paired_public_order_is_identical": "Paired public speaking order is identical",
        "preceding_public_messages_are_in_prompt_history": "Each turn receives preceding public messages",
        "metadata_order_matches_generator": "Released metadata preserves generator order",
        "exposure_count_supported": "Exact exposure count is supported",
    }
    for key, label in assumption_labels.items():
        lines.append("| {} | {} |".format(label, "PASS" if assumptions[key] else "FAIL"))

    lines.extend([
        "",
        "The source constructs the seeded four-name list once per scenario, assigns its first two "
        "members as colluders only in the collusion condition, and shuffles public order from a "
        "scenario-only seed. It builds each prompt from the accumulated public-message list before "
        "appending the current response. Every released pair matches that construction exactly.",
        "",
        "`exposure_count` therefore counts earlier messages from positions 0–1. These are actual "
        "colluder messages in collusion runs and matched honest source-position messages in controls; "
        "the prediction CSV distinguishes the two with `exposure_messages_are_collusive`.",
        "",
        "### Pairing caveat: honest prompt style",
        "",
        "The five requested assumptions pass, but identity/role matching is not full prompt matching. "
        "The generator draws collusion-run honest styles and control-run styles from separate arrays. "
        "Of {} observer identity pairs, {} use different standard/aggressive system-prompt styles; "
        "{} of 31 scenarios have at least one mismatch. This is recorded rather than silently "
        "treated as a perfectly prompt-matched design.".format(
            output["assumption_verification"]["observer_prompt_style_audit"]["observer_identity_pairs"],
            output["assumption_verification"]["observer_prompt_style_audit"]["different_prompt_style_pairs"],
            output["assumption_verification"]["observer_prompt_style_audit"]["scenario_pairs_with_at_least_one_observer_style_mismatch"],
        ),
        "",
        "## Experiment A — role-source decomposition",
        "",
        "| Round | Source group | n | Mean | SD | Median | Q1 | Q3 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    role_order = [
        "colluders_in_collusion",
        "honest_observers_in_collusion",
        "paired_control_colluder_positions",
        "paired_control_observer_positions",
    ]
    role = role.loc[role["role_source_group"].isin(role_order)].copy()
    role["group_order"] = role["role_source_group"].map({value: index for index, value in enumerate(role_order)})
    for _, row in role.sort_values(["round", "group_order"]).iterrows():
        lines.append(
            "| R{} | {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                int(row["round"]),
                row["role_source_group"],
                int(row["observation_count"]),
                row["mean"],
                row["sample_sd"],
                row["median"],
                row["q1"],
                row["q3"],
            )
        )

    r1_contrasts = role_contrasts.loc[role_contrasts["round"] == 1].set_index("position_set")
    lines.extend([
        "",
        "At R1, the mean collusion-minus-control shift is **{:.4f}** for positions 0–1 "
        "but **{:.4f}** for positions 2–3. The saturated R1 system detector is therefore "
        "concentrated in agents carrying the explicit colluder role, not in honest observers.".format(
            r1_contrasts.loc["positions_0_1", "mean_collusion_minus_control"],
            r1_contrasts.loc["positions_2_3", "mean_collusion_minus_control"],
        ),
    ])

    lines.extend([
        "",
        "## Experiment B — honest-observer detection",
        "",
        "Only positions 2–3 enter either run-risk score.",
        "",
        "| View | Time | Method | Pooled OOF AUROC (95% paired-bootstrap CI) | Fold mean | Fold SD | Paired mean Δ (95% CI) |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for _, row in summary.sort_values(["temporal_view", "round", "method"]).iterrows():
        auc_ci = row["pooled_oof_auroc_bootstrap_ci_95"]
        diff_ci = row["paired_mean_score_difference_bootstrap_ci_95"]
        lines.append(
            "| {} | {} | {} | {:.4f} [{:.4f}, {:.4f}] | {:.4f} | {:.4f} | {:.4f} [{:.4f}, {:.4f}] |".format(
                row["temporal_view"],
                row["round_or_prefix"],
                row["method"],
                row["pooled_oof_auroc"],
                auc_ci[0],
                auc_ci[1],
                row["fold_macro_mean_auroc"],
                row["fold_population_sd_auroc"],
                row["paired_mean_score_difference"],
                diff_ci[0],
                diff_ci[1],
            )
        )

    lines.extend([
        "",
        "All 14 fold rows are retained for every configuration. Undefined/single-class fold count: {}.".format(
            len(output["experiment_b"]["metric_issues"])
        ),
        "",
        "### Leave-one-domain-out fold AUROCs",
        "",
    ])
    for keys, config in fold_metrics.groupby(
        ["temporal_view", "round", "round_or_prefix", "method"], sort=True
    ):
        temporal_view, _, time_label, method = keys
        values = []
        for _, fold in config.sort_values("fold").iterrows():
            rendered = "undefined ({})".format(fold["metric_issue"]) if pd.isna(fold["auroc"]) else "{:.4f}".format(fold["auroc"])
            values.append("{}={}".format(fold["fold"], rendered))
        lines.append(
            "- `{}` / `{}` / `{}`: {}".format(
                temporal_view, time_label, method, ", ".join(values)
            )
        )

    lines.extend([
        "",
        "## Exposure diagnostic (round-only activations)",
        "",
        "Prefix scores average activations recorded at several exposure levels, so this diagnostic "
        "uses only the activation from the stated round. Each shift is collusion minus its exact "
        "matched control for the same scenario and observer identity.",
        "",
        "| Round | Prior position-0/1 messages | Observer pairs | Mean score shift | 95% scenario-bootstrap CI |",
        "|---|---:|---:|---:|---:|",
    ])
    exposure = exposure.loc[exposure["grouping"] == "round_and_exposure"]
    for _, row in exposure.sort_values(["round", "exposure_count"]).iterrows():
        ci = row["paired_mean_shift_bootstrap_ci_95"]
        lines.append(
            "| R{} | {} | {} | {:.4f} | [{:.4f}, {:.4f}] |".format(
                int(row["round"]),
                int(row["exposure_count"]),
                int(row["observer_pair_count"]),
                row["paired_mean_score_shift"],
                ci[0],
                ci[1],
            )
        )

    r1 = summary.loc[
        (summary["temporal_view"] == "round_only") & (summary["round"] == 1)
    ].set_index("method")
    r3 = summary.loc[
        (summary["temporal_view"] == "round_only") & (summary["round"] == 3)
    ].set_index("method")
    lines.extend([
        "",
        "## Interpretation",
        "",
    ])
    r1_values = r1["pooled_oof_auroc"]
    r3_values = r3["pooled_oof_auroc"]
    if (r1_values >= 0.85).all():
        lines.append(
            "Both observer-only methods are already very high at R1. This matches interpretation D, "
            "so the exposure strata above—not an assumption of pre-turn innocence—must be consulted."
        )
    elif (r1_values.between(0.4, 0.6)).all() and (r3_values.between(0.4, 0.6)).all():
        lines.append(
            "Observer-only detection remains near chance from R1 through R3. This supports "
            "interpretation C: in this benchmark, the fixed activation defense appears to depend "
            "primarily on inspecting agents that carry collusive intent."
        )
    elif (r3_values >= 0.7).any() and (r3_values > r1_values + 0.1).any():
        lines.append(
            "At least one observer-only method rises substantially above chance and improves by "
            "more than 0.10 from R1 to R3. This is consistent with interpretations A and B: a "
            "detectable honest-agent footprint that accumulates with interaction."
        )
    elif (r3_values >= 0.7).any():
        lines.append(
            "At least one observer-only method is substantially above chance, supporting "
            "interpretation A. The temporal pattern is not, by itself, strong evidence of monotonic "
            "accumulation."
        )
    else:
        lines.append(
            "The observer-only result is mixed rather than a clean confirmation of A–D. The exact "
            "AUROCs, paired intervals, and exposure strata should be treated as the primary result."
        )
    lines.extend([
        "",
        "Experiment A should be read alongside this: concentration of the largest positive score "
        "shift in positions 0–1 would explain why the original four-agent Peak/Split detector "
        "saturated while the deployment-relevant observer-only detector did not.",
        "",
        "No intervention-threshold, TFIR, PIC, or E2 analysis was run.",
        "",
    ])
    return "\n".join(lines)


def main():
    metadata = json.loads((CORE_DIR / "metadata_gen.json").read_text(encoding="utf-8"))
    phase2_metrics = json.loads(PHASE2_METRICS.read_text(encoding="utf-8"))
    phase2_scores = pd.read_csv(PHASE2_AGENT_SCORES)

    name_pool = load_name_pool(CORE_SOURCE)
    positions, turns, assumption_audit = audit_core_structure(
        metadata, name_pool, source_evidence(CORE_SOURCE)
    )
    if assumption_audit["status"] != "PASS":
        raise RuntimeError(
            "Core structural assumptions failed; experiment was not run: {}".format(
                assumption_audit["issues"]
            )
        )

    reuse_audit = validate_phase2_scores(phase2_scores, phase2_metrics)
    if reuse_audit["status"] != "PASS":
        raise RuntimeError(
            "Saved fixed-probe score audit failed; experiment was not run: {}".format(
                reuse_audit["issues"]
            )
        )

    enriched = enrich_fixed_probe_scores(phase2_scores, positions, turns)
    round_source_frame, role_summary = role_source_summary(enriched)
    role_contrasts = paired_role_contrasts(round_source_frame)

    observers = select_honest_observer_scores(enriched)
    observer_risks = aggregate_observer_risk(observers)
    observers_with_risks = add_run_risks_to_observers(observers, observer_risks)

    # Required raw-first ordering: all later metrics/figures use this persisted CSV.
    PREDICTION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    observers_with_risks.sort_values(
        ["fold", "scenario_id", "mode", "agent_name", "temporal_view", "round"]
    ).to_csv(PREDICTION_OUTPUT, index=False)
    saved_observers = pd.read_csv(PREDICTION_OUTPUT)
    saved_risks = aggregate_observer_risk(saved_observers)

    domains = sorted(saved_observers["domain"].unique())
    fold_metrics, summary_metrics, pair_metrics, metric_issues = calculate_observer_metrics(
        saved_risks,
        domains,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        seed=SEED,
    )
    exposure_pairs, exposure_summary = calculate_exposure_diagnostic(
        saved_observers,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        seed=SEED,
    )

    output_issues = assumption_audit["issues"] + reuse_audit["issues"] + metric_issues
    output = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not output_issues else "ISSUES",
        "scope": "Trusted-observer benchmark diagnostic only; no E2 or intervention-threshold analysis.",
        "sources": {
            "upstream_repository": "https://github.com/aaronrose227/narcbench",
            "upstream_git_commit": git_commit(UPSTREAM_ROOT),
            "upstream_source_modified": False,
            "core_generation_source": str(CORE_SOURCE.relative_to(PROJECT_ROOT)),
            "core_generation_source_sha256": sha256_file(CORE_SOURCE),
            "dataset_repository": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "metadata_sha256": sha256_file(CORE_DIR / "metadata_gen.json"),
            "phase2_metrics_sha256": sha256_file(PHASE2_METRICS),
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
            "metadata_rows": len(metadata),
            "run_count": positions["run_id"].nunique(),
            "scenario_pair_count": positions["scenario_id"].nunique(),
            "domains": domains,
            "observer_positions_zero_indexed": [2, 3],
        },
        "assumption_verification": assumption_audit,
        "fixed_probe_score_reuse": reuse_audit,
        "methodology": {
            "fold_protocol": "official leave-one-domain-out",
            "probe_training": phase2_metrics["methodology"]["probe_training"],
            "probe_reuse": "Exact persisted Phase 2 OOF agent scores; no classifier or scaler refit.",
            "temporal_views": phase2_metrics["methodology"]["temporal_views"],
            "observer_selection": "Only seeded generator positions 2 and 3, honest in both matched conditions.",
            "run_risk_methods": {
                "mean_observer_suspicion": "mean fixed-probe score of positions 2 and 3",
                "peak_observer_suspicion": "maximum fixed-probe score of positions 2 and 3",
            },
            "fold_summary": "unweighted mean and population SD across all expected domains",
            "paired_difference": "collusion run risk minus matched control run risk",
            "bootstrap": {
                "method": "percentile bootstrap",
                "unit": "matched scenario pair",
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": SEED,
                "confidence_level": 0.95,
            },
        },
        "experiment_a": {
            "description": "Round-only fixed-probe score distributions by role/source position.",
            "distribution_summary": role_summary.to_dict(orient="records"),
            "paired_position_contrasts": role_contrasts.to_dict(orient="records"),
        },
        "experiment_b": {
            "summary": summary_metrics.to_dict(orient="records"),
            "fold_level": fold_metrics.to_dict(orient="records"),
            "paired_scenario_scores": pair_metrics.to_dict(orient="records"),
            "metric_issues": metric_issues,
        },
        "exposure_diagnostic": {
            "supported": assumption_audit["assumptions"]["exposure_count_supported"],
            "definition": (
                "Number of preceding public messages authored by paired generator positions 0–1. "
                "They are actual colluder messages only when mode=collusion."
            ),
            "prefix_note": (
                "Prefix rows store endpoint exposure_count and source_exposure_counts; score-shift "
                "stratification uses round-only activations."
            ),
            "paired_observation_count": len(exposure_pairs),
            "score_shift_summary": exposure_summary.to_dict(orient="records"),
        },
        "predictions": {
            "path": str(PREDICTION_OUTPUT.relative_to(PROJECT_ROOT)),
            "rows": len(saved_observers),
            "sha256": sha256_file(PREDICTION_OUTPUT),
            "observer_agent_count_per_run": 2,
            "colluder_activation_rows": int((saved_observers["generator_position"] < 2).sum()),
        },
        "issues": output_issues,
    }
    output = json_ready(output)

    METRICS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    # Figures are built after raw predictions are persisted and reloaded.
    plot_role_decomposition(enriched, ROLE_FIGURE)
    plot_observer_detectability(summary_metrics, OBSERVER_FIGURE)
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT.write_text(build_report(output), encoding="utf-8")

    for path in (PREDICTION_OUTPUT, METRICS_OUTPUT, ROLE_FIGURE, OBSERVER_FIGURE, REPORT_OUTPUT):
        print("Saved {}".format(path.relative_to(PROJECT_ROOT)), flush=True)
    print("Status: {}".format(output["status"]), flush=True)
    return 0 if output["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
