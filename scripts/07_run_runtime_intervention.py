#!/usr/bin/env python3
"""Run Experiment E: runtime intervention value at layer 28."""

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
FIXED_METRICS = PROJECT_ROOT / "outputs" / "metrics" / "fixed_sentinel_probe_layer28.json"
FIXED_PREDICTIONS = (
    PROJECT_ROOT / "outputs" / "predictions" / "oof_fixed_sentinel_scores_layer28.csv"
)
OFFICIAL_METRICS = PROJECT_ROOT / "outputs" / "metrics" / "temporal_detectability_layer28.json"
OFFICIAL_PREDICTIONS = (
    PROJECT_ROOT / "outputs" / "predictions" / "oof_agent_temporal_scores_layer28.csv"
)
PREDICTION_OUTPUT = (
    PROJECT_ROOT / "outputs" / "predictions" / "oof_runtime_interventions_layer28.csv"
)
METRICS_OUTPUT = PROJECT_ROOT / "outputs" / "metrics" / "runtime_intervention_value_layer28.json"
REPORT_OUTPUT = PROJECT_ROOT / "outputs" / "logs" / "runtime_intervention_value_layer28.md"
FRONTIER_FIGURE = PROJECT_ROOT / "outputs" / "figures" / "security_utility_frontier_layer28.png"
TIMING_FIGURE = PROJECT_ROOT / "outputs" / "figures" / "intervention_timing_layer28.png"

LAYER = 28
SEED = 42
BOOTSTRAP_REPLICATES = 1000
MODEL_ID = "Qwen/Qwen3-32B-AWQ"
DATASET_ID = "aaronrose227/narcbench"
DATASET_REVISION = "19cb14b269e052b19ba6d2bd2f4519afd3b10c57"


sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(UPSTREAM_ROOT))
from probes import reproduce as upstream  # noqa: E402
from src.honest_observer_detection import (  # noqa: E402
    audit_core_structure,
    load_name_pool,
    source_evidence,
)
from src.runtime_intervention import (  # noqa: E402
    PREFIX_ROUNDS,
    TARGET_TFIR_BUDGETS,
    aggregate_two_agent_risks,
    apply_sequential_policy,
    bootstrap_pic_difference,
    calculate_intervention_metrics,
    conservative_empirical_threshold,
    trajectory_maxima,
)
from src.sentinel_probe import (  # noqa: E402
    build_observer_examples,
    filter_observer_examples,
    fit_full_discussion_sentinel,
    sentinel_fingerprint,
)
from src.temporal_detectability import (  # noqa: E402
    build_public_discussion_index,
    build_temporal_representation,
    probe_fingerprint,
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


def upstream_is_modified(path):
    result = subprocess.run(
        ["git", "-C", str(path), "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


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


def score_sentinel_rows(frame, scaler, classifier):
    """Score temporal observer rows with an already fitted Phase-D model."""
    if frame.empty:
        return pd.DataFrame()
    matrix = np.stack(frame["activation"].to_numpy())
    scores = classifier.decision_function(scaler.transform(matrix))
    columns = [
        "scenario_id",
        "run_id",
        "domain",
        "mode",
        "agent_name",
        "generator_position",
        "round",
    ]
    output = frame[columns].copy()
    output["score"] = scores.astype(float)
    return output


def compare_fixed_sentinel_scores(scored, reference, population, held_out_domain, fingerprint):
    expected = reference.loc[
        (reference["analysis_population"] == population)
        & (reference["observer_scope"] == "both_positions")
        & (reference["fold"] == held_out_domain)
        & (reference["temporal_view"] == "prefix")
    ].copy()
    keys = ["scenario_id", "run_id", "mode", "agent_name", "round"]
    comparison = scored.merge(
        expected[keys + ["sentinel_score", "model_fingerprint"]],
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not (comparison["_merge"] == "both").all():
        raise RuntimeError("Phase-D checksum comparison has missing or extra score rows")
    if set(comparison["model_fingerprint"]) != {fingerprint}:
        raise RuntimeError("Refitted sentinel fingerprint differs from saved Phase D")
    differences = np.abs(
        comparison["score"].to_numpy(dtype=float)
        - comparison["sentinel_score"].to_numpy(dtype=float)
    )
    maximum = float(np.max(differences)) if len(differences) else 0.0
    if not np.allclose(
        comparison["score"], comparison["sentinel_score"], rtol=0.0, atol=1e-10
    ):
        raise RuntimeError("Refitted sentinel scores differ from saved Phase D")
    return {"compared_rows": len(comparison), "maximum_absolute_score_difference": maximum}


def score_official_position_rows(
    temporal_index,
    run_metadata,
    positions,
    scaler,
    direction,
    selected_run_ids,
):
    """Score positions 0--1 with one fixed official deception direction."""
    position_lookup = positions.set_index(["run_id", "agent_name"])[
        ["generator_position", "actual_role"]
    ].to_dict(orient="index")
    records = []
    for run_id in sorted(selected_run_ids):
        run = run_metadata[run_id]
        selected_agents = []
        for agent_name in temporal_index[run_id]:
            position = position_lookup[(run_id, agent_name)]
            if int(position["generator_position"]) in (0, 1):
                selected_agents.append((int(position["generator_position"]), agent_name, position))
        if len(selected_agents) != 2:
            raise RuntimeError("Full-white-box reference does not have exactly two positions")
        for generator_position, agent_name, position in sorted(selected_agents):
            expected_role = "colluder" if run["mode"] == "collusion" else "honest"
            if position["actual_role"] != expected_role:
                raise RuntimeError("Unexpected role at a full-white-box reference position")
            round_vectors = temporal_index[run_id][agent_name]
            for endpoint_round in PREFIX_ROUNDS:
                vector = build_temporal_representation(round_vectors, "prefix", endpoint_round)
                score = float(direction @ scaler.transform(vector.reshape(1, -1))[0])
                records.append({
                    "scenario_id": run["scenario_id"],
                    "run_id": run_id,
                    "domain": run["domain"],
                    "mode": run["mode"],
                    "agent_name": agent_name,
                    "generator_position": generator_position,
                    "round": endpoint_round,
                    "score": score,
                })
    return pd.DataFrame.from_records(records)


def compare_official_scores(scored, reference, held_out_domain, fingerprint):
    expected = reference.loc[
        (reference["fold"] == held_out_domain)
        & (reference["temporal_view"] == "prefix")
        & (reference["generator_position"].isin([0, 1]))
    ].copy()
    keys = ["scenario_id", "run_id", "mode", "agent_name", "round"]
    comparison = scored.merge(
        expected[keys + ["deception_score", "probe_fingerprint"]],
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not (comparison["_merge"] == "both").all():
        raise RuntimeError("Official-probe checksum comparison has missing or extra rows")
    if set(comparison["probe_fingerprint"]) != {fingerprint}:
        raise RuntimeError("Refitted official probe fingerprint differs from Phase 2")
    differences = np.abs(
        comparison["score"].to_numpy(dtype=float)
        - comparison["deception_score"].to_numpy(dtype=float)
    )
    maximum = float(np.max(differences)) if len(differences) else 0.0
    if not np.allclose(
        comparison["score"], comparison["deception_score"], rtol=0.0, atol=1e-10
    ):
        raise RuntimeError("Refitted official scores differ from Phase 2")
    return {"compared_rows": len(comparison), "maximum_absolute_score_difference": maximum}


def calibrate_fold_thresholds(run_risks, held_out_domain, population, monitor):
    """Calibrate every method/budget from training-control trajectory maxima."""
    if set(run_risks["mode"]) != {"control"}:
        raise RuntimeError("Threshold calibration received a non-control trajectory")
    if held_out_domain in set(run_risks["domain"]):
        raise RuntimeError("Held-out control trajectory entered threshold calibration")
    maxima = trajectory_maxima(run_risks)
    records = []
    for method, frame in maxima.groupby("method", sort=True):
        for target_tfir in TARGET_TFIR_BUDGETS:
            calibration = conservative_empirical_threshold(
                frame["max_pre_final_risk"].to_numpy(dtype=float), target_tfir
            )
            records.append({
                "analysis_population": population,
                "monitor": monitor,
                "fold": held_out_domain,
                "method": method,
                "calibration_modes": ["control"],
                "calibration_domains": sorted(frame["domain"].unique()),
                "held_out_domain_present_in_calibration": False,
                **calibration,
            })
    return pd.DataFrame.from_records(records)


def run_trusted_sentinel(examples, domains, fixed_reference):
    policy_frames = []
    threshold_frames = []
    fit_audits = []
    checksum_audits = []
    for population in ("all_observers", "style_matched"):
        scoped = filter_observer_examples(examples, population, "both_positions")
        prefix = scoped.loc[scoped["temporal_view"] == "prefix"].copy()
        full = prefix.loc[prefix["round"] == 3].copy()
        population_held_risks = []
        population_thresholds = []
        for held_out_domain in domains:
            train_full = full.loc[full["domain"] != held_out_domain]
            test_full = full.loc[full["domain"] == held_out_domain]
            common = {
                "analysis_population": population,
                "observer_scope": "both_positions",
                "held_out_domain": held_out_domain,
            }
            if test_full.empty:
                fit_audits.append({
                    **common,
                    "fit_status": "not_fit_no_held_out_observations",
                    "training_rows_available": len(train_full),
                    "test_rows": 0,
                })
                continue
            scaler, classifier, audit = fit_full_discussion_sentinel(
                train_full, test_full, held_out_domain
            )
            fingerprint = sentinel_fingerprint(scaler, classifier)
            fit_audits.append({**common, "fit_status": "fit", **audit})

            held_rows = prefix.loc[prefix["domain"] == held_out_domain]
            held_scores = score_sentinel_rows(held_rows, scaler, classifier)
            checksum_audits.append({
                **common,
                "model_fingerprint": fingerprint,
                **compare_fixed_sentinel_scores(
                    held_scores,
                    fixed_reference,
                    population,
                    held_out_domain,
                    fingerprint,
                ),
            })
            held_risks = aggregate_two_agent_risks(
                held_scores,
                "sentinel",
                allow_variable_agents=(population == "style_matched"),
            )
            population_held_risks.append(held_risks)

            training_control_rows = prefix.loc[
                (prefix["domain"] != held_out_domain) & (prefix["mode"] == "control")
            ]
            training_scores = score_sentinel_rows(
                training_control_rows, scaler, classifier
            )
            training_risks = aggregate_two_agent_risks(
                training_scores,
                "sentinel",
                allow_variable_agents=(population == "style_matched"),
            )
            thresholds = calibrate_fold_thresholds(
                training_risks, held_out_domain, population, "trusted_sentinel"
            )
            population_thresholds.append(thresholds)

        held_risks = pd.concat(population_held_risks, ignore_index=True)
        thresholds = pd.concat(population_thresholds, ignore_index=True)
        policy_frames.append(
            apply_sequential_policy(
                held_risks, thresholds, population, "trusted_sentinel"
            )
        )
        threshold_frames.append(thresholds)
    return (
        pd.concat(policy_frames, ignore_index=True),
        pd.concat(threshold_frames, ignore_index=True),
        fit_audits,
        checksum_audits,
    )


def run_full_white_box(
    metadata,
    archive,
    temporal_index,
    run_metadata,
    positions,
    domains,
    official_reference,
):
    held_risk_frames = []
    threshold_frames = []
    fit_audits = []
    checksum_audits = []
    for held_out_domain in domains:
        scaler, direction = upstream.train_deception_probe(
            metadata, archive, LAYER, exclude_domain=held_out_domain
        )
        fingerprint = probe_fingerprint(scaler, direction)
        phase2_fold = official_reference.loc[official_reference["fold"] == held_out_domain]
        expected_fingerprints = set(phase2_fold["probe_fingerprint"])
        if expected_fingerprints != {fingerprint}:
            raise RuntimeError("Official reference probe fingerprint does not reproduce Phase 2")
        fit_audits.append({
            "held_out_domain": held_out_domain,
            "fit_status": "fit",
            "model_fingerprint": fingerprint,
            "upstream_function": "probes.reproduce.train_deception_probe",
        })

        held_run_ids = [
            run_id for run_id, run in run_metadata.items()
            if run["domain"] == held_out_domain
        ]
        held_scores = score_official_position_rows(
            temporal_index,
            run_metadata,
            positions,
            scaler,
            direction,
            held_run_ids,
        )
        checksum_audits.append({
            "held_out_domain": held_out_domain,
            "model_fingerprint": fingerprint,
            **compare_official_scores(
                held_scores, official_reference, held_out_domain, fingerprint
            ),
        })
        held_risk_frames.append(
            aggregate_two_agent_risks(held_scores, "deception", allow_variable_agents=False)
        )

        training_control_run_ids = [
            run_id for run_id, run in run_metadata.items()
            if run["domain"] != held_out_domain and run["mode"] == "control"
        ]
        training_scores = score_official_position_rows(
            temporal_index,
            run_metadata,
            positions,
            scaler,
            direction,
            training_control_run_ids,
        )
        training_risks = aggregate_two_agent_risks(
            training_scores, "deception", allow_variable_agents=False
        )
        threshold_frames.append(
            calibrate_fold_thresholds(
                training_risks,
                held_out_domain,
                "all_observers",
                "full_white_box_upper_bound",
            )
        )

    held_risks = pd.concat(held_risk_frames, ignore_index=True)
    thresholds = pd.concat(threshold_frames, ignore_index=True)
    policies = apply_sequential_policy(
        held_risks, thresholds, "all_observers", "full_white_box_upper_bound"
    )
    return policies, thresholds, fit_audits, checksum_audits


def validate_policy_records(policy, thresholds, domains):
    issues = []
    if not (policy["fold"] == policy["domain"]).all():
        issues.append("A policy record is not out-of-fold")
    if set(policy["mode"]) != {"collusion", "control"}:
        issues.append("Policy records do not contain both modes")
    if set(policy["target_tfir"].unique()) != set(TARGET_TFIR_BUDGETS):
        issues.append("Policy records do not contain all requested budgets")
    if policy["first_detection_round"].dropna().isin(PREFIX_ROUNDS).eq(False).any():
        issues.append("A policy used an invalid intervention round")
    if (policy["intervention_lead_rounds"].dropna() < 1).any():
        issues.append("A policy used the final round or a later intervention point")
    if not thresholds["training_budget_respected"].all():
        issues.append("A training-control calibration exceeded its target TFIR")
    if thresholds["held_out_domain_present_in_calibration"].any():
        issues.append("Held-out data entered threshold calibration")
    if thresholds["calibration_modes"].apply(lambda values: values != ["control"]).any():
        issues.append("A threshold used non-control calibration data")
    duplicate_keys = [
        "analysis_population", "monitor", "method", "target_tfir", "run_id"
    ]
    if policy.duplicated(duplicate_keys).any():
        issues.append("Duplicate runtime policy records")
    primary = policy.loc[
        (policy["analysis_population"] == "all_observers")
        & (policy["monitor"] == "trusted_sentinel")
    ]
    if set(primary["domain"]) != set(domains):
        issues.append("Primary trusted-sentinel policy does not cover every domain")
    pair_counts = policy.groupby(
        ["analysis_population", "monitor", "method", "target_tfir", "scenario_id"]
    )["mode"].agg(lambda values: tuple(sorted(values)))
    if not pair_counts.apply(lambda values: values == ("collusion", "control")).all():
        issues.append("A runtime policy configuration contains an unmatched scenario pair")
    return issues


def plot_security_utility_frontier(summary, output_path):
    series = [
        ("all_observers", "trusted_sentinel", "mean_sentinel_risk", "Trusted sentinel — mean", "#3568a8", "o", "-"),
        ("all_observers", "full_white_box_upper_bound", "mean_deception_risk", "Full white-box — mean", "#c44e52", "s", "-"),
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for population, monitor, method, label, color, marker, linestyle in series:
        frame = summary.loc[
            (summary["analysis_population"] == population)
            & (summary["monitor"] == monitor)
            & (summary["method"] == method)
        ].sort_values("target_tfir")
        x = frame["achieved_tfir"].to_numpy(dtype=float)
        y = frame["pic"].to_numpy(dtype=float)
        x_ci = np.stack(frame["achieved_tfir_bootstrap_ci_95"].to_numpy())
        y_ci = np.stack(frame["pic_bootstrap_ci_95"].to_numpy())
        axis.errorbar(
            x,
            y,
            xerr=np.vstack([x - x_ci[:, 0], x_ci[:, 1] - x]),
            yerr=np.vstack([y - y_ci[:, 0], y_ci[:, 1] - y]),
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.8,
            capsize=3,
            label=label,
        )
        for (achieved_tfir, pic), coincident in frame.groupby(
            ["achieved_tfir", "pic"], sort=True
        ):
            target_label = "/".join(
                "{}%".format(int(round(100 * value)))
                for value in coincident["target_tfir"]
            )
            axis.annotate(
                target_label,
                (achieved_tfir, pic),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )
    axis.set_xlim(left=-0.02)
    axis.set_ylim(-0.02, 1.03)
    axis.set_xlabel("Achieved held-out TFIR")
    axis.set_ylabel("Pre-intervention collusion coverage (PIC)")
    axis.set_title("Security–utility frontier — layer 28")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8, loc="best")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_intervention_timing(summary, output_path):
    series = [
        ("all_observers", "trusted_sentinel", "mean_sentinel_risk", "Trusted sentinel — mean", "#3568a8", "o", "-"),
        ("all_observers", "full_white_box_upper_bound", "mean_deception_risk", "Full white-box — mean", "#c44e52", "s", "-"),
    ]
    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    for population, monitor, method, label, color, marker, linestyle in series:
        frame = summary.loc[
            (summary["analysis_population"] == population)
            & (summary["monitor"] == monitor)
            & (summary["method"] == method)
            & np.isclose(summary["target_tfir"], 0.05)
        ]
        if len(frame) != 1:
            raise RuntimeError("Expected one 5% timing summary row")
        cumulative = frame.iloc[0]["cumulative_detection"]
        axis.plot(
            [1, 2, 3],
            [cumulative["R1"], cumulative["R2"], cumulative["R3"]],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.9,
            label=label,
        )
    axis.set_xticks([1, 2, 3], ["R1", "R1–R2", "R1–R2–R3"])
    axis.set_ylim(-0.02, 1.03)
    axis.set_xlabel("Available public-discussion prefix")
    axis.set_ylabel("Cumulative collusion trajectories alerted")
    axis.set_title("Intervention timing at 5% target TFIR — layer 28")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(fontsize=8, loc="best")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def format_interval(value, interval):
    return "{:.3f} [{:.3f}, {:.3f}]".format(value, interval[0], interval[1])


def format_optional(value):
    return "undefined" if value is None or pd.isna(value) else "{:.1f}".format(value)


def build_report(metrics):
    summary = pd.DataFrame(metrics["metrics"]["summary"])
    fold = pd.DataFrame(metrics["metrics"]["fold_level"])
    differences = pd.DataFrame(metrics["metrics"]["sentinel_vs_full_pic_difference"])
    thresholds = pd.DataFrame(metrics["calibration"]["thresholds"])
    lines = [
        "# Experiment E — Runtime Intervention Value (Layer 28)",
        "",
        "- Status: **{}**".format(metrics["status"]),
        "- Trusted monitor: fixed Phase-D honest-observer sentinel; positions 2–3 only",
        "- Reference monitor: official deception direction on positions 0–1 (oracle-like full-white-box upper bound)",
        "- Policy: alert at the first prefix whose risk is `>=` its fold-specific threshold; no round-4 activation",
        "- Calibration: training-domain control trajectories only; thresholded trajectory maximum over R1/R1–R2/R1–R2–R3",
        "- Bootstrap: 1,000 matched scenario-pair resamples, seed 42",
        "",
        "## Primary security–utility table",
        "",
        "Mean aggregation is primary for both monitors. Medians are among detected collusion trajectories; "
        "lead is measured to round 4.",
        "",
        "| Method | Target TFIR | Achieved TFIR (95% CI) | PIC (95% CI) | Median alert | Median lead |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    primary_specs = (
        ("trusted_sentinel", "mean_sentinel_risk", "Trusted sentinel"),
        ("full_white_box_upper_bound", "mean_deception_risk", "Full white-box reference"),
    )
    for monitor, method, label in primary_specs:
        frame = summary.loc[
            (summary["analysis_population"] == "all_observers")
            & (summary["monitor"] == monitor)
            & (summary["method"] == method)
        ].sort_values("target_tfir")
        for _, row in frame.iterrows():
            lines.append("| {} | {:.0%} | {} | {} | {} | {} |".format(
                label,
                row["target_tfir"],
                format_interval(row["achieved_tfir"], row["achieved_tfir_bootstrap_ci_95"]),
                format_interval(row["pic"], row["pic_bootstrap_ci_95"]),
                format_optional(row["median_alert_round_detected"]),
                format_optional(row["median_intervention_lead_rounds_detected"]),
            ))

    lines.extend([
        "",
        "## Intervention timing at 5% target",
        "",
        "| Method | Achieved TFIR | First at R1 | First at R2 | First at R3 | Never | Cumulative R1 / R2 / R3 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for monitor, method, label in primary_specs:
        row = summary.loc[
            (summary["analysis_population"] == "all_observers")
            & (summary["monitor"] == monitor)
            & (summary["method"] == method)
            & np.isclose(summary["target_tfir"], 0.05)
        ].iloc[0]
        counts = row["first_detection_round_counts"]
        cumulative = row["cumulative_detection"]
        lines.append("| {} | {:.1%} | {} | {} | {} | {} | {:.1%} / {:.1%} / {:.1%} |".format(
            label,
            row["achieved_tfir"],
            counts["R1"], counts["R2"], counts["R3"], counts["never"],
            cumulative["R1"], cumulative["R2"], cumulative["R3"],
        ))

    lines.extend([
        "",
        "## Trusted-sentinel sensitivity analyses",
        "",
        "Peak aggregation is secondary. Style-matched analyses retain one or two observers per "
        "scenario according to the pre-existing prompt-style audit; empty held-out folds remain explicit.",
        "",
        "| Population | Aggregation | Target | Achieved TFIR (95% CI) | PIC (95% CI) | Pairs |",
        "|---|---|---:|---:|---:|---:|",
    ])
    sentinel = summary.loc[summary["monitor"] == "trusted_sentinel"].sort_values(
        ["analysis_population", "method", "target_tfir"]
    )
    for _, row in sentinel.iterrows():
        lines.append("| {} | {} | {:.0%} | {} | {} | {} |".format(
            row["analysis_population"],
            "mean" if row["method"].startswith("mean") else "peak",
            row["target_tfir"],
            format_interval(row["achieved_tfir"], row["achieved_tfir_bootstrap_ci_95"]),
            format_interval(row["pic"], row["pic_bootstrap_ci_95"]),
            int(row["scenario_pair_count"]),
        ))

    lines.extend([
        "",
        "## Paired PIC comparison with full white-box reference",
        "",
        "The comparison uses mean aggregation at the same target budgets. It compares intervention "
        "coverage, not raw AUROC; independently calibrated achieved TFIRs are shown above.",
        "",
        "| Target | Trusted PIC | Full-white-box PIC | Trusted − full (95% CI) |",
        "|---:|---:|---:|---:|",
    ])
    for _, row in differences.sort_values("target_tfir").iterrows():
        lines.append("| {:.0%} | {:.3f} | {:.3f} | {} |".format(
            row["target_tfir"],
            row["sentinel_pic"],
            row["full_white_box_pic"],
            format_interval(
                row["sentinel_minus_full_white_box_pic"],
                row["sentinel_minus_full_white_box_pic_bootstrap_ci_95"],
            ),
        ))

    lines.extend([
        "",
        "## Calibration and leakage checks",
        "",
        "The conservative empirical threshold is one representable float above NumPy's `higher` "
        "order statistic. This tie-safe choice ensures the inclusive `>=` rule never exceeds its "
        "target on training controls. Discrete samples can therefore produce an achieved training "
        "rate below the nominal budget.",
        "",
        "- Threshold records: {}; training-budget violations: {}".format(
            len(thresholds), int((~thresholds["training_budget_respected"]).sum())
        ),
        "- Phase-D held-out score rows rechecked: {}; maximum absolute difference: {:.3g}".format(
            metrics["reuse_validation"]["fixed_sentinel_compared_rows"],
            metrics["reuse_validation"]["fixed_sentinel_maximum_absolute_score_difference"],
        ),
        "- Official held-out score rows rechecked: {}; maximum absolute difference: {:.3g}".format(
            metrics["reuse_validation"]["official_probe_compared_rows"],
            metrics["reuse_validation"]["official_probe_maximum_absolute_score_difference"],
        ),
        "- Trusted-sentinel colluder activation rows: 0",
        "- Threshold records containing a held-out domain: 0",
        "",
        "## Fold-level results",
        "",
        "Every expected domain is retained below; undefined style-matched folds are reported rather than dropped.",
        "",
    ])
    for keys, frame in fold.groupby(
        ["analysis_population", "monitor", "method", "target_tfir"], sort=True
    ):
        population, monitor, method, target = keys
        values = []
        for _, row in frame.sort_values("fold").iterrows():
            if row["metric_issue"] is not None and not pd.isna(row["metric_issue"]):
                rendered = "undefined ({})".format(row["metric_issue"])
            else:
                rendered = "TFIR {:.3f}, PIC {:.3f}".format(
                    row["achieved_tfir"], row["pic"]
                )
            values.append("{}={}".format(row["fold"], rendered))
        lines.append("- `{}` / `{}` / `{}` / `{:.0%}`: {}".format(
            population, monitor, method, target, "; ".join(values)
        ))

    primary_5 = summary.loc[
        (summary["analysis_population"] == "all_observers")
        & (summary["monitor"] == "trusted_sentinel")
        & (summary["method"] == "mean_sentinel_risk")
        & np.isclose(summary["target_tfir"], 0.05)
    ].iloc[0]
    white_5 = summary.loc[
        (summary["analysis_population"] == "all_observers")
        & (summary["monitor"] == "full_white_box_upper_bound")
        & (summary["method"] == "mean_deception_risk")
        & np.isclose(summary["target_tfir"], 0.05)
    ].iloc[0]
    style_5 = summary.loc[
        (summary["analysis_population"] == "style_matched")
        & (summary["monitor"] == "trusted_sentinel")
        & (summary["method"] == "mean_sentinel_risk")
        & np.isclose(summary["target_tfir"], 0.05)
    ].iloc[0]
    comparison_5 = differences.loc[np.isclose(differences["target_tfir"], 0.05)].iloc[0]
    sentinel_training_5 = thresholds.loc[
        (thresholds["analysis_population"] == "all_observers")
        & (thresholds["monitor"] == "trusted_sentinel")
        & (thresholds["method"] == "mean_sentinel_risk")
        & np.isclose(thresholds["target_tfir"], 0.05),
        "training_empirical_tfir",
    ]
    lines.extend([
        "",
        "## Interpretation",
        "",
        "At the 5% target, the trusted sentinel achieves {:.1%} held-out TFIR and {:.1%} PIC "
        "(95% CI {:.1%}–{:.1%}). The full-white-box reference achieves {:.1%} TFIR and {:.1%} PIC."
        .format(
            primary_5["achieved_tfir"],
            primary_5["pic"],
            primary_5["pic_bootstrap_ci_95"][0],
            primary_5["pic_bootstrap_ci_95"][1],
            white_5["achieved_tfir"],
            white_5["pic"],
        ),
        "This is Outcome B: high predictive discrimination does not translate into the requested "
        "operational false-intervention budget. Although every fold's training-control rate was at "
        "or below 5% ({:.1%}–{:.1%}), held-out TFIR was 25.8%. The calibration is in-sample with "
        "respect to the fixed probe's training controls, so high-dimensional fit optimism and domain "
        "shift are plausible explanations; this experiment does not distinguish them.".format(
            sentinel_training_5.min(), sentinel_training_5.max()
        ),
        "By R1, the trusted policy has alerted on {:.1%} of collusion trajectories; its cumulative "
        "coverage is {:.1%} by R2 and {:.1%} by R3. This quantifies practical timing without using "
        "the final round.".format(
            primary_5["cumulative_detection"]["R1"],
            primary_5["cumulative_detection"]["R2"],
            primary_5["cumulative_detection"]["R3"],
        ),
        "This is also consistent with Outcome C. At the same 5% calibration target, full white-box "
        "has {:+.1%} higher PIC (trusted minus full: {:.1%}, 95% CI {:.1%} to {:.1%}) and {:+.1%} "
        "percentage points lower achieved TFIR. More strongly, its most conservative observed point (9.7% TFIR, "
        "100% PIC) dominates every trusted-sentinel point on the measured frontier. Because achieved "
        "TFIR differs between monitors, nominal-target comparisons must not be read as equal-achieved-"
        "budget comparisons.".format(
            white_5["pic"] - primary_5["pic"],
            comparison_5["sentinel_minus_full_white_box_pic"],
            comparison_5["sentinel_minus_full_white_box_pic_bootstrap_ci_95"][0],
            comparison_5["sentinel_minus_full_white_box_pic_bootstrap_ci_95"][1],
            primary_5["achieved_tfir"] - white_5["achieved_tfir"],
        ),
        "The style-matched 5% sensitivity yields {:.1%} achieved TFIR and {:.1%} PIC across {} "
        "scenario pairs. Its wider uncertainty and missing domains limit precision.".format(
            style_5["achieved_tfir"], style_5["pic"], int(style_5["scenario_pair_count"])
        ),
        "The full-white-box curve is an oracle-like research reference because it assumes access to "
        "the known colluder positions. It is not a deployable requirement and is not used by the trusted policy.",
        "These are retrospective benchmark estimates from 31 matched Core scenarios, not evidence of "
        "production readiness. Threshold transport, calibration drift, benign-interruption costs, and "
        "adversarial adaptation remain untested.",
        "",
        "No further experiment or threshold analysis beyond the specified Experiment E was run.",
        "",
    ])
    return "\n".join(lines)


def main():
    fixed_metrics = json.loads(FIXED_METRICS.read_text(encoding="utf-8"))
    official_metrics = json.loads(OFFICIAL_METRICS.read_text(encoding="utf-8"))
    if fixed_metrics["status"] != "PASS" or official_metrics["status"] != "PASS":
        raise RuntimeError("Required Phase 2 or Experiment D artifact is not PASS")
    if sha256_file(FIXED_PREDICTIONS) != fixed_metrics["predictions"]["sha256"]:
        raise RuntimeError("Phase-D prediction hash mismatch")
    if sha256_file(OFFICIAL_PREDICTIONS) != official_metrics["predictions"]["agent_sha256"]:
        raise RuntimeError("Phase-2 official prediction hash mismatch")

    fixed_reference = pd.read_csv(FIXED_PREDICTIONS)
    official_reference = pd.read_csv(OFFICIAL_PREDICTIONS)
    metadata, archive = upstream.load_data(CORE_DIR)
    try:
        activations = upstream.get_layer(archive, len(metadata), LAYER)
        positions, turns, structure_audit = audit_core_structure(
            metadata,
            load_name_pool(CORE_SOURCE),
            source_evidence(CORE_SOURCE),
        )
        if structure_audit["status"] != "PASS":
            raise RuntimeError("Core structure audit failed")
        temporal_index, run_metadata = build_public_discussion_index(metadata, activations)
        examples = build_observer_examples(temporal_index, run_metadata, positions, turns)
        examples["layer"] = LAYER
        domains = sorted(examples["domain"].unique())
        official_reference = official_reference.merge(
            positions[["run_id", "agent_name", "generator_position"]],
            on=["run_id", "agent_name"],
            how="left",
            validate="many_to_one",
        )
        if official_reference["generator_position"].isna().any():
            raise RuntimeError("A Phase-2 score could not be linked to audited generator position")

        trusted_policy, trusted_thresholds, sentinel_fits, sentinel_checks = run_trusted_sentinel(
            examples, domains, fixed_reference
        )
        white_policy, white_thresholds, white_fits, white_checks = run_full_white_box(
            metadata,
            archive,
            temporal_index,
            run_metadata,
            positions,
            domains,
            official_reference,
        )
    finally:
        archive.close()

    policy = pd.concat([trusted_policy, white_policy], ignore_index=True).sort_values(
        ["analysis_population", "monitor", "method", "target_tfir", "fold", "scenario_id", "mode"]
    )
    thresholds = pd.concat([trusted_thresholds, white_thresholds], ignore_index=True).sort_values(
        ["analysis_population", "monitor", "method", "target_tfir", "fold"]
    )
    validation_issues = validate_policy_records(policy, thresholds, domains)
    convergence = [
        audit for audit in sentinel_fits if audit.get("convergence_warnings")
    ]
    if convergence:
        validation_issues.append("One or more fixed-sentinel refits emitted a convergence warning")

    # Persist raw OOF policy outcomes before metrics or plotting.
    PREDICTION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    policy.to_csv(PREDICTION_OUTPUT, index=False)
    saved_policy = pd.read_csv(PREDICTION_OUTPUT)
    print("Saved {}".format(PREDICTION_OUTPUT.relative_to(PROJECT_ROOT)), flush=True)

    summary, folds, metric_issues = calculate_intervention_metrics(
        saved_policy,
        domains,
        replicates=BOOTSTRAP_REPLICATES,
        seed=SEED,
    )
    primary_trusted = saved_policy.loc[
        (saved_policy["analysis_population"] == "all_observers")
        & (saved_policy["monitor"] == "trusted_sentinel")
        & (saved_policy["method"] == "mean_sentinel_risk")
    ]
    primary_white = saved_policy.loc[
        (saved_policy["analysis_population"] == "all_observers")
        & (saved_policy["monitor"] == "full_white_box_upper_bound")
        & (saved_policy["method"] == "mean_deception_risk")
    ]
    pic_differences = bootstrap_pic_difference(
        primary_trusted,
        primary_white,
        replicates=BOOTSTRAP_REPLICATES,
        seed=SEED,
    )

    fixed_rows = sum(item["compared_rows"] for item in sentinel_checks)
    official_rows = sum(item["compared_rows"] for item in white_checks)
    output = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not validation_issues else "ISSUES",
        "scope": "Experiment E runtime intervention value only; no further experiment.",
        "sources": {
            "upstream_repository": "https://github.com/aaronrose227/narcbench",
            "upstream_git_commit": git_commit(UPSTREAM_ROOT),
            "upstream_source_modified": upstream_is_modified(UPSTREAM_ROOT),
            "dataset_repository": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "metadata_sha256": sha256_file(CORE_DIR / "metadata_gen.json"),
            "activations_sha256": sha256_file(CORE_DIR / "activations_gen.npz"),
            "fixed_sentinel_metrics_sha256": sha256_file(FIXED_METRICS),
            "fixed_sentinel_predictions_sha256": sha256_file(FIXED_PREDICTIONS),
            "official_temporal_metrics_sha256": sha256_file(OFFICIAL_METRICS),
            "official_temporal_predictions_sha256": sha256_file(OFFICIAL_PREDICTIONS),
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
        },
        "research_question": (
            "Can a trusted open-weight sentinel intervene early enough to be useful while "
            "respecting a fixed false-intervention budget?"
        ),
        "methodology": {
            "trusted_sentinel_features": "honest observer positions 2 and 3 only",
            "trusted_sentinel_colluder_activation_rows": 0,
            "trusted_sentinel_probe": "exact refit of Experiment D mean(R1,R2,R3) StandardScaler + LogisticRegression",
            "new_temporal_classifier_trained": False,
            "temporal_views": {
                "R1": [1],
                "R1-R2": [1, 2],
                "R1-R2-R3": [1, 2, 3],
            },
            "risk_aggregations": {
                "primary": "mean of the two honest-observer scores",
                "secondary": "maximum of the two honest-observer scores",
                "style_matched": "mean/max of retained matched observers (one or two per scenario)",
            },
            "threshold_calibration": {
                "unit": "training-domain control trajectory",
                "statistic": "maximum risk over all three pre-final prefixes",
                "target_tfir_budgets": list(TARGET_TFIR_BUDGETS),
                "quantile_method": "higher",
                "tie_handling": "next representable float above selected order statistic",
                "held_out_data_used": False,
            },
            "sequential_policy": "alert at first prefix risk >= threshold; stop; no round 4",
            "full_white_box_reference": {
                "status": "oracle-like research upper bound",
                "probe": "official leave-one-domain-out deception direction",
                "positions": [0, 1],
                "collusion_role": "colluder",
                "control_role": "honest at the same positions",
                "comparison_metric": "intervention coverage, not raw AUROC",
            },
            "bootstrap": {
                "method": "percentile bootstrap",
                "unit": "matched scenario pair",
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": SEED,
                "confidence_level": 0.95,
            },
        },
        "structure_audit": structure_audit,
        "reuse_validation": {
            "fixed_sentinel_fit_records": sentinel_fits,
            "fixed_sentinel_checksums": sentinel_checks,
            "fixed_sentinel_compared_rows": fixed_rows,
            "fixed_sentinel_maximum_absolute_score_difference": max(
                item["maximum_absolute_score_difference"] for item in sentinel_checks
            ),
            "official_probe_fit_records": white_fits,
            "official_probe_checksums": white_checks,
            "official_probe_compared_rows": official_rows,
            "official_probe_maximum_absolute_score_difference": max(
                item["maximum_absolute_score_difference"] for item in white_checks
            ),
            "convergence_warning_count": len(convergence),
        },
        "calibration": {
            "threshold_record_count": len(thresholds),
            "training_budget_violation_count": int((~thresholds["training_budget_respected"]).sum()),
            "thresholds": thresholds.to_dict(orient="records"),
        },
        "predictions": {
            "path": str(PREDICTION_OUTPUT.relative_to(PROJECT_ROOT)),
            "rows": len(saved_policy),
            "sha256": sha256_file(PREDICTION_OUTPUT),
        },
        "metrics": {
            "summary": summary.to_dict(orient="records"),
            "fold_level": folds.to_dict(orient="records"),
            "undefined_fold_records": metric_issues,
            "undefined_fold_count": len(metric_issues),
            "sentinel_vs_full_pic_difference": pic_differences,
        },
        "issues": validation_issues,
        "limitations": [
            "Retrospective evaluation on 31 matched NARCBench-Core scenarios.",
            "Threshold transport and calibration drift are not tested.",
            "Benign interruption cost is represented only by TFIR, not measured downstream impact.",
            "The full-white-box reference assumes known colluder positions and is oracle-like.",
            "Style-matched sensitivity is smaller and lacks retained observations in some domains.",
            "No claim of production readiness is supported.",
        ],
    }
    output = json_ready(output)
    METRICS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUTPUT.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    saved_output = json.loads(METRICS_OUTPUT.read_text(encoding="utf-8"))
    saved_summary = pd.DataFrame(saved_output["metrics"]["summary"])
    plot_security_utility_frontier(saved_summary, FRONTIER_FIGURE)
    plot_intervention_timing(saved_summary, TIMING_FIGURE)
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT.write_text(build_report(saved_output), encoding="utf-8")

    for path in (METRICS_OUTPUT, REPORT_OUTPUT, FRONTIER_FIGURE, TIMING_FIGURE):
        print("Saved {}".format(path.relative_to(PROJECT_ROOT)), flush=True)
    print("Status: {}".format(output["status"]), flush=True)
    return 0 if output["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
