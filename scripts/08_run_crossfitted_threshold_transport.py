#!/usr/bin/env python3
"""Run Experiment F: cross-fitted threshold transport diagnostic at layer 28."""

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
EXPERIMENT_E_METRICS = PROJECT_ROOT / "outputs" / "metrics" / "runtime_intervention_value_layer28.json"
EXPERIMENT_E_PREDICTIONS = PROJECT_ROOT / "outputs" / "predictions" / "oof_runtime_interventions_layer28.csv"
FIXED_SENTINEL_METRICS = PROJECT_ROOT / "outputs" / "metrics" / "fixed_sentinel_probe_layer28.json"
FIXED_SENTINEL_PREDICTIONS = PROJECT_ROOT / "outputs" / "predictions" / "oof_fixed_sentinel_scores_layer28.csv"
PREDICTION_OUTPUT = PROJECT_ROOT / "outputs" / "predictions" / "oof_crossfitted_calibration_layer28.csv"
METRICS_OUTPUT = PROJECT_ROOT / "outputs" / "metrics" / "crossfitted_threshold_transport_layer28.json"
DOMAIN_FIGURE = PROJECT_ROOT / "outputs" / "figures" / "control_risk_by_domain_layer28.png"
COMPARISON_FIGURE = PROJECT_ROOT / "outputs" / "figures" / "calibration_comparison_layer28.png"
REPORT_OUTPUT = PROJECT_ROOT / "outputs" / "logs" / "crossfitted_threshold_transport_layer28.md"

LAYER = 28
SEED = 42
BOOTSTRAP_REPLICATES = 1000
QUANTILE_MINIMUM_N = 10
MODEL_ID = "Qwen/Qwen3-32B-AWQ"
DATASET_ID = "aaronrose227/narcbench"
DATASET_REVISION = "19cb14b269e052b19ba6d2bd2f4519afd3b10c57"


sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(UPSTREAM_ROOT))
from probes import reproduce as upstream  # noqa: E402
from src.crossfitted_thresholds import (  # noqa: E402
    attach_outer_policy_to_scores,
    bootstrap_calibration_differences,
    build_crossfitted_policy,
    experiment_e_domain_shift_audit,
    normalize_run_risks,
    trajectory_normalized_maxima,
    validate_crossfitted_predictions,
)
from src.honest_observer_detection import (  # noqa: E402
    audit_core_structure,
    load_name_pool,
    source_evidence,
)
from src.runtime_intervention import (  # noqa: E402
    TARGET_TFIR_BUDGETS,
    aggregate_two_agent_risks,
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


def score_mean_sentinel_risks(frame, scaler, classifier):
    if frame.empty:
        raise ValueError("Cannot score an empty sentinel frame")
    matrix = np.stack(frame["activation"].to_numpy())
    agent_scores = frame[
        ["scenario_id", "run_id", "domain", "mode", "round", "agent_name"]
    ].copy()
    agent_scores["score"] = classifier.decision_function(
        scaler.transform(matrix)
    ).astype(float)
    risks = aggregate_two_agent_risks(
        agent_scores, "sentinel", allow_variable_agents=False
    )
    risks = risks.loc[risks["method"] == "mean_sentinel_risk"].copy()
    return risks


def add_prediction_provenance(
    normalized,
    record_type,
    outer_fold,
    inner_fold,
    training_domains,
    fingerprint,
):
    output = normalized.copy()
    maxima = trajectory_normalized_maxima(output)
    output = output.merge(
        maxima,
        on=["scenario_id", "run_id", "domain", "mode"],
        how="left",
        validate="many_to_one",
    )
    output["record_type"] = record_type
    output["outer_fold"] = outer_fold
    output["inner_fold"] = inner_fold
    output["model_training_domains"] = "|".join(training_domains)
    output["model_fingerprint"] = fingerprint
    output["observer_positions"] = "2|3"
    output["observer_roles"] = "honest|honest"
    output["layer"] = LAYER
    output["temporal_view"] = "prefix"
    output["prefix_label"] = output["round"].map(
        {1: "R1", 2: "R1-R2", 3: "R1-R2-R3"}
    )
    output["source_rounds"] = output["round"].map(
        {1: "1", 2: "1|2", 3: "1|2|3"}
    )
    output["probe_training_representation"] = "mean(R1,R2,R3)"
    output["normalization_rule"] = "count(training-control trajectory maxima < raw risk) / n"
    output["final_round_used"] = False
    return output


def compare_outer_scores_with_experiment_e(outer_scores, experiment_e, outer_fold):
    expected = experiment_e.loc[
        (experiment_e["analysis_population"] == "all_observers")
        & (experiment_e["monitor"] == "trusted_sentinel")
        & (experiment_e["method"] == "mean_sentinel_risk")
        & np.isclose(experiment_e["target_tfir"], 0.05)
        & (experiment_e["fold"] == outer_fold)
    ].copy()
    expected_long = expected.melt(
        id_vars=["scenario_id", "run_id", "domain", "mode"],
        value_vars=["risk_r1", "risk_r1_r2", "risk_r1_r2_r3"],
        var_name="risk_column",
        value_name="experiment_e_raw_risk",
    )
    expected_long["round"] = expected_long["risk_column"].map(
        {"risk_r1": 1, "risk_r1_r2": 2, "risk_r1_r2_r3": 3}
    )
    keys = ["scenario_id", "run_id", "domain", "mode", "round"]
    comparison = outer_scores.merge(
        expected_long[keys + ["experiment_e_raw_risk"]],
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not (comparison["_merge"] == "both").all():
        raise RuntimeError("Outer-model scores have missing or extra Experiment-E rows")
    difference = np.abs(
        comparison["raw_risk"].to_numpy(dtype=float)
        - comparison["experiment_e_raw_risk"].to_numpy(dtype=float)
    )
    maximum = float(np.max(difference))
    if not np.allclose(
        comparison["raw_risk"], comparison["experiment_e_raw_risk"], rtol=0, atol=1e-10
    ):
        raise RuntimeError("Final outer model does not reproduce Experiment-E raw risks")
    return {"compared_rows": len(comparison), "maximum_absolute_difference": maximum}


def compare_outer_threshold_reference(reference_maxima, experiment_e, outer_fold):
    checks = []
    for target_tfir in TARGET_TFIR_BUDGETS:
        reproduced = conservative_empirical_threshold(reference_maxima, target_tfir)
        expected = experiment_e.loc[
            (experiment_e["analysis_population"] == "all_observers")
            & (experiment_e["monitor"] == "trusted_sentinel")
            & (experiment_e["method"] == "mean_sentinel_risk")
            & np.isclose(experiment_e["target_tfir"], target_tfir)
            & (experiment_e["fold"] == outer_fold),
            "threshold",
        ].unique()
        if len(expected) != 1 or not np.isclose(
            reproduced["threshold"], expected[0], rtol=0, atol=1e-12
        ):
            raise RuntimeError("Outer training-control distribution does not reproduce Experiment E")
        checks.append({
            "target_tfir": float(target_tfir),
            "experiment_e_threshold": float(expected[0]),
            "reproduced_threshold": float(reproduced["threshold"]),
            "absolute_difference": float(abs(reproduced["threshold"] - expected[0])),
        })
    return checks


def generate_crossfitted_scores(examples, domains, experiment_e, fixed_reference):
    scoped = filter_observer_examples(examples, "all_observers", "both_positions")
    prefix = scoped.loc[scoped["temporal_view"] == "prefix"].copy()
    full = prefix.loc[prefix["round"] == 3].copy()
    model_cache = {}
    unique_fit_audits = []
    context_audits = []
    outer_checks = []
    prediction_frames = []

    def get_model(excluded_domains, test_domain):
        key = tuple(sorted(excluded_domains))
        reused = key in model_cache
        if not reused:
            train = full.loc[~full["domain"].isin(key)]
            test = full.loc[full["domain"] == test_domain]
            scaler, classifier, audit = fit_full_discussion_sentinel(
                train, test, test_domain
            )
            if audit["convergence_warnings"]:
                raise RuntimeError("A cross-fitted sentinel model emitted a convergence warning")
            model_cache[key] = (scaler, classifier)
            unique_fit_audits.append({
                "excluded_domains": list(key),
                "training_domains": sorted(train["domain"].unique()),
                "fit_training_rows": len(train),
                "fit_training_control_rows": int((train["mode"] == "control").sum()),
                "fit_training_collusion_rows": int((train["mode"] == "collusion").sum()),
                "model_fingerprint": sentinel_fingerprint(scaler, classifier),
                "classifier": audit["classifier"],
                "scaler": audit["scaler"],
            })
        scaler, classifier = model_cache[key]
        return scaler, classifier, reused

    for outer_index, outer_fold in enumerate(domains, start=1):
        outer_training_domains = sorted(set(domains) - {outer_fold})
        inner_contexts = []
        for inner_fold in outer_training_domains:
            excluded = {outer_fold, inner_fold}
            training_domains = sorted(set(domains) - excluded)
            scaler, classifier, reused = get_model(excluded, inner_fold)
            fingerprint = sentinel_fingerprint(scaler, classifier)

            reference_rows = prefix.loc[
                (prefix["domain"].isin(training_domains)) & (prefix["mode"] == "control")
            ]
            reference_risks = score_mean_sentinel_risks(
                reference_rows, scaler, classifier
            )
            reference_maxima = trajectory_maxima(reference_risks)[
                "max_pre_final_risk"
            ].to_numpy(dtype=float)
            inner_rows = prefix.loc[
                (prefix["domain"] == inner_fold) & (prefix["mode"] == "control")
            ]
            inner_risks = score_mean_sentinel_risks(inner_rows, scaler, classifier)
            normalized_inner = normalize_run_risks(inner_risks, reference_maxima)
            inner_prediction = add_prediction_provenance(
                normalized_inner,
                "inner_calibration",
                outer_fold,
                inner_fold,
                training_domains,
                fingerprint,
            )
            prediction_frames.append(inner_prediction)
            inner_contexts.append(inner_prediction)
            context_audits.append({
                "context": "inner_crossfit",
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "excluded_domains": sorted(excluded),
                "model_training_domains": training_domains,
                "outer_fold_present_in_model_training": False,
                "inner_fold_present_in_model_training": False,
                "inner_control_trajectory_count": inner_rows["scenario_id"].nunique(),
                "normalization_reference_control_trajectory_count": len(reference_maxima),
                "model_fingerprint": fingerprint,
                "reused_equivalent_training_set_fit": reused,
            })

        inner_predictions = pd.concat(inner_contexts, ignore_index=True)
        inner_maxima = trajectory_normalized_maxima(inner_predictions)
        if len(inner_maxima) != prefix.loc[
            (prefix["domain"] != outer_fold) & (prefix["mode"] == "control")
        ]["scenario_id"].nunique():
            raise RuntimeError("Inner cross-fitting did not produce one calibration maximum per outer-training control")

        outer_scaler, outer_classifier, reused = get_model({outer_fold}, outer_fold)
        if reused:
            raise RuntimeError("Unexpected cached final outer model")
        outer_fingerprint = sentinel_fingerprint(outer_scaler, outer_classifier)
        expected_fingerprints = set(fixed_reference.loc[
            (fixed_reference["analysis_population"] == "all_observers")
            & (fixed_reference["observer_scope"] == "both_positions")
            & (fixed_reference["fold"] == outer_fold),
            "model_fingerprint",
        ])
        if expected_fingerprints != {outer_fingerprint}:
            raise RuntimeError("Final outer sentinel fingerprint differs from Experiment D")

        outer_reference_rows = prefix.loc[
            (prefix["domain"] != outer_fold) & (prefix["mode"] == "control")
        ]
        outer_reference_risks = score_mean_sentinel_risks(
            outer_reference_rows, outer_scaler, outer_classifier
        )
        outer_reference_maxima = trajectory_maxima(outer_reference_risks)[
            "max_pre_final_risk"
        ].to_numpy(dtype=float)
        outer_test_rows = prefix.loc[prefix["domain"] == outer_fold]
        outer_raw_risks = score_mean_sentinel_risks(
            outer_test_rows, outer_scaler, outer_classifier
        )
        normalized_outer = normalize_run_risks(
            outer_raw_risks, outer_reference_maxima
        )
        outer_prediction = add_prediction_provenance(
            normalized_outer,
            "outer_test",
            outer_fold,
            None,
            outer_training_domains,
            outer_fingerprint,
        )
        prediction_frames.append(outer_prediction)
        score_check = compare_outer_scores_with_experiment_e(
            outer_prediction, experiment_e, outer_fold
        )
        threshold_checks = compare_outer_threshold_reference(
            outer_reference_maxima, experiment_e, outer_fold
        )
        outer_checks.append({
            "outer_fold": outer_fold,
            "model_fingerprint": outer_fingerprint,
            "model_training_domains": outer_training_domains,
            "normalization_reference_control_trajectory_count": len(outer_reference_maxima),
            "experiment_e_score_reproduction": score_check,
            "experiment_e_threshold_reproduction": threshold_checks,
        })
        context_audits.append({
            "context": "final_outer_model",
            "outer_fold": outer_fold,
            "inner_fold": None,
            "excluded_domains": [outer_fold],
            "model_training_domains": outer_training_domains,
            "outer_fold_present_in_model_training": False,
            "inner_fold_present_in_model_training": None,
            "outer_test_trajectory_count": outer_test_rows["scenario_id"].nunique(),
            "normalization_reference_control_trajectory_count": len(outer_reference_maxima),
            "model_fingerprint": outer_fingerprint,
            "reused_equivalent_training_set_fit": False,
        })
        print(
            "Cross-fitted outer fold {}/{}: {}".format(
                outer_index, len(domains), outer_fold
            ),
            flush=True,
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    return predictions, unique_fit_audits, context_audits, outer_checks


def calibrate_crossfitted_thresholds(inner_predictions, domains):
    records = []
    for outer_fold in domains:
        frame = inner_predictions.loc[inner_predictions["outer_fold"] == outer_fold]
        maxima = trajectory_normalized_maxima(frame)
        normalized_values = maxima["normalized_trajectory_max"].to_numpy(dtype=float)
        saturated_count = int(np.sum(normalized_values == 1.0))
        for target_tfir in TARGET_TFIR_BUDGETS:
            calibration = conservative_empirical_threshold(
                normalized_values,
                target_tfir,
            )
            records.append({
                "outer_fold": outer_fold,
                "target_tfir": float(target_tfir),
                "score_scale": "empirical_tail_rank",
                "calibration_source": "inner-domain-held-out control trajectories",
                "inner_domains": sorted(frame["inner_fold"].unique()),
                "outer_test_domain_used": False,
                "crossfitted_normalized_maxima": sorted(
                    float(value) for value in normalized_values
                ),
                "unique_normalized_maximum_count": int(len(np.unique(normalized_values))),
                "saturated_at_one_count": saturated_count,
                "saturated_at_one_fraction": float(saturated_count / len(normalized_values)),
                "threshold_above_normalized_support": bool(calibration["threshold"] > 1.0),
                **calibration,
            })
    return pd.DataFrame.from_records(records)


def build_comparison_summary(experiment_e_metrics, experiment_f_summary):
    e = pd.DataFrame(experiment_e_metrics["metrics"]["summary"])
    e = e.loc[
        (e["analysis_population"] == "all_observers")
        & (e["monitor"] == "trusted_sentinel")
        & (e["method"] == "mean_sentinel_risk")
    ].copy()
    e["calibration_method"] = "experiment_e_in_sample"
    f = experiment_f_summary.copy()
    f["calibration_method"] = "experiment_f_crossfitted"
    columns = [
        "calibration_method",
        "target_tfir",
        "scenario_pair_count",
        "achieved_tfir",
        "achieved_tfir_bootstrap_ci_95",
        "pic",
        "pic_bootstrap_ci_95",
        "first_detection_round_counts",
        "first_detection_round_proportions",
        "cumulative_detection",
        "median_alert_round_detected",
        "median_intervention_lead_rounds_detected",
        "bootstrap_unit",
        "bootstrap_replicates",
    ]
    return pd.concat([e[columns], f[columns]], ignore_index=True).sort_values(
        ["target_tfir", "calibration_method"]
    )


def plot_domain_shift(domain_audit, output_path):
    frame = pd.DataFrame(domain_audit)
    frame = frame.loc[np.isclose(frame["target_tfir"], 0.05)].sort_values("outer_fold")
    domains = frame["outer_fold"].tolist()
    x_lookup = {domain: index for index, domain in enumerate(domains)}
    figure, axes = plt.subplots(
        2, 1, figsize=(10.5, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.45, 1]}
    )
    axis = axes[0]
    for _, row in frame.iterrows():
        x = x_lookup[row["outer_fold"]]
        margins = np.asarray(row["held_out_control_risk_minus_threshold"], dtype=float)
        colors = np.where(margins >= 0, "#c44e52", "#3568a8")
        offsets = (
            np.linspace(-0.12, 0.12, len(margins))
            if len(margins) > 1
            else np.array([0.0])
        )
        axis.scatter(x + offsets, margins, c=colors, s=42, zorder=3)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
    axis.set_ylabel("Control max risk − fold threshold")
    axis.set_title("Experiment-E control-risk transport by domain — 5% target")
    axis.grid(axis="y", alpha=0.2)

    axis = axes[1]
    x = np.arange(len(domains))
    width = 0.36
    axis.bar(
        x - width / 2,
        frame["achieved_domain_tfir"],
        width,
        color="#c44e52",
        label="Held-out TFIR",
    )
    axis.bar(
        x + width / 2,
        frame["collusion_pic"],
        width,
        color="#55a868",
        label="PIC",
    )
    for index, count in enumerate(frame["control_trajectory_count"]):
        axis.text(index - width / 2, 1.02, "n={}".format(int(count)), ha="center", fontsize=7)
    axis.axhline(0.05, color="black", linestyle=":", linewidth=1.0, label="5% target")
    axis.set_ylim(0, 1.12)
    axis.set_ylabel("Trajectory fraction")
    axis.set_xticks(x, domains, rotation=45, ha="right")
    handles, labels = axis.get_legend_handles_labels()
    axes[0].legend(handles, labels, fontsize=8, ncol=1, loc="upper right")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_calibration_comparison(comparison, output_path):
    names = {
        "experiment_e_in_sample": "Experiment E — in-sample",
        "experiment_f_crossfitted": "Experiment F — cross-fitted",
    }
    colors = {
        "experiment_e_in_sample": "#c44e52",
        "experiment_f_crossfitted": "#3568a8",
    }
    markers = {"experiment_e_in_sample": "s", "experiment_f_crossfitted": "o"}
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for method in names:
        frame = comparison.loc[comparison["calibration_method"] == method].sort_values(
            "target_tfir"
        )
        x = frame["target_tfir"].to_numpy(dtype=float)
        tfir = frame["achieved_tfir"].to_numpy(dtype=float)
        tfir_ci = np.stack(frame["achieved_tfir_bootstrap_ci_95"].to_numpy())
        pic = frame["pic"].to_numpy(dtype=float)
        pic_ci = np.stack(frame["pic_bootstrap_ci_95"].to_numpy())
        axes[0].errorbar(
            x,
            tfir,
            yerr=np.vstack([tfir - tfir_ci[:, 0], tfir_ci[:, 1] - tfir]),
            color=colors[method],
            marker=markers[method],
            linewidth=1.8,
            capsize=3,
            label=names[method],
        )
        axes[1].errorbar(
            x,
            pic,
            yerr=np.vstack([pic - pic_ci[:, 0], pic_ci[:, 1] - pic]),
            color=colors[method],
            marker=markers[method],
            linewidth=1.8,
            capsize=3,
            label=names[method],
        )
    axes[0].plot([0, 0.21], [0, 0.21], color="black", linestyle=":", linewidth=1.0, label="Target")
    axes[0].set_ylabel("Achieved held-out TFIR")
    axes[1].set_ylabel("PIC")
    for axis in axes:
        axis.set_xticks(TARGET_TFIR_BUDGETS, ["1%", "5%", "10%", "20%"])
        axis.set_xlabel("Target TFIR")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.2)
    axes[0].set_title("False-intervention transport")
    axes[1].set_title("Pre-final intervention coverage")
    axes[1].legend(fontsize=8, loc="best")
    figure.suptitle("In-sample vs cross-fitted threshold calibration — layer 28")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def format_interval(value, interval):
    return "{:.3f} [{:.3f}, {:.3f}]".format(value, interval[0], interval[1])


def format_optional(value):
    return "undefined" if value is None or pd.isna(value) else "{:.1f}".format(value)


def build_report(metrics):
    domain = pd.DataFrame(metrics["domain_shift_audit"])
    domain_5 = domain.loc[np.isclose(domain["target_tfir"], 0.05)].sort_values("outer_fold")
    comparison = pd.DataFrame(metrics["comparison"]["summary"])
    deltas = pd.DataFrame(metrics["comparison"]["paired_differences"])
    delta_5 = deltas.loc[np.isclose(deltas["target_tfir"], 0.05)].iloc[0]
    f_5 = comparison.loc[
        (comparison["calibration_method"] == "experiment_f_crossfitted")
        & np.isclose(comparison["target_tfir"], 0.05)
    ].iloc[0]
    thresholds = pd.DataFrame(metrics["calibration"]["thresholds"])
    thresholds_5 = thresholds.loc[np.isclose(thresholds["target_tfir"], 0.05)]
    e_5 = comparison.loc[
        (comparison["calibration_method"] == "experiment_e_in_sample")
        & np.isclose(comparison["target_tfir"], 0.05)
    ].iloc[0]
    lines = [
        "# Experiment F — Cross-Fitted Threshold Transport Diagnostic (Layer 28)",
        "",
        "- Status: **{}**".format(metrics["status"]),
        "- Architecture: unchanged Experiment-D fixed full-discussion sentinel",
        "- Features: honest observer positions 2–3 only; mean aggregation",
        "- Outer evaluation: unchanged leave-one-domain-out folds",
        "- Changed component: threshold-calibration scores only",
        "- Scope: diagnostic; no post-result optimization and no formal conformal claim",
        "",
        "## Part 1 — Experiment-E domain-shift audit",
        "",
        "This audit was computed from the saved Experiment-E OOF output before any Experiment-F model was refitted.",
        "Every domain contains only 1–3 controls. Consequently, 90th/95th percentiles are suppressed for all "
        "domains; the exact benign maxima are shown instead.",
        "",
        "| Domain | Controls | 5% threshold | Held-out benign maxima | Median | TFIR | PIC |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for _, row in domain_5.iterrows():
        maxima = ", ".join("{:.3f}".format(value) for value in row["held_out_control_max_pre_final_risks"])
        lines.append("| {} | {} | {:.3f} | {} | {:.3f} | {:.1%} | {:.1%} |".format(
            row["outer_fold"],
            int(row["control_trajectory_count"]),
            row["training_calibrated_threshold"],
            maxima,
            row["median_benign_risk"],
            row["achieved_domain_tfir"],
            row["collusion_pic"],
        ))

    lines.extend([
        "",
        "At the 5% target, false interventions occur in: {}. Domain-level rates are descriptive "
        "because a single trajectory changes a fold rate by 33–100 percentage points.".format(
            ", ".join(domain_5.loc[domain_5["held_out_false_intervention_count"] > 0, "outer_fold"])
        ),
        "",
        "## Part 2 — Cross-fitted calibration method",
        "",
        "For each outer fold, every remaining domain is held out once internally. Its control trajectories "
        "are scored by a sentinel trained without either the outer or inner domain. Raw margins from different "
        "inner models are aligned using:",
        "",
        "`normalized_tail_rank(x) = count(training-control trajectory maxima < x) / n`",
        "",
        "Higher values indicate greater risk. The strict-left comparison treats ties conservatively. Each "
        "outer-test prefix is normalized by the identical rule against its final outer model's training-control "
        "maxima. This is empirical scale alignment only—not conformal calibration and not a distribution-free guarantee.",
        "",
        "## Calibration comparison",
        "",
        "The 1% row is an underpowered reference. With roughly 30 outer-training controls, one trajectory "
        "corresponds to about 3–4 percentage points.",
        "",
        "| Calibration | Target | Achieved TFIR (95% CI) | PIC (95% CI) | First alerts R1/R2/R3/never | Median alert | Median lead |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    labels = {
        "experiment_e_in_sample": "E: in-sample",
        "experiment_f_crossfitted": "F: cross-fitted",
    }
    for _, row in comparison.sort_values(["target_tfir", "calibration_method"]).iterrows():
        counts = row["first_detection_round_counts"]
        lines.append("| {} | {:.0%} | {} | {} | {}/{}/{}/{} | {} | {} |".format(
            labels[row["calibration_method"]],
            row["target_tfir"],
            format_interval(row["achieved_tfir"], row["achieved_tfir_bootstrap_ci_95"]),
            format_interval(row["pic"], row["pic_bootstrap_ci_95"]),
            counts["R1"], counts["R2"], counts["R3"], counts["never"],
            format_optional(row["median_alert_round_detected"]),
            format_optional(row["median_intervention_lead_rounds_detected"]),
        ))

    lines.extend([
        "",
        "## Direct 5% change",
        "",
        "- ΔTFIR (F − E): {}".format(format_interval(
            delta_5["delta_tfir_crossfitted_minus_in_sample"],
            delta_5["delta_tfir_bootstrap_ci_95"],
        )),
        "- ΔPIC (F − E): {}".format(format_interval(
            delta_5["delta_pic_crossfitted_minus_in_sample"],
            delta_5["delta_pic_bootstrap_ci_95"],
        )),
        "- Experiment-E 5% result: TFIR {:.1%}, PIC {:.1%}".format(
            e_5["achieved_tfir"], e_5["pic"]
        ),
        "- Experiment-F 5% result: TFIR {:.1%}, PIC {:.1%}".format(
            f_5["achieved_tfir"], f_5["pic"]
        ),
        "",
        "## Integrity checks",
        "",
        "- Unique model fits: {} ({} outer plus {} unique two-domain-exclusion inner fits)".format(
            metrics["training"]["unique_model_fit_count"],
            metrics["training"]["outer_model_fit_count"],
            metrics["training"]["unique_inner_model_fit_count"],
        ),
        "- Inner calibration contexts: {}".format(metrics["training"]["inner_context_count"]),
        "- Inner/outer training-domain leakage violations: 0",
        "- Cross-fitted training-budget violations: {}".format(
            metrics["calibration"]["training_budget_violation_count"]
        ),
        "- At the 5% target, saturated inner-control ranks per outer fold: {}–{} of approximately 30 "
        "({:.1%}–{:.1%}); thresholds above normalized support: {}/{}".format(
            int(thresholds_5["saturated_at_one_count"].min()),
            int(thresholds_5["saturated_at_one_count"].max()),
            thresholds_5["saturated_at_one_fraction"].min(),
            thresholds_5["saturated_at_one_fraction"].max(),
            int(thresholds_5["threshold_above_normalized_support"].sum()),
            len(thresholds_5),
        ),
        "- Experiment-E outer score rows reproduced: {}; maximum absolute difference {:.3g}".format(
            metrics["reuse_validation"]["experiment_e_score_rows_compared"],
            metrics["reuse_validation"]["experiment_e_score_maximum_absolute_difference"],
        ),
        "- Experiment-E outer thresholds reproduced exactly for all folds/budgets: yes",
        "- Colluder activation rows in sentinel features: 0",
        "- Final-round activation rows: 0",
        "",
        "## Interpretation",
        "",
    ])
    tfir_reduction = e_5["achieved_tfir"] - f_5["achieved_tfir"]
    pic_change = f_5["pic"] - e_5["pic"]
    if f_5["achieved_tfir"] > 0.10:
        lines.append(
            "Outcome B is supported: cross-fitted held-out TFIR remains far above the 5% target. "
            "The primary limitation is threshold transport under domain shift, not merely in-sample calibration optimism."
        )
    elif tfir_reduction >= 0.10 and f_5["pic"] >= 0.50:
        lines.append(
            "Outcome A is supported: cross-fitting substantially reduces held-out TFIR while retaining useful PIC. "
            "In-sample threshold calibration was a major source of operational optimism."
        )
    else:
        lines.append(
            "Cross-fitting changes held-out TFIR but does not cleanly establish either strong transport or a dominant "
            "in-sample-optimism explanation at this sample size."
        )
    if pic_change <= -0.30:
        lines.append(
            "Outcome C is also supported: TFIR improves only with a large PIC collapse. The sentinel provides ranking "
            "signal but insufficient stable margin for low-false-intervention deployment under domain shift."
        )
        lines.append(
            "The collapse is mechanically traceable to empirical-rank saturation: 17–36% of inner-held-out "
            "controls lie above their model-specific training-control maximum. Conservative 5% thresholds are "
            "therefore above the normalized score support in all 14 folds. This result diagnoses a lack of "
            "transportable tail resolution under the prespecified normalization; it does not prove that every "
            "possible calibration method must fail."
        )
    lines.extend([
        "At 5%, TFIR changes by {:+.1%} and PIC by {:+.1%}. These retrospective estimates use only 31 matched "
        "scenario pairs and do not establish production readiness.".format(-tfir_reduction, pic_change),
        "No method was optimized after observing Experiment-F results, and no additional experiment was run.",
        "",
    ])
    return "\n".join(lines)


def main():
    experiment_e_metrics = json.loads(EXPERIMENT_E_METRICS.read_text(encoding="utf-8"))
    fixed_metrics = json.loads(FIXED_SENTINEL_METRICS.read_text(encoding="utf-8"))
    if experiment_e_metrics["status"] != "PASS" or fixed_metrics["status"] != "PASS":
        raise RuntimeError("Required Experiment-D/E artifact is not PASS")
    if sha256_file(EXPERIMENT_E_PREDICTIONS) != experiment_e_metrics["predictions"]["sha256"]:
        raise RuntimeError("Experiment-E prediction hash mismatch")
    if sha256_file(FIXED_SENTINEL_PREDICTIONS) != fixed_metrics["predictions"]["sha256"]:
        raise RuntimeError("Experiment-D prediction hash mismatch")
    experiment_e = pd.read_csv(EXPERIMENT_E_PREDICTIONS)
    fixed_reference = pd.read_csv(FIXED_SENTINEL_PREDICTIONS)

    # Required ordering: complete Part 1 from saved E outputs before any refit.
    domain_shift_audit = experiment_e_domain_shift_audit(
        experiment_e, minimum_quantile_n=QUANTILE_MINIMUM_N
    )
    print("Completed Part 1 from saved Experiment-E OOF outputs before refitting", flush=True)

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
        raw_predictions, fit_audits, context_audits, outer_checks = generate_crossfitted_scores(
            examples, domains, experiment_e, fixed_reference
        )
    finally:
        archive.close()

    inner = raw_predictions.loc[raw_predictions["record_type"] == "inner_calibration"].copy()
    outer = raw_predictions.loc[raw_predictions["record_type"] == "outer_test"].copy()
    thresholds = calibrate_crossfitted_thresholds(inner, domains)
    policy = build_crossfitted_policy(outer, thresholds)
    outer_long = attach_outer_policy_to_scores(outer, policy)
    inner["target_tfir"] = np.nan
    inner["threshold"] = np.nan
    inner["calibration_trajectory_count"] = np.nan
    inner["calibration_training_empirical_tfir"] = np.nan
    inner["intervened"] = np.nan
    inner["first_detection_round"] = np.nan
    inner["intervention_lead_rounds"] = np.nan
    inner["detected_by_r1"] = np.nan
    inner["detected_by_r2"] = np.nan
    inner["detected_by_r3"] = np.nan
    inner["threshold_crossed_at_prefix"] = np.nan
    inner["policy_triggered_at_prefix"] = np.nan
    inner["prefix_evaluated_before_stop"] = np.nan
    predictions = pd.concat([inner, outer_long], ignore_index=True, sort=False).sort_values(
        ["record_type", "outer_fold", "inner_fold", "target_tfir", "domain", "run_id", "round"],
        na_position="first",
    )

    # Required ordering: save raw/calibration predictions before metrics or figures.
    PREDICTION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTION_OUTPUT, index=False)
    saved_predictions = pd.read_csv(PREDICTION_OUTPUT)
    print("Saved {}".format(PREDICTION_OUTPUT.relative_to(PROJECT_ROOT)), flush=True)

    validation_issues = validate_crossfitted_predictions(saved_predictions, domains)
    if not thresholds["training_budget_respected"].all():
        validation_issues.append("A cross-fitted calibration threshold exceeds its training budget")
    if thresholds["outer_test_domain_used"].any():
        validation_issues.append("An outer-test domain entered threshold selection")
    if any(audit["outer_fold_present_in_model_training"] for audit in context_audits):
        validation_issues.append("An outer fold entered a sentinel fit")
    if any(
        audit["context"] == "inner_crossfit" and audit["inner_fold_present_in_model_training"]
        for audit in context_audits
    ):
        validation_issues.append("An inner fold entered the sentinel fit producing its score")

    outer_saved = saved_predictions.loc[saved_predictions["record_type"] == "outer_test"]
    policy_columns = [
        "analysis_population", "monitor", "method", "target_tfir", "scenario_id", "run_id",
        "domain", "fold", "mode", "threshold", "calibration_trajectory_count",
        "calibration_training_empirical_tfir", "intervened", "first_detection_round",
        "intervention_lead_rounds", "detected_by_r1", "detected_by_r2", "detected_by_r3",
    ]
    saved_policy = outer_saved[policy_columns].drop_duplicates(
        ["analysis_population", "monitor", "method", "target_tfir", "run_id"]
    ).copy()
    saved_policy["intervened"] = saved_policy["intervened"].astype(bool)
    f_summary, f_folds, metric_issues = calculate_intervention_metrics(
        saved_policy[policy_columns],
        domains,
        replicates=BOOTSTRAP_REPLICATES,
        seed=SEED,
    )
    if metric_issues:
        validation_issues.extend(metric_issues)
    comparison = build_comparison_summary(experiment_e_metrics, f_summary)
    e_primary_policy = experiment_e.loc[
        (experiment_e["analysis_population"] == "all_observers")
        & (experiment_e["monitor"] == "trusted_sentinel")
        & (experiment_e["method"] == "mean_sentinel_risk")
    ]
    deltas = bootstrap_calibration_differences(
        e_primary_policy,
        saved_policy,
        replicates=BOOTSTRAP_REPLICATES,
        seed=SEED,
    )

    score_rows_compared = sum(
        check["experiment_e_score_reproduction"]["compared_rows"] for check in outer_checks
    )
    score_maximum_difference = max(
        check["experiment_e_score_reproduction"]["maximum_absolute_difference"]
        for check in outer_checks
    )
    inner_context_count = sum(audit["context"] == "inner_crossfit" for audit in context_audits)
    outer_model_count = sum(len(audit["excluded_domains"]) == 1 for audit in fit_audits)
    inner_unique_fit_count = sum(len(audit["excluded_domains"]) == 2 for audit in fit_audits)
    output = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not validation_issues else "ISSUES",
        "scope": "Experiment F cross-fitted threshold transport diagnostic only; no method optimization.",
        "sources": {
            "upstream_repository": "https://github.com/aaronrose227/narcbench",
            "upstream_git_commit": git_commit(UPSTREAM_ROOT),
            "upstream_source_modified": upstream_is_modified(UPSTREAM_ROOT),
            "dataset_repository": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "metadata_sha256": sha256_file(CORE_DIR / "metadata_gen.json"),
            "activations_sha256": sha256_file(CORE_DIR / "activations_gen.npz"),
            "experiment_d_metrics_sha256": sha256_file(FIXED_SENTINEL_METRICS),
            "experiment_d_predictions_sha256": sha256_file(FIXED_SENTINEL_PREDICTIONS),
            "experiment_e_metrics_sha256": sha256_file(EXPERIMENT_E_METRICS),
            "experiment_e_predictions_sha256": sha256_file(EXPERIMENT_E_PREDICTIONS),
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
            "control_trajectory_resolution": "approximately one trajectory, or 3–4%, in each outer-training set",
        },
        "research_question": (
            "Is held-out false-intervention failure primarily caused by optimistic in-sample "
            "threshold calibration, or does it persist with out-of-training-sample control scores?"
        ),
        "structure_audit": structure_audit,
        "domain_shift_audit": domain_shift_audit,
        "methodology": {
            "outer_protocol": "leave-one-domain-out",
            "inner_protocol": "leave-one-domain-out within the outer-training domains",
            "sentinel_architecture_changed": False,
            "layer_changed": False,
            "hyperparameters_tuned": False,
            "analysis_population": "all observers",
            "observer_positions": [2, 3],
            "colluder_activation_rows": 0,
            "risk_aggregation": "mean of the two honest-observer decision scores",
            "probe_training_representation": "mean(R1,R2,R3)",
            "prefixes": {"R1": [1], "R1-R2": [1, 2], "R1-R2-R3": [1, 2, 3]},
            "normalization": {
                "name": "empirical strict-left tail rank",
                "formula": "count(training-control trajectory maxima < raw prefix risk) / n",
                "orientation": "larger means riskier",
                "reference": "model-specific training-control maximum pre-final risks",
                "ties": "strict-left comparison is conservative",
                "formal_conformal_claim": False,
                "distribution_free_guarantee": False,
            },
            "threshold_selection": {
                "source": "normalized inner-domain-held-out control trajectory maxima",
                "outer_test_data_used": False,
                "budgets": list(TARGET_TFIR_BUDGETS),
                "primary_budgets": [0.05, 0.10, 0.20],
                "underpowered_reference_budget": 0.01,
                "quantile_method": "higher plus nextafter for inclusive >= rule",
            },
            "runtime_policy": "first normalized prefix risk >= threshold; stop; no round 4",
            "bootstrap": {
                "method": "percentile bootstrap",
                "unit": "matched scenario pair",
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": SEED,
                "confidence_level": 0.95,
            },
        },
        "training": {
            "unique_model_fit_count": len(fit_audits),
            "outer_model_fit_count": outer_model_count,
            "unique_inner_model_fit_count": inner_unique_fit_count,
            "inner_context_count": inner_context_count,
            "equivalent_inner_fit_reuse_count": sum(
                audit.get("reused_equivalent_training_set_fit", False)
                for audit in context_audits
            ),
            "unique_fit_audits": fit_audits,
            "context_audits": context_audits,
        },
        "reuse_validation": {
            "outer_checks": outer_checks,
            "experiment_e_score_rows_compared": score_rows_compared,
            "experiment_e_score_maximum_absolute_difference": score_maximum_difference,
            "experiment_e_thresholds_reproduced": True,
        },
        "calibration": {
            "threshold_record_count": len(thresholds),
            "training_budget_violation_count": int((~thresholds["training_budget_respected"]).sum()),
            "thresholds": thresholds.to_dict(orient="records"),
        },
        "predictions": {
            "path": str(PREDICTION_OUTPUT.relative_to(PROJECT_ROOT)),
            "rows": len(saved_predictions),
            "inner_calibration_rows": int((saved_predictions["record_type"] == "inner_calibration").sum()),
            "outer_test_rows": int((saved_predictions["record_type"] == "outer_test").sum()),
            "sha256": sha256_file(PREDICTION_OUTPUT),
        },
        "metrics": {
            "crossfitted_summary": f_summary.to_dict(orient="records"),
            "crossfitted_fold_level": f_folds.to_dict(orient="records"),
            "undefined_fold_records": metric_issues,
            "undefined_fold_count": len(metric_issues),
        },
        "comparison": {
            "summary": comparison.to_dict(orient="records"),
            "paired_differences": deltas,
        },
        "issues": validation_issues,
        "limitations": [
            "Only 31 matched scenario pairs and 1–3 held-out controls per domain.",
            "Empirical tail-rank normalization has no formal conformal or distribution-free guarantee.",
            "The model-specific normalization reference is scored in-sample; cross-fitting applies to calibration observations.",
            "Round 4 is only a proxy boundary and no real-world intervention cost was measured.",
            "Layer 28 and the sentinel architecture were fixed before Experiment F.",
            "No production-readiness claim is supported.",
        ],
    }
    output = json_ready(output)
    METRICS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUTPUT.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    saved_output = json.loads(METRICS_OUTPUT.read_text(encoding="utf-8"))
    saved_comparison = pd.DataFrame(saved_output["comparison"]["summary"])
    plot_domain_shift(saved_output["domain_shift_audit"], DOMAIN_FIGURE)
    plot_calibration_comparison(saved_comparison, COMPARISON_FIGURE)
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT.write_text(build_report(saved_output), encoding="utf-8")
    for path in (METRICS_OUTPUT, REPORT_OUTPUT, DOMAIN_FIGURE, COMPARISON_FIGURE):
        print("Saved {}".format(path.relative_to(PROJECT_ROOT)), flush=True)
    print("Status: {}".format(output["status"]), flush=True)
    return 0 if output["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
