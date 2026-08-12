#!/usr/bin/env python3
"""Run Phase 2: layer-28 temporal detectability on NARCBench-Core."""

import hashlib
import json
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
CORE_DIR = PROJECT_ROOT / "data" / "narcbench" / "activations" / "qwen3_32b" / "core"
AGENT_OUTPUT = PROJECT_ROOT / "outputs" / "predictions" / "oof_agent_temporal_scores_layer28.csv"
RUN_OUTPUT = PROJECT_ROOT / "outputs" / "predictions" / "oof_temporal_scores_layer28.csv"
METRICS_OUTPUT = PROJECT_ROOT / "outputs" / "metrics" / "temporal_detectability_layer28.json"
REPORT_OUTPUT = PROJECT_ROOT / "outputs" / "logs" / "temporal_detectability_layer28.md"
ROUND_FIGURE = PROJECT_ROOT / "outputs" / "figures" / "round_only_detectability_layer28.png"
PREFIX_FIGURE = PROJECT_ROOT / "outputs" / "figures" / "prefix_detectability_layer28.png"

LAYER = 28
MODEL_ID = "Qwen/Qwen3-32B-AWQ"
DATASET_ID = "aaronrose227/narcbench"
DATASET_REVISION = "19cb14b269e052b19ba6d2bd2f4519afd3b10c57"


sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(UPSTREAM_ROOT))
from probes import reproduce as upstream  # noqa: E402
from src.temporal_detectability import (  # noqa: E402
    METHODS,
    TEMPORAL_VIEWS,
    aggregate_run_scores,
    array_sha256,
    build_public_discussion_index,
    calculate_auroc_metrics,
    official_probe_candidate_indices,
    probe_fingerprint,
    score_fold_agents,
    validate_oof_predictions,
)


def git_commit(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def audit_fold_training(metadata, activations, held_out_domain, scaler, direction):
    audit = official_probe_candidate_indices(
        metadata, held_out_domain, upstream._domain_from_scenario_id
    )
    selected = audit["selected_honest"] + audit["selected_colluder"]
    selected_domains = sorted({metadata[index]["scenario_domain"] for index in selected})
    selected_matrix = activations[selected]
    expected_mean = np.mean(selected_matrix, axis=0, dtype=np.float64)
    expected_var = np.var(selected_matrix, axis=0, dtype=np.float64)
    scaler_matches_selected = bool(
        np.allclose(scaler.mean_, expected_mean, rtol=1e-10, atol=1e-10)
        and np.allclose(scaler.var_, expected_var, rtol=1e-10, atol=1e-10)
        and int(scaler.n_samples_seen_) == len(selected)
    )
    if held_out_domain in selected_domains:
        raise RuntimeError("Held-out domain entered the official probe training rows")
    if not scaler_matches_selected:
        raise RuntimeError("Fitted scaler does not match audited upstream training rows")
    return {
        "held_out_domain": held_out_domain,
        "candidate_rows": {
            "honest": len(audit["candidate_honest"]),
            "colluder": len(audit["candidate_colluder"]),
        },
        "selected_training_rows": {
            "honest": len(audit["selected_honest"]),
            "colluder": len(audit["selected_colluder"]),
            "total": len(selected),
        },
        "selected_training_domains": selected_domains,
        "held_out_domain_present_in_training": False,
        "scaler_matches_audited_selected_rows": scaler_matches_selected,
        "scaler": {
            "n_samples_seen": int(scaler.n_samples_seen_),
            "n_features_in": int(scaler.n_features_in_),
            "mean_sha256": array_sha256(scaler.mean_),
            "scale_sha256": array_sha256(scaler.scale_),
            "variance_sha256": array_sha256(scaler.var_),
        },
        "deception_direction": {
            "length": int(direction.shape[0]),
            "l2_norm": float(np.linalg.norm(direction)),
            "sha256": array_sha256(direction),
        },
        "probe_fingerprint": probe_fingerprint(scaler, direction),
    }


def plot_view_from_saved_predictions(run_output, summary_frame, temporal_view, output_path):
    """Make a basic plot after reloading the saved run-level prediction CSV."""
    saved_runs = pd.read_csv(run_output)
    view_runs = saved_runs.loc[saved_runs["temporal_view"] == temporal_view].copy()
    if view_runs.empty:
        raise RuntimeError("Saved predictions have no {} rows".format(temporal_view))
    view_metrics = summary_frame.loc[summary_frame["temporal_view"] == temporal_view].copy()
    view_metrics = view_metrics.sort_values("round")
    labels = (
        ["R1", "R2", "R3"]
        if temporal_view == "round_only"
        else ["R1", "R1-R2", "R1-R2-R3"]
    )
    colors = {"peak_suspicion": "#3568a8", "deception_split": "#d06735"}
    names = {"peak_suspicion": "Peak Suspicion", "deception_split": "Deception Split"}
    offsets = {"peak_suspicion": -0.025, "deception_split": 0.025}
    markers = {"peak_suspicion": "o", "deception_split": "s"}

    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    for method in METHODS:
        method_metrics = view_metrics.loc[view_metrics["method"] == method].sort_values("round")
        if len(method_metrics) != 3:
            raise RuntimeError("Expected three saved metric points for " + method)
        x_values = np.arange(3) + offsets[method]
        means = method_metrics["fold_macro_mean_auroc"].to_numpy(dtype=float)
        stds = method_metrics["fold_population_std_auroc"].to_numpy(dtype=float)
        pooled = method_metrics["pooled_oof_auroc"].to_numpy(dtype=float)
        axis.errorbar(
            x_values,
            means,
            yerr=stds,
            marker=markers[method],
            linewidth=1.8,
            capsize=3,
            color=colors[method],
            label=names[method] + " (fold mean ± SD)",
        )
        axis.plot(
            x_values,
            pooled,
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            color=colors[method],
            label=names[method] + " (pooled OOF)",
        )

    axis.axhline(0.5, color="grey", linestyle=":", linewidth=1.0)
    axis.set_xticks(np.arange(3), labels)
    axis.set_ylim(0.45, 1.02)
    axis.set_ylabel("Run-level AUROC")
    axis.set_xlabel("Round" if temporal_view == "round_only" else "Available prefix")
    axis.set_title(
        "Round-only detectability (Layer 28)"
        if temporal_view == "round_only"
        else "Prefix detectability (Layer 28)"
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8, loc="lower right")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_report(metrics):
    summary = pd.DataFrame(metrics["metrics"]["summary"])
    lines = [
        "# Phase 2: Temporal Detectability (Layer 28)",
        "",
        "- Status: **{}**".format(metrics["status"]),
        "- Probe: one fixed upstream scaler/direction per leave-domain-out fold",
        "- Folds: {}".format(len(metrics["dataset"]["domains"])),
        "- Metric issues: {}".format(len(metrics["metrics"]["issues"])),
        "",
        "| View | Time | Method | Fold mean AUROC | Fold SD | Pooled OOF AUROC |",
        "|---|---|---|---:|---:|---:|",
    ]
    for _, row in summary.sort_values(["temporal_view", "round", "method"]).iterrows():
        lines.append(
            "| {} | {} | {} | {:.6f} | {:.6f} | {:.6f} |".format(
                row["temporal_view"],
                row["round_or_prefix"],
                row["method"],
                row["fold_macro_mean_auroc"],
                row["fold_population_std_auroc"],
                row["pooled_oof_auroc"],
            )
        )
    lines.extend([
        "",
        "Predictions were saved before metrics and figures were computed. No final-round "
        "activation, threshold calibration, TFIR, PIC, or E2 intervention logic was used.",
        "",
    ])
    return "\n".join(lines)


def main():
    metadata, archive = upstream.load_data(CORE_DIR)
    try:
        activations = upstream.get_layer(archive, len(metadata), LAYER)
        temporal_index, run_metadata = build_public_discussion_index(metadata, activations)
        domains = sorted({values["domain"] for values in run_metadata.values()})
        agent_records = []
        training_audits = []

        for held_out_domain in domains:
            print("Scoring held-out domain: {}".format(held_out_domain), flush=True)
            scaler, direction = upstream.train_deception_probe(
                metadata, archive, LAYER, exclude_domain=held_out_domain
            )
            training_audits.append(
                audit_fold_training(metadata, activations, held_out_domain, scaler, direction)
            )
            agent_records.extend(
                score_fold_agents(
                    temporal_index,
                    run_metadata,
                    held_out_domain,
                    scaler,
                    direction,
                    LAYER,
                )
            )
    finally:
        archive.close()

    agent_frame = pd.DataFrame.from_records(agent_records).sort_values(
        ["fold", "run_id", "agent_name", "temporal_view", "round"]
    )
    run_frame = aggregate_run_scores(agent_frame).sort_values(
        ["fold", "run_id", "temporal_view", "round", "method"]
    )

    # Required ordering: persist raw OOF predictions before metric calculation or plotting.
    AGENT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    agent_frame.to_csv(AGENT_OUTPUT, index=False)
    run_frame.to_csv(RUN_OUTPUT, index=False)
    print("Saved {}".format(AGENT_OUTPUT), flush=True)
    print("Saved {}".format(RUN_OUTPUT), flush=True)

    saved_agent_frame = pd.read_csv(AGENT_OUTPUT)
    saved_run_frame = pd.read_csv(RUN_OUTPUT)
    integrity_issues = validate_oof_predictions(saved_agent_frame, saved_run_frame, domains)
    fold_metrics, summary_metrics, metric_issues = calculate_auroc_metrics(
        saved_run_frame, domains
    )
    all_issues = integrity_issues + metric_issues

    output = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not all_issues else "ISSUES",
        "scope": "Phase 2 temporal detectability only; no threshold calibration or intervention metrics.",
        "sources": {
            "upstream_repository": "https://github.com/aaronrose227/narcbench",
            "upstream_git_commit": git_commit(UPSTREAM_ROOT),
            "upstream_source_modified": False,
            "dataset_repository": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "metadata_sha256": sha256_file(CORE_DIR / "metadata_gen.json"),
            "activations_sha256": sha256_file(CORE_DIR / "activations_gen.npz"),
        },
        "environment": {
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
            "cpu_only": True,
        },
        "model": MODEL_ID,
        "dataset": {
            "tier": "Core",
            "layer": LAYER,
            "metadata_rows": len(metadata),
            "run_count": len(run_metadata),
            "scenario_count": len(run_metadata) // 2,
            "domains": domains,
        },
        "methodology": {
            "fold_protocol": "leave-one-domain-out",
            "probe_fit_count": len(domains),
            "probe_reuse": "One upstream-fitted scaler/direction reused across all six temporal views in each fold.",
            "probe_training": {
                "upstream_function": "probes.reproduce.train_deception_probe",
                "mode": "collusion",
                "channel": "public",
                "phase": "discussion",
                "unit": "agent-turn activation row",
                "labels": {"honest": 0, "colluder": 1},
                "balancing": "seed-42 honest-row downsampling without replacement",
                "scaler": "StandardScaler()",
                "classifier": "LogisticRegression(C=1.0, max_iter=1000, random_state=42)",
                "direction": "L2-normalized fitted coefficient vector",
            },
            "temporal_views": {
                "round_only": {"R1": [1], "R2": [2], "R3": [3]},
                "prefix": {"R1": [1], "R1-R2": [1, 2], "R1-R2-R3": [1, 2, 3]},
            },
            "monitors": {
                "peak_suspicion": "maximum agent deception score",
                "deception_split": "mean(top 2 agent scores) - mean(bottom 2 agent scores)",
            },
            "fold_summary": "unweighted mean and population standard deviation across domains",
            "pooled_summary": "AUROC over all out-of-fold run predictions",
        },
        "training_audits": training_audits,
        "predictions": {
            "agent_path": str(AGENT_OUTPUT.relative_to(PROJECT_ROOT)),
            "agent_rows": len(saved_agent_frame),
            "agent_sha256": sha256_file(AGENT_OUTPUT),
            "run_path": str(RUN_OUTPUT.relative_to(PROJECT_ROOT)),
            "run_rows": len(saved_run_frame),
            "run_sha256": sha256_file(RUN_OUTPUT),
        },
        "metrics": {
            "summary": summary_metrics.to_dict(orient="records"),
            "fold_level": fold_metrics.to_dict(orient="records"),
            "issues": all_issues,
        },
    }
    output = json_ready(output)
    METRICS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT.write_text(build_report(output), encoding="utf-8")

    plot_view_from_saved_predictions(RUN_OUTPUT, summary_metrics, "round_only", ROUND_FIGURE)
    plot_view_from_saved_predictions(RUN_OUTPUT, summary_metrics, "prefix", PREFIX_FIGURE)

    print("Saved {}".format(METRICS_OUTPUT), flush=True)
    print("Saved {}".format(REPORT_OUTPUT), flush=True)
    print("Saved {}".format(ROUND_FIGURE), flush=True)
    print("Saved {}".format(PREFIX_FIGURE), flush=True)
    print("Status: {}".format(output["status"]), flush=True)
    return 0 if output["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
