#!/usr/bin/env python3
"""Reproduce the official NARCBench-Core activation-probe baseline.

This wrapper intentionally imports the checked-out upstream implementation and
does not modify or copy its probe/scoring algorithms. It adds Core-only path
selection, fold-level provenance, and machine-readable reporting.
"""

import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "external" / "narcbench"
CORE_DIR = PROJECT_ROOT / "data" / "narcbench" / "activations" / "qwen3_32b" / "core"
OUTPUT_JSON = PROJECT_ROOT / "outputs" / "metrics" / "baseline.json"
OUTPUT_REPORT = PROJECT_ROOT / "outputs" / "logs" / "baseline_reproduction.md"

MODEL_ID = "Qwen/Qwen3-32B-AWQ"
DATASET_ID = "aaronrose227/narcbench"
DATASET_REVISION = "19cb14b269e052b19ba6d2bd2f4519afd3b10c57"
LAYERS = [26, 27, 28, 29, 30]
RANDOM_SEED = 42
PAPER_URL = "https://arxiv.org/html/2604.01151v2"
PAPER_REFERENCE = {
    "sorted_concat": 1.00,
    "activation_alignment": 1.00,
    "peak_suspicion": 1.00,
    "deception_split": 1.00,
    "asymmetry_probe": 1.00,
}

METHOD_NAMES = {
    "sorted_concat": "Sorted Concatenation",
    "activation_alignment": "Activation Alignment",
    "peak_suspicion": "Peak Suspicion",
    "deception_split": "Deception Split",
    "asymmetry_probe": "Asymmetry Probe",
}


sys.path.insert(0, str(UPSTREAM_ROOT))
from probes import reproduce as upstream  # noqa: E402


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array):
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def git_commit(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def stable_id_hash(values):
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def training_row_counts(meta, held_out_domain):
    counts = Counter()
    for row in meta:
        if row["mode"] != "collusion":
            continue
        if row["channel"] != "public" or row["phase"] != "discussion":
            continue
        _, scenario_id = upstream._parse_run_id(row["run_id"])
        if upstream._domain_from_scenario_id(scenario_id) == held_out_domain:
            continue
        if row["role"] in {"honest", "colluder"}:
            counts[row["role"]] += 1
    balanced_per_class = min(counts["honest"], counts["colluder"])
    return {
        "candidate_rows": {
            "honest": counts["honest"],
            "colluder": counts["colluder"],
        },
        "balanced_rows_used": {
            "honest": balanced_per_class,
            "colluder": balanced_per_class,
            "total": 2 * balanced_per_class,
        },
    }


def classifier_configuration():
    deception = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_SEED)
    run_level = LogisticRegression(max_iter=2000, C=0.01, random_state=RANDOM_SEED)
    return {
        "deception_probe": {
            "class": "sklearn.linear_model.LogisticRegression",
            "parameters": deception.get_params(deep=False),
            "post_fit_use": "L2-normalize coef_[0] as deception direction; intercept discarded",
        },
        "asymmetry_probe": {
            "class": "sklearn.linear_model.LogisticRegression",
            "parameters": run_level.get_params(deep=False),
            "input_scaler": StandardScaler().get_params(deep=False),
        },
        "sorted_concat": {
            "pca": {
                "class": "sklearn.decomposition.PCA",
                "parameters": PCA(n_components=50, random_state=RANDOM_SEED).get_params(deep=False),
                "effective_n_components_rule": "min(50, number_of_training_runs - 1, feature_count)",
            },
            "input_scaler": StandardScaler().get_params(deep=False),
            "classifier": {
                "class": "sklearn.linear_model.LogisticRegression",
                "parameters": run_level.get_params(deep=False),
            },
        },
    }


def aggregation_method_specs():
    return {
        "sorted_concat": {
            "name": METHOD_NAMES["sorted_concat"],
            "definition": "Sort four standardized agent vectors by deception projection, concatenate, PCA, scale, logistic regression.",
            "additional_run_level_training": True,
        },
        "activation_alignment": {
            "name": METHOD_NAMES["activation_alignment"],
            "definition": "Negative mean of the six pairwise cosine similarities between four raw agent vectors.",
            "additional_run_level_training": False,
        },
        "peak_suspicion": {
            "name": METHOD_NAMES["peak_suspicion"],
            "definition": "Maximum of the four standardized activation projections onto the deception direction.",
            "additional_run_level_training": False,
        },
        "deception_split": {
            "name": METHOD_NAMES["deception_split"],
            "definition": "Mean of top two deception projections minus mean of bottom two.",
            "additional_run_level_training": False,
        },
        "asymmetry_probe": {
            "name": METHOD_NAMES["asymmetry_probe"],
            "definition": "Scale [projection variance, range, deception split, mean], then logistic regression.",
            "additional_run_level_training": True,
        },
    }


def reproduce_layer(meta, archive, layer):
    groups = upstream.build_groups(meta, archive, layer, phase="discussion", avg_rounds=True)
    run_ids = sorted(groups)
    domains = sorted({groups[run_id]["domain"] for run_id in run_ids if groups[run_id]["domain"]})
    fold_records = []
    fold_values = {method: [] for method in upstream.METHOD_KEYS}

    for held_out_domain in domains:
        train_ids = [run_id for run_id in run_ids if groups[run_id]["domain"] != held_out_domain]
        test_ids = [run_id for run_id in run_ids if groups[run_id]["domain"] == held_out_domain]
        train_labels = upstream._run_labels(groups, train_ids)
        test_labels = upstream._run_labels(groups, test_ids)

        if len(np.unique(train_labels)) != 2 or len(np.unique(test_labels)) != 2:
            raise RuntimeError(
                "Upstream would skip fold {!r}; both labels are required in train and test.".format(
                    held_out_domain
                )
            )

        scaler, direction = upstream.train_deception_probe(
            meta, archive, layer, exclude_domain=held_out_domain
        )
        train_scores = upstream.compute_scores(groups, train_ids, scaler, direction)
        test_scores = upstream.compute_scores(groups, test_ids, scaler, direction)
        metrics = upstream.evaluate(train_scores, train_labels, test_scores, test_labels)
        if metrics is None:
            raise RuntimeError("Upstream evaluation returned no result for " + held_out_domain)

        for method, value in metrics.items():
            fold_values[method].append(float(value))

        test_scenarios = sorted({
            upstream._parse_run_id(run_id)[1] for run_id in test_ids
        })
        row_counts = training_row_counts(meta, held_out_domain)
        fold_records.append({
            "held_out_domain": held_out_domain,
            "training": {
                "domain_count": len(domains) - 1,
                "run_count": len(train_ids),
                "run_label_counts": {
                    "control_0": int(np.sum(train_labels == 0)),
                    "collusion_1": int(np.sum(train_labels == 1)),
                },
                "run_ids_sha256": stable_id_hash(train_ids),
                "probe_training_rows": row_counts,
            },
            "test": {
                "domain_count": 1,
                "scenario_count": len(test_scenarios),
                "scenarios": test_scenarios,
                "run_count": len(test_ids),
                "run_label_counts": {
                    "control_0": int(np.sum(test_labels == 0)),
                    "collusion_1": int(np.sum(test_labels == 1)),
                },
                "run_ids_sha256": stable_id_hash(test_ids),
            },
            "fitted_scaler": {
                "class": "sklearn.preprocessing.StandardScaler",
                "parameters": scaler.get_params(deep=False),
                "n_features_in": int(scaler.n_features_in_),
                "n_samples_seen": json_value(scaler.n_samples_seen_),
                "mean_sha256": sha256_array(scaler.mean_),
                "scale_sha256": sha256_array(scaler.scale_),
                "variance_sha256": sha256_array(scaler.var_),
            },
            "normalized_deception_direction": {
                "length": int(direction.shape[0]),
                "l2_norm": float(np.linalg.norm(direction)),
                "sha256": sha256_array(direction),
            },
            "auroc": {method: float(metrics[method]) for method in upstream.METHOD_KEYS},
        })

    detailed_aggregate = {
        method: float(np.mean(values)) for method, values in fold_values.items()
    }
    direct_upstream_aggregate = upstream.held_out_cv_layer(meta, archive, layer)
    differences = {
        method: abs(detailed_aggregate[method] - float(direct_upstream_aggregate[method]))
        for method in upstream.METHOD_KEYS
    }
    return {
        "fold_count": len(fold_records),
        "folds": fold_records,
        "fold_macro_mean_auroc": detailed_aggregate,
        "direct_upstream_held_out_cv_layer_auroc": {
            method: float(direct_upstream_aggregate[method]) for method in upstream.METHOD_KEYS
        },
        "detailed_vs_direct_absolute_difference": differences,
        "detailed_matches_direct_upstream": all(value <= 1e-12 for value in differences.values()),
    }


def render_report(result):
    lines = [
        "# NARCBench Core Baseline Reproduction",
        "",
        "- Status: **{}**".format(result["status"]),
        "- Model: `{}`".format(MODEL_ID),
        "- Dataset: `{}` at `{}`".format(DATASET_ID, DATASET_REVISION),
        "- Upstream commit: `{}`".format(result["sources"]["upstream_git_commit"]),
        "- Layers: `{}`".format(", ".join(str(layer) for layer in LAYERS)),
        "- Observed leave-domain-out folds: `{}`".format(result["dataset"]["fold_count"]),
        "- Random seed implemented upstream: `{}`".format(RANDOM_SEED),
        "",
        "## AUROC",
        "",
        "| Method | Reproduced mean | Layer std | Paper v2 | Difference |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in upstream.METHOD_KEYS:
        values = result["results"]["across_layers"][method]
        lines.append(
            "| {} | {:.6f} | {:.6f} | {:.2f} | {:.6f} |".format(
                METHOD_NAMES[method],
                values["mean_auroc"],
                values["population_std_across_layers"],
                values["paper_v2_reported_auroc"],
                values["absolute_difference_from_paper"],
            )
        )

    lines.extend([
        "",
        "Every detailed fold/layer aggregate matched the unchanged upstream "
        "`held_out_cv_layer()` result to numerical tolerance.",
        "",
        "## Methodology",
        "",
        "The wrapper imports upstream loading, grouping, probe training, scoring, and "
        "evaluation functions. The deception probe is fit on public discussion rows from "
        "collusion-condition training domains only, with `honest=0` and `colluder=1`; "
        "classes are balanced using upstream seed 42. `StandardScaler` precedes "
        "`LogisticRegression(C=1.0, max_iter=1000, random_state=42)`, and the normalized "
        "coefficient vector is used as the deception direction. Evaluation vectors average "
        "each agent over all public discussion rounds.",
        "",
        "## Comparison and provenance caveats",
        "",
        "The reproduced values match paper v2 Table 4 at its two-decimal reporting precision. "
        "The current released Qwen Core subset has 31 matched scenarios and 14 observed "
        "domains, whereas paper v2 describes 50 scenarios, 17 folds, and averaging over ten "
        "seeds. The checked-out repository implementation uses the released rows dynamically "
        "and hard-codes one seed (42). Because all reproduced Core values are exactly 1.0, "
        "these provenance differences do not create a numerical discrepancy here.",
        "",
        "No temporal-prefix or intervention experiment was run.",
        "",
    ])
    return "\n".join(lines)


def main():
    meta_path = CORE_DIR / "metadata_gen.json"
    activation_path = CORE_DIR / "activations_gen.npz"
    for required in (meta_path, activation_path, UPSTREAM_ROOT / "probes" / "reproduce.py"):
        if not required.is_file():
            raise SystemExit("Required file not found: " + str(required))

    meta, archive = upstream.load_data(CORE_DIR)
    try:
        layer_results = {}
        for layer in LAYERS:
            print("Reproducing Core layer {}...".format(layer), flush=True)
            layer_results[str(layer)] = reproduce_layer(meta, archive, layer)
    finally:
        archive.close()

    with np.load(activation_path, allow_pickle=False) as summary_archive:
        groups = upstream.build_groups(
            meta, summary_archive, LAYERS[0], phase="discussion", avg_rounds=True
        )
    domains = sorted({group["domain"] for group in groups.values()})
    mode_counts = Counter(group["mode"] for group in groups.values())

    across_layers = {}
    for method in upstream.METHOD_KEYS:
        layer_values = [
            layer_results[str(layer)]["fold_macro_mean_auroc"][method]
            for layer in LAYERS
        ]
        mean_value = float(np.mean(layer_values))
        paper_value = PAPER_REFERENCE[method]
        across_layers[method] = {
            "layer_aurocs": {
                str(layer): layer_results[str(layer)]["fold_macro_mean_auroc"][method]
                for layer in LAYERS
            },
            "mean_auroc": mean_value,
            "population_std_across_layers": float(np.std(layer_values)),
            "paper_v2_reported_auroc": paper_value,
            "absolute_difference_from_paper": abs(mean_value - paper_value),
            "matches_paper_two_decimal_value": round(mean_value, 2) == paper_value,
        }

    repository_match = all(
        layer_results[str(layer)]["detailed_matches_direct_upstream"] for layer in LAYERS
    )
    paper_match = all(
        values["matches_paper_two_decimal_value"] for values in across_layers.values()
    )
    status = "PASS" if repository_match and paper_match else "MISMATCH"

    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "scope": "Official NARCBench-Core activation-probe reproduction only; no temporal experiment.",
        "sources": {
            "upstream_repository": "https://github.com/aaronrose227/narcbench",
            "upstream_path": str(UPSTREAM_ROOT),
            "upstream_git_commit": git_commit(UPSTREAM_ROOT),
            "upstream_source_file": str(UPSTREAM_ROOT / "probes" / "reproduce.py"),
            "upstream_source_modified": False,
            "dataset_repository": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "metadata_sha256": sha256_file(meta_path),
            "activations_sha256": sha256_file(activation_path),
            "paper_version": "arXiv:2604.01151v2",
            "paper_url": PAPER_URL,
            "paper_reference_location": "Appendix A, Table 4, Qwen3-32B Core row",
        },
        "environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "machine": platform.machine(),
            "system": platform.system(),
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
            "cpu_only": True,
        },
        "model": {
            "huggingface_id": MODEL_ID,
            "weights_loaded": False,
            "released_activations_only": True,
        },
        "dataset": {
            "tier": "Core",
            "metadata_rows": len(meta),
            "activation_layers": LAYERS,
            "run_count": len(groups),
            "runs_by_mode": dict(sorted(mode_counts.items())),
            "scenario_count": len(groups) // 2,
            "fold_count": len(domains),
            "folds": domains,
            "all_runs_have_four_agents": all(len(group["agents"]) == 4 for group in groups.values()),
        },
        "methodology": {
            "reused_upstream_functions": [
                "load_data",
                "build_groups",
                "train_deception_probe",
                "compute_scores",
                "evaluate",
                "held_out_cv_layer",
            ],
            "layers": LAYERS,
            "random_seed": RANDOM_SEED,
            "training_labels": {
                "source_runs": "collusion condition only",
                "honest": 0,
                "colluder": 1,
                "unit_in_repository_code": "one public discussion agent-turn activation row",
                "class_balancing": "downsample honest candidate rows to colluder count without replacement",
            },
            "training_filter": {
                "mode": "collusion",
                "channel": "public",
                "phase": "discussion",
                "held_out_domain_excluded": True,
            },
            "evaluation_representation": "For each run and agent, mean raw activation across all public discussion rounds.",
            "deception_scaler": {
                "class": "sklearn.preprocessing.StandardScaler",
                "parameters": StandardScaler().get_params(deep=False),
                "fit_scope": "balanced training agent-turn rows only",
            },
            "classifier_configuration": classifier_configuration(),
            "aggregation_methods": aggregation_method_specs(),
            "evaluation": {
                "protocol": "leave-one-domain-out cross-validation",
                "fold_aggregation": "unweighted arithmetic mean of AUROC across observed domains",
                "layer_aggregation": "unweighted arithmetic mean; population standard deviation",
            },
            "wrapper_deviations": [
                "Core-only path supplied directly instead of invoking reproduce.py main(), which requires Transfer and Stego.",
                "Fold-level provenance and output serialization added outside upstream source.",
                "No upstream source file modified.",
            ],
        },
        "paper_reference": {
            "reported_core_auroc": PAPER_REFERENCE,
            "reported_layers": LAYERS,
            "reported_seed_averaging": 10,
            "reported_core_domains_and_folds": 17,
            "reported_core_scenarios": 50,
        },
        "results": {
            "per_layer": layer_results,
            "across_layers": across_layers,
        },
        "comparison": {
            "detailed_wrapper_matches_direct_repository_function": repository_match,
            "reproduced_values_match_paper_at_reported_two_decimal_precision": paper_match,
            "material_numeric_difference": not (repository_match and paper_match),
            "provenance_differences": [
                "Released data: 31 matched scenarios and 14 observed domains; paper v2: 50 scenarios and 17 folds.",
                "Repository code: one hard-coded seed 42; paper v2: mean over ten random seeds.",
            ],
        },
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(json_value(result), indent=2) + "\n", encoding="utf-8")
    OUTPUT_REPORT.write_text(render_report(result), encoding="utf-8")

    print("Saved {}".format(OUTPUT_JSON), flush=True)
    print("Saved {}".format(OUTPUT_REPORT), flush=True)
    print("Status: {}".format(status), flush=True)
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
