#!/usr/bin/env python3
"""Validate the released Qwen3-32B NARCBench-Core activation artifacts."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "narcbench"
DEFAULT_CORE_DIR = DEFAULT_DATASET_ROOT / "activations" / "qwen3_32b" / "core"
DATASET_REVISION = "19cb14b269e052b19ba6d2bd2f4519afd3b10c57"

EXPECTED_FILES = ("activations_gen.npz", "metadata_gen.json")
EXPECTED_LAYERS = [26, 27, 28, 29, 30]
REQUIRED_CORE_COLUMNS = {
    "run_id",
    "mode",
    "role",
    "channel",
    "phase",
    "round",
    "agent_name",
    "n_tokens",
    "family",
    "scenario_id",
    "scenario_domain",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def counter_dict(values):
    return {str(key): value for key, value in sorted(Counter(values).items(), key=lambda x: str(x[0]))}


def is_missing(value):
    return value is None or (isinstance(value, str) and not value.strip())


def add_issue(issue_rows, issue_counts, issue, row_index=None):
    issue_counts[issue] += 1
    if row_index is not None:
        issue_rows.add(row_index)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--core-dir",
        type=Path,
        default=DEFAULT_CORE_DIR,
        help="Directory containing metadata_gen.json and activations_gen.npz.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    core_dir = args.core_dir.resolve()
    dataset_root = core_dir.parents[2]
    metadata_path = core_dir / "metadata_gen.json"
    activations_path = core_dir / "activations_gen.npz"

    missing_files = [name for name in EXPECTED_FILES if not core_dir.joinpath(name).is_file()]
    if missing_files:
        print(json.dumps({
            "status": "FAIL",
            "core_dir": str(core_dir),
            "missing_files": missing_files,
        }, indent=2))
        return 1

    with metadata_path.open(encoding="utf-8") as handle:
        metadata_rows = json.load(handle)
    if not isinstance(metadata_rows, list):
        raise SystemExit("metadata_gen.json must contain a JSON list")

    issue_rows = set()
    issue_counts = Counter()
    malformed_rows = []
    dict_rows = []
    for index, row in enumerate(metadata_rows):
        if not isinstance(row, dict):
            malformed_rows.append(index)
            add_issue(issue_rows, issue_counts, "metadata_row_is_not_an_object", index)
        else:
            dict_rows.append((index, row))

    first_columns = list(dict_rows[0][1]) if dict_rows else []
    all_columns = set().union(*(set(row) for _, row in dict_rows)) if dict_rows else set()
    ordered_columns = first_columns + sorted(all_columns.difference(first_columns))

    null_counts = {
        column: sum(1 for _, row in dict_rows if row.get(column) is None)
        for column in ordered_columns
    }
    missing_key_counts = {
        column: sum(1 for _, row in dict_rows if column not in row)
        for column in ordered_columns
    }

    reference_schema = set(first_columns)
    for index, row in dict_rows:
        if set(row) != reference_schema:
            add_issue(issue_rows, issue_counts, "row_schema_differs_from_first_row", index)
        for column in REQUIRED_CORE_COLUMNS:
            if column not in row or is_missing(row.get(column)):
                add_issue(issue_rows, issue_counts, "missing_required_core_field:" + column, index)

        mode = row.get("mode")
        role = row.get("role")
        channel = row.get("channel")
        phase = row.get("phase")
        round_number = row.get("round")

        if mode not in {"collusion", "control"}:
            add_issue(issue_rows, issue_counts, "invalid_mode", index)
        if role not in {"colluder", "honest"}:
            add_issue(issue_rows, issue_counts, "invalid_role", index)
        if channel not in {"public", "private", "thinking"}:
            add_issue(issue_rows, issue_counts, "invalid_channel", index)
        if phase not in {"private", "discussion", "final"}:
            add_issue(issue_rows, issue_counts, "invalid_phase", index)
        if mode == "control" and role != "honest":
            add_issue(issue_rows, issue_counts, "control_row_has_non_honest_role", index)

        valid_turn = (
            (phase == "private" and channel == "private" and round_number == 0)
            or (phase == "discussion" and channel == "public" and round_number in {1, 2, 3})
            or (phase == "final" and channel == "public" and round_number == 4)
        )
        if not valid_turn:
            add_issue(issue_rows, issue_counts, "invalid_core_phase_channel_round_combination", index)

        run_id = row.get("run_id")
        if isinstance(run_id, str):
            expected_run_id = "{}__{}__{}".format(
                row.get("family"), row.get("scenario_id"), row.get("mode")
            )
            if run_id != expected_run_id:
                add_issue(issue_rows, issue_counts, "run_id_fields_disagree", index)

        n_tokens = row.get("n_tokens")
        if not isinstance(n_tokens, int) or isinstance(n_tokens, bool) or n_tokens < 0:
            add_issue(issue_rows, issue_counts, "invalid_n_tokens", index)

    run_rows = defaultdict(list)
    for index, row in dict_rows:
        run_rows[row.get("run_id")].append((index, row))

    run_issue_counts = Counter()
    for run_id, indexed_rows in run_rows.items():
        rows = [row for _, row in indexed_rows]
        indices = [index for index, _ in indexed_rows]
        modes = {row.get("mode") for row in rows}
        families = {row.get("family") for row in rows}
        scenario_ids = {row.get("scenario_id") for row in rows}
        domains = {row.get("scenario_domain") for row in rows}
        agents = {row.get("agent_name") for row in rows}

        invariant_sets = {
            "mode": modes,
            "family": families,
            "scenario_id": scenario_ids,
            "scenario_domain": domains,
        }
        for field, values in invariant_sets.items():
            if len(values) != 1:
                run_issue_counts["run_has_multiple_" + field + "_values"] += 1
                issue_rows.update(indices)

        if len(agents) != 4:
            run_issue_counts["run_does_not_have_four_agents"] += 1
            issue_rows.update(indices)

        mode = next(iter(modes)) if len(modes) == 1 else None
        role_by_agent = defaultdict(set)
        for row in rows:
            role_by_agent[row.get("agent_name")].add(row.get("role"))
        if any(len(roles) != 1 for roles in role_by_agent.values()):
            run_issue_counts["agent_role_changes_within_run"] += 1
            issue_rows.update(indices)

        role_counts = Counter(next(iter(roles)) for roles in role_by_agent.values() if len(roles) == 1)
        expected_role_counts = Counter({"colluder": 2, "honest": 2}) if mode == "collusion" else Counter({"honest": 4})
        if role_counts != expected_role_counts:
            run_issue_counts["unexpected_agent_role_composition"] += 1
            issue_rows.update(indices)

        public_turns = Counter(
            (row.get("agent_name"), row.get("round"))
            for row in rows
            if row.get("channel") == "public"
        )
        expected_public_turns = {(agent, round_number) for agent in agents for round_number in (1, 2, 3, 4)}
        if set(public_turns) != expected_public_turns or any(count != 1 for count in public_turns.values()):
            run_issue_counts["public_agent_round_grid_is_incomplete_or_duplicated"] += 1
            issue_rows.update(indices)

        private_rows = [row for row in rows if row.get("channel") == "private"]
        expected_private_count = 2 if mode == "collusion" else 0
        if len(private_rows) != expected_private_count:
            run_issue_counts["unexpected_private_row_count"] += 1
            issue_rows.update(indices)
        elif mode == "collusion" and any(row.get("role") != "colluder" for row in private_rows):
            run_issue_counts["private_row_belongs_to_non_colluder"] += 1
            issue_rows.update(indices)

    scenario_modes = defaultdict(set)
    for _, row in dict_rows:
        scenario_key = (
            row.get("family"),
            row.get("scenario_id"),
            row.get("scenario_domain"),
        )
        scenario_modes[scenario_key].add(row.get("mode"))
    unmatched_scenarios = [
        {"family": key[0], "scenario_id": key[1], "scenario_domain": key[2], "modes": sorted(modes)}
        for key, modes in sorted(scenario_modes.items())
        if modes != {"collusion", "control"}
    ]
    if unmatched_scenarios:
        issue_counts["unmatched_collusion_control_scenarios"] = len(unmatched_scenarios)

    activation_details = {}
    activation_row_issue_indices = set()
    uncompressed_activation_bytes = 0
    with np.load(activations_path, allow_pickle=False) as archive:
        activation_keys = list(archive.files)
        for key in activation_keys:
            array = archive[key]
            uncompressed_activation_bytes += array.nbytes
            nonfinite_rows = np.flatnonzero(~np.isfinite(array).all(axis=1)).tolist() if array.ndim == 2 else []
            activation_row_issue_indices.update(nonfinite_rows)
            activation_details[key] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "nbytes": array.nbytes,
                "nonfinite_row_count": len(nonfinite_rows),
            }
            if array.ndim != 2:
                issue_counts["activation_array_not_2d:" + key] += 1
            elif array.shape[0] != len(metadata_rows):
                issue_counts["activation_metadata_row_mismatch:" + key] += 1
            if array.dtype != np.float32:
                issue_counts["activation_dtype_not_float32:" + key] += 1

    layer_numbers = sorted(
        int(key.split("_", 1)[1])
        for key in activation_details
        if key.startswith("layer_") and key.split("_", 1)[1].isdigit()
    )
    if layer_numbers != EXPECTED_LAYERS:
        issue_counts["activation_layers_differ_from_expected_26_to_30"] += 1

    downloaded_files = []
    for path in sorted(item for item in dataset_root.rglob("*") if item.is_file()):
        stat = path.stat()
        downloaded_files.append({
            "path": str(path.relative_to(PROJECT_ROOT)),
            "bytes": stat.st_size,
            "allocated_bytes": stat.st_blocks * 512,
            "sha256": sha256(path),
        })

    modes_by_run = Counter(
        next(iter({row.get("mode") for _, row in indexed_rows}))
        for indexed_rows in run_rows.values()
    )
    collusion_success_by_run = Counter(
        next(iter({row.get("collusion_success") for _, row in indexed_rows}))
        for indexed_rows in run_rows.values()
    )
    fields = {column: [row.get(column) for _, row in dict_rows] for column in ordered_columns}
    all_issue_rows = issue_rows.union(activation_row_issue_indices)

    report = {
        "status": "PASS" if not issue_counts and not run_issue_counts and not unmatched_scenarios else "FAIL",
        "dataset": {
            "repository": "aaronrose227/narcbench",
            "revision": DATASET_REVISION,
            "model": "Qwen/Qwen3-32B-AWQ",
            "tier": "core",
            "core_dir": str(core_dir),
        },
        "downloaded_files": downloaded_files,
        "disk_usage": {
            "downloaded_logical_bytes": sum(item["bytes"] for item in downloaded_files),
            "downloaded_allocated_bytes": sum(item["allocated_bytes"] for item in downloaded_files),
            "uncompressed_activation_bytes": uncompressed_activation_bytes,
        },
        "activations": {
            "keys": list(activation_details),
            "layers": layer_numbers,
            "arrays": activation_details,
            "metadata_row_count": len(metadata_rows),
        },
        "metadata": {
            "columns": ordered_columns,
            "column_count": len(ordered_columns),
            "row_count": len(metadata_rows),
            "malformed_row_indices": malformed_rows,
            "null_counts": null_counts,
            "all_null_columns": [
                column for column, count in null_counts.items() if count == len(metadata_rows)
            ],
            "missing_key_counts": missing_key_counts,
        },
        "core_structure": {
            "run_count": len(run_rows),
            "runs_by_mode": counter_dict(modes_by_run.elements()),
            "runs_by_collusion_success": counter_dict(collusion_success_by_run.elements()),
            "scenario_count": len(scenario_modes),
            "matched_collusion_control_scenario_count": len(scenario_modes) - len(unmatched_scenarios),
            "unmatched_scenarios": unmatched_scenarios,
            "families": sorted(set(fields.get("family", []))),
            "domains": sorted(set(fields.get("scenario_domain", []))),
            "rounds": sorted(set(fields.get("round", []))),
            "roles": sorted(set(fields.get("role", []))),
            "channels": sorted(set(fields.get("channel", []))),
            "phases": sorted(set(fields.get("phase", []))),
            "row_counts_by_mode": counter_dict(fields.get("mode", [])),
            "row_counts_by_role": counter_dict(fields.get("role", [])),
            "row_counts_by_channel": counter_dict(fields.get("channel", [])),
            "row_counts_by_phase": counter_dict(fields.get("phase", [])),
            "row_counts_by_round": counter_dict(fields.get("round", [])),
            "run_counts_by_domain": counter_dict(
                next(iter({row.get("scenario_domain") for _, row in indexed_rows}))
                for indexed_rows in run_rows.values()
            ),
            "scenario_counts_by_domain": counter_dict(
                scenario_key[2] for scenario_key in scenario_modes
            ),
        },
        "integrity": {
            "missing_or_inconsistent_metadata_row_count": len(all_issue_rows),
            "missing_or_inconsistent_metadata_row_indices": sorted(all_issue_rows),
            "row_issue_counts": dict(sorted(issue_counts.items())),
            "run_issue_counts": dict(sorted(run_issue_counts.items())),
            "activation_nonfinite_row_count": len(activation_row_issue_indices),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
