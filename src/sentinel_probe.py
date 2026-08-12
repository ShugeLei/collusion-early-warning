"""Leakage-safe honest-agent sentinel probes for NARCBench-Core."""

from __future__ import annotations

import hashlib
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.temporal_detectability import (
    PRE_FINAL_ROUNDS,
    TEMPORAL_VIEWS,
    build_temporal_representation,
    source_rounds,
    temporal_label,
)


ANALYSIS_POPULATIONS = ("all_observers", "style_matched")
OBSERVER_SCOPES = ("position_2", "position_3", "both_positions")


def _array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def sentinel_fingerprint(scaler, classifier):
    digest = hashlib.sha256()
    for array in (
        scaler.mean_,
        scaler.scale_,
        scaler.var_,
        classifier.coef_,
        classifier.intercept_,
    ):
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def build_observer_examples(temporal_index, run_metadata, positions, turns):
    """Build all honest-observer temporal examples with paired style metadata."""
    observer_positions = positions.loc[positions["generator_position"] >= 2].copy()
    style_pairs = observer_positions.pivot(
        index=["scenario_id", "agent_name", "generator_position"],
        columns="mode",
        values="prompt_style",
    ).reset_index()
    style_pairs = style_pairs.rename(
        columns={"collusion": "collusion_prompt_style", "control": "control_prompt_style"}
    )
    style_pairs["paired_prompt_style_matched"] = (
        style_pairs["collusion_prompt_style"] == style_pairs["control_prompt_style"]
    )
    observer_positions = observer_positions.merge(
        style_pairs,
        on=["scenario_id", "agent_name", "generator_position"],
        how="left",
        validate="many_to_one",
    )

    turn_lookup = turns.set_index(["run_id", "agent_name", "round"]).to_dict(orient="index")
    exposure_sequences = (
        turns.loc[turns["generator_position"] >= 2]
        .sort_values(["run_id", "agent_name", "round"])
        .groupby(["run_id", "agent_name"])["exposure_count"]
        .apply(lambda values: tuple(int(value) for value in values))
        .to_dict()
    )

    records = []
    for _, position in observer_positions.iterrows():
        run_id = position["run_id"]
        agent_name = position["agent_name"]
        if run_metadata[run_id]["roles"][agent_name] != "honest":
            raise ValueError("Observer example contains a non-honest role")
        round_vectors = temporal_index[run_id][agent_name]
        for temporal_view in TEMPORAL_VIEWS:
            for endpoint_round in PRE_FINAL_ROUNDS:
                rounds_used = source_rounds(temporal_view, endpoint_round)
                turn = turn_lookup[(run_id, agent_name, endpoint_round)]
                records.append({
                    "scenario_id": position["scenario_id"],
                    "run_id": run_id,
                    "domain": run_metadata[run_id]["domain"],
                    "mode": position["mode"],
                    "environment_label": int(position["mode"] == "collusion"),
                    "agent_name": agent_name,
                    "role": "honest",
                    "generator_position": int(position["generator_position"]),
                    "prompt_style": position["prompt_style"],
                    "collusion_prompt_style": position["collusion_prompt_style"],
                    "control_prompt_style": position["control_prompt_style"],
                    "paired_prompt_style_matched": bool(position["paired_prompt_style_matched"]),
                    "temporal_view": temporal_view,
                    "round": endpoint_round,
                    "round_or_prefix": temporal_label(temporal_view, endpoint_round),
                    "source_rounds": "|".join(str(value) for value in rounds_used),
                    "turn_in_round": int(turn["turn_in_round"]),
                    "public_turn_index": int(turn["public_turn_index"]),
                    "exposure_count": int(turn["exposure_count"]),
                    "actual_colluder_message_exposure_count": int(
                        turn["actual_colluder_message_exposure_count"]
                    ),
                    "exposure_messages_are_collusive": bool(
                        turn["exposure_messages_are_collusive"]
                    ),
                    "source_exposure_counts": "|".join(
                        str(value)
                        for value in exposure_sequences[(run_id, agent_name)][:endpoint_round]
                    ),
                    "activation": build_temporal_representation(
                        round_vectors, temporal_view, endpoint_round
                    ),
                })
    frame = pd.DataFrame.from_records(records)
    validate_observer_pairs(frame)
    return frame


def validate_observer_pairs(examples):
    """Require exact collusion/control pairing for every retained observer feature."""
    pair_columns = [
        "scenario_id",
        "agent_name",
        "generator_position",
        "temporal_view",
        "round",
    ]
    counts = examples.groupby(pair_columns, dropna=False)["mode"].agg(
        lambda values: tuple(sorted(values))
    )
    invalid = counts[counts != ("collusion", "control")]
    if not invalid.empty:
        raise ValueError("Observer features contain unmatched condition rows")
    if examples.duplicated(pair_columns + ["mode"]).any():
        raise ValueError("Observer features contain duplicate paired rows")


def filter_observer_examples(examples, analysis_population, observer_scope):
    if analysis_population not in ANALYSIS_POPULATIONS:
        raise ValueError("Unknown analysis population: {}".format(analysis_population))
    if observer_scope not in OBSERVER_SCOPES:
        raise ValueError("Unknown observer scope: {}".format(observer_scope))
    frame = examples
    if analysis_population == "style_matched":
        frame = frame.loc[frame["paired_prompt_style_matched"]]
    if observer_scope == "position_2":
        frame = frame.loc[frame["generator_position"] == 2]
    elif observer_scope == "position_3":
        frame = frame.loc[frame["generator_position"] == 3]
    else:
        frame = frame.loc[frame["generator_position"].isin([2, 3])]
    frame = frame.copy()
    if not frame.empty:
        validate_observer_pairs(frame)
    return frame


def population_audit(examples):
    """Count every included and excluded observer identity pair explicitly."""
    identity_pairs = examples[
        [
            "scenario_id",
            "domain",
            "agent_name",
            "generator_position",
            "paired_prompt_style_matched",
            "collusion_prompt_style",
            "control_prompt_style",
        ]
    ].drop_duplicates()
    records = []
    for population in ANALYSIS_POPULATIONS:
        for scope in OBSERVER_SCOPES:
            frame = identity_pairs
            if scope == "position_2":
                frame = frame.loc[frame["generator_position"] == 2]
            elif scope == "position_3":
                frame = frame.loc[frame["generator_position"] == 3]
            retained = (
                frame
                if population == "all_observers"
                else frame.loc[frame["paired_prompt_style_matched"]]
            )
            records.append({
                "analysis_population": population,
                "observer_scope": scope,
                "available_observer_identity_pairs": len(frame),
                "retained_observer_identity_pairs": len(retained),
                "excluded_style_mismatched_pairs": len(frame) - len(retained),
                "retained_scenario_count": retained["scenario_id"].nunique(),
                "retained_domains": sorted(retained["domain"].unique()),
                "missing_domains": sorted(set(identity_pairs["domain"]) - set(retained["domain"])),
            })
    mismatch = identity_pairs.loc[~identity_pairs["paired_prompt_style_matched"]].copy()
    exclusions_by_domain_position = (
        mismatch.groupby(["domain", "generator_position"])
        .size()
        .rename("excluded_pair_count")
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "population_scope_counts": records,
        "style_mismatch_exclusions_by_domain_and_position": exclusions_by_domain_position,
    }


def fit_sentinel_fold(train_frame, test_frame, held_out_domain):
    """Fit a scaler/classifier on training domains and score one held-out domain."""
    if train_frame.empty:
        raise ValueError("Sentinel training frame is empty")
    if test_frame.empty:
        raise ValueError("Sentinel test frame is empty")
    if held_out_domain in set(train_frame["domain"]):
        raise ValueError("Held-out domain entered sentinel training data")
    if set(test_frame["domain"]) != {held_out_domain}:
        raise ValueError("Sentinel test data are not exclusively held out")
    if set(train_frame["environment_label"]) != {0, 1}:
        raise ValueError("Sentinel training data do not contain both classes")

    train_matrix = np.stack(train_frame["activation"].to_numpy())
    train_labels = train_frame["environment_label"].to_numpy(dtype=int)
    test_matrix = np.stack(test_frame["activation"].to_numpy())
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_matrix)
    classifier = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(scaled_train, train_labels)
    convergence_warnings = [
        str(item.message) for item in captured if issubclass(item.category, ConvergenceWarning)
    ]

    scaled_test = scaler.transform(test_matrix)
    decision_scores = classifier.decision_function(scaled_test)
    probabilities = classifier.predict_proba(scaled_test)[:, list(classifier.classes_).index(1)]
    coefficient = classifier.coef_[0]
    coefficient_norm = float(np.linalg.norm(coefficient))
    if coefficient_norm == 0.0:
        raise ValueError("Sentinel classifier has a zero direction")
    unit_direction = coefficient / coefficient_norm
    fingerprint = sentinel_fingerprint(scaler, classifier)

    predictions = []
    scored_test = test_frame.copy()
    scored_test["sentinel_score"] = decision_scores
    scored_test["sentinel_probability"] = probabilities
    output_columns = [column for column in test_frame.columns if column != "activation"]
    for _, row in scored_test.iterrows():
        predictions.append({
            **{column: row[column] for column in output_columns},
            "fold": held_out_domain,
            "sentinel_score": float(row["sentinel_score"]),
            "sentinel_probability": float(row["sentinel_probability"]),
            "model_fingerprint": fingerprint,
        })

    diagnostic_records = []
    pair_columns = ["scenario_id", "agent_name", "generator_position"]
    for keys, pair in scored_test.groupby(pair_columns, sort=True):
        if set(pair["mode"]) != {"collusion", "control"} or len(pair) != 2:
            raise ValueError("Sentinel test observations are not exactly paired")
        collusion = pair.loc[pair["mode"] == "collusion"].iloc[0]
        control = pair.loc[pair["mode"] == "control"].iloc[0]
        collusion_vector = np.asarray(collusion["activation"], dtype=float)
        control_vector = np.asarray(control["activation"], dtype=float)
        delta = collusion_vector - control_vector
        denominator = np.linalg.norm(collusion_vector) * np.linalg.norm(control_vector)
        cosine = float(np.dot(collusion_vector, control_vector) / denominator) if denominator else None
        scaled_delta = delta / scaler.scale_
        projection = float(unit_direction @ scaled_delta)
        decision_shift = float(collusion["sentinel_score"] - control["sentinel_score"])
        expected_decision_shift = float(coefficient @ scaled_delta)
        # sklearn preserves float32 in parts of transform/decision_function;
        # recomputing the algebra in float64 can differ at roughly 1e-7.
        if not np.isclose(decision_shift, expected_decision_shift, rtol=1e-6, atol=1e-6):
            raise ValueError("Paired decision shift is inconsistent with the sentinel direction")
        diagnostic_records.append({
            "scenario_id": keys[0],
            "agent_name": keys[1],
            "generator_position": int(keys[2]),
            "domain": held_out_domain,
            "fold": held_out_domain,
            "delta_activation_l2": float(np.linalg.norm(delta)),
            "paired_activation_cosine_similarity": cosine,
            "delta_projection_on_unit_sentinel_direction": projection,
            "sentinel_decision_score_shift": decision_shift,
            "model_coefficient_l2_norm": coefficient_norm,
            "model_fingerprint": fingerprint,
        })

    label_counts = Counter(train_labels)
    audit = {
        "held_out_domain": held_out_domain,
        "training_rows": len(train_frame),
        "test_rows": len(test_frame),
        "training_control_rows": label_counts.get(0, 0),
        "training_collusion_rows": label_counts.get(1, 0),
        "training_scenario_count": train_frame["scenario_id"].nunique(),
        "test_scenario_count": test_frame["scenario_id"].nunique(),
        "training_domains": sorted(train_frame["domain"].unique()),
        "held_out_domain_present_in_training": False,
        "feature_count": train_matrix.shape[1],
        "scaler": {
            "class": "StandardScaler",
            "n_samples_seen": int(scaler.n_samples_seen_),
            "mean_sha256": _array_sha256(scaler.mean_),
            "scale_sha256": _array_sha256(scaler.scale_),
            "variance_sha256": _array_sha256(scaler.var_),
        },
        "classifier": {
            "class": "LogisticRegression",
            "C": float(classifier.C),
            "max_iter": int(classifier.max_iter),
            "random_state": int(classifier.random_state),
            "solver": classifier.solver,
            "penalty": classifier.penalty,
            "fit_intercept": bool(classifier.fit_intercept),
            "class_weight": classifier.class_weight,
            "classes": [int(value) for value in classifier.classes_],
            "n_iter": int(classifier.n_iter_[0]),
            "coefficient_l2_norm": coefficient_norm,
            "coefficient_sha256": _array_sha256(classifier.coef_),
            "intercept_sha256": _array_sha256(classifier.intercept_),
        },
        "model_fingerprint": fingerprint,
        "convergence_warnings": convergence_warnings,
    }
    return predictions, diagnostic_records, audit


def _score_with_fitted_sentinel(test_frame, held_out_domain, scaler, classifier):
    """Score one held-out temporal view using an already fitted sentinel."""
    if test_frame.empty:
        raise ValueError("Fixed sentinel test frame is empty")
    if set(test_frame["domain"]) != {held_out_domain}:
        raise ValueError("Fixed sentinel test data are not exclusively held out")
    test_matrix = np.stack(test_frame["activation"].to_numpy())
    scaled_test = scaler.transform(test_matrix)
    decision_scores = classifier.decision_function(scaled_test)
    probabilities = classifier.predict_proba(scaled_test)[:, list(classifier.classes_).index(1)]
    coefficient = classifier.coef_[0]
    coefficient_norm = float(np.linalg.norm(coefficient))
    if coefficient_norm == 0.0:
        raise ValueError("Fixed sentinel classifier has a zero direction")
    unit_direction = coefficient / coefficient_norm
    fingerprint = sentinel_fingerprint(scaler, classifier)

    scored_test = test_frame.copy()
    scored_test["sentinel_score"] = decision_scores
    scored_test["sentinel_probability"] = probabilities
    output_columns = [column for column in test_frame.columns if column != "activation"]
    predictions = [
        {
            **{column: row[column] for column in output_columns},
            "fold": held_out_domain,
            "sentinel_score": float(row["sentinel_score"]),
            "sentinel_probability": float(row["sentinel_probability"]),
            "model_fingerprint": fingerprint,
        }
        for _, row in scored_test.iterrows()
    ]

    diagnostics = []
    pair_columns = ["scenario_id", "agent_name", "generator_position"]
    for keys, pair in scored_test.groupby(pair_columns, sort=True):
        if set(pair["mode"]) != {"collusion", "control"} or len(pair) != 2:
            raise ValueError("Fixed sentinel observations are not exactly paired")
        collusion = pair.loc[pair["mode"] == "collusion"].iloc[0]
        control = pair.loc[pair["mode"] == "control"].iloc[0]
        collusion_vector = np.asarray(collusion["activation"], dtype=float)
        control_vector = np.asarray(control["activation"], dtype=float)
        delta = collusion_vector - control_vector
        denominator = np.linalg.norm(collusion_vector) * np.linalg.norm(control_vector)
        cosine = float(np.dot(collusion_vector, control_vector) / denominator) if denominator else None
        scaled_delta = delta / scaler.scale_
        projection = float(unit_direction @ scaled_delta)
        decision_shift = float(collusion["sentinel_score"] - control["sentinel_score"])
        expected_decision_shift = float(coefficient @ scaled_delta)
        if not np.isclose(decision_shift, expected_decision_shift, rtol=1e-6, atol=1e-6):
            raise ValueError("Fixed sentinel score shift is inconsistent with its direction")
        diagnostics.append({
            "scenario_id": keys[0],
            "agent_name": keys[1],
            "generator_position": int(keys[2]),
            "domain": held_out_domain,
            "fold": held_out_domain,
            "delta_activation_l2": float(np.linalg.norm(delta)),
            "paired_activation_cosine_similarity": cosine,
            "delta_projection_on_unit_sentinel_direction": projection,
            "sentinel_decision_score_shift": decision_shift,
            "model_coefficient_l2_norm": coefficient_norm,
            "model_fingerprint": fingerprint,
        })
    return predictions, diagnostics


def fit_full_discussion_sentinel(train_frame, test_frame, held_out_domain):
    """Fit one sentinel on non-held-out full-discussion mean activations."""
    if train_frame.empty or test_frame.empty:
        raise ValueError("Fixed sentinel requires non-empty train and test full-discussion rows")
    if held_out_domain in set(train_frame["domain"]):
        raise ValueError("Held-out domain entered fixed sentinel training data")
    if set(test_frame["domain"]) != {held_out_domain}:
        raise ValueError("Fixed sentinel full-discussion test rows are not held out")
    if set(train_frame["environment_label"]) != {0, 1}:
        raise ValueError("Fixed sentinel training data do not contain both classes")
    if set(train_frame["temporal_view"]) != {"prefix"} or set(train_frame["round"]) != {3}:
        raise ValueError("Fixed sentinel must train only on mean(R1,R2,R3)")

    train_matrix = np.stack(train_frame["activation"].to_numpy())
    train_labels = train_frame["environment_label"].to_numpy(dtype=int)
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_matrix)
    classifier = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(scaled_train, train_labels)
    convergence_warnings = [
        str(item.message) for item in captured if issubclass(item.category, ConvergenceWarning)
    ]
    coefficient = classifier.coef_[0]
    coefficient_norm = float(np.linalg.norm(coefficient))
    fingerprint = sentinel_fingerprint(scaler, classifier)
    counts = Counter(train_labels)
    audit = {
        "held_out_domain": held_out_domain,
        "training_representation": "mean(R1,R2,R3)",
        "training_temporal_view": "prefix",
        "training_endpoint_round": 3,
        "training_source_rounds": "1|2|3",
        "training_rows": len(train_frame),
        "test_full_discussion_rows": len(test_frame),
        "training_control_rows": counts.get(0, 0),
        "training_collusion_rows": counts.get(1, 0),
        "training_scenario_count": train_frame["scenario_id"].nunique(),
        "test_scenario_count": test_frame["scenario_id"].nunique(),
        "training_domains": sorted(train_frame["domain"].unique()),
        "held_out_domain_present_in_training": False,
        "feature_count": train_matrix.shape[1],
        "scaler": {
            "class": "StandardScaler",
            "n_samples_seen": int(scaler.n_samples_seen_),
            "mean_sha256": _array_sha256(scaler.mean_),
            "scale_sha256": _array_sha256(scaler.scale_),
            "variance_sha256": _array_sha256(scaler.var_),
        },
        "classifier": {
            "class": "LogisticRegression",
            "C": float(classifier.C),
            "max_iter": int(classifier.max_iter),
            "random_state": int(classifier.random_state),
            "solver": classifier.solver,
            "penalty": classifier.penalty,
            "fit_intercept": bool(classifier.fit_intercept),
            "class_weight": classifier.class_weight,
            "classes": [int(value) for value in classifier.classes_],
            "n_iter": int(classifier.n_iter_[0]),
            "coefficient_l2_norm": coefficient_norm,
            "coefficient_sha256": _array_sha256(classifier.coef_),
            "intercept_sha256": _array_sha256(classifier.intercept_),
        },
        "model_fingerprint": fingerprint,
        "convergence_warnings": convergence_warnings,
    }
    return scaler, classifier, audit


def run_lodo_fixed_sentinel(examples, expected_domains):
    """Fit one full-discussion probe per fold/scope and reuse it at every time."""
    prediction_records = []
    diagnostic_records = []
    training_audits = []
    for population in ANALYSIS_POPULATIONS:
        for scope in OBSERVER_SCOPES:
            scoped = filter_observer_examples(examples, population, scope)
            full_discussion = scoped.loc[
                (scoped["temporal_view"] == "prefix") & (scoped["round"] == 3)
            ].copy()
            for held_out_domain in expected_domains:
                train = full_discussion.loc[full_discussion["domain"] != held_out_domain]
                test_full = full_discussion.loc[full_discussion["domain"] == held_out_domain]
                common = {
                    "analysis_population": population,
                    "observer_scope": scope,
                    "held_out_domain": held_out_domain,
                    "fixed_across_temporal_views": True,
                }
                if test_full.empty:
                    training_audits.append({
                        **common,
                        "fit_status": "not_fit_no_held_out_observations",
                        "training_rows_available": len(train),
                        "test_rows": 0,
                    })
                    continue
                scaler, classifier, audit = fit_full_discussion_sentinel(
                    train, test_full, held_out_domain
                )
                training_audits.append({**common, "fit_status": "fit", **audit})
                for temporal_view in TEMPORAL_VIEWS:
                    for endpoint_round in PRE_FINAL_ROUNDS:
                        test = scoped.loc[
                            (scoped["domain"] == held_out_domain)
                            & (scoped["temporal_view"] == temporal_view)
                            & (scoped["round"] == endpoint_round)
                        ]
                        predictions, diagnostics = _score_with_fitted_sentinel(
                            test, held_out_domain, scaler, classifier
                        )
                        label = temporal_label(temporal_view, endpoint_round)
                        provenance = {
                            "analysis_population": population,
                            "observer_scope": scope,
                            "fixed_across_temporal_views": True,
                            "probe_training_representation": "mean(R1,R2,R3)",
                        }
                        for record in predictions:
                            record.update(provenance)
                        for record in diagnostics:
                            record.update({
                                **provenance,
                                "temporal_view": temporal_view,
                                "round": endpoint_round,
                                "round_or_prefix": label,
                                "source_rounds": "|".join(
                                    str(value)
                                    for value in source_rounds(temporal_view, endpoint_round)
                                ),
                            })
                        prediction_records.extend(predictions)
                        diagnostic_records.extend(diagnostics)
    return (
        pd.DataFrame.from_records(prediction_records),
        pd.DataFrame.from_records(diagnostic_records),
        training_audits,
    )


def run_lodo_sentinel(examples, expected_domains):
    """Train every requested population/scope/time LODO sentinel model."""
    prediction_records = []
    diagnostic_records = []
    training_audits = []
    for population in ANALYSIS_POPULATIONS:
        for scope in OBSERVER_SCOPES:
            scoped = filter_observer_examples(examples, population, scope)
            for temporal_view in TEMPORAL_VIEWS:
                for endpoint_round in PRE_FINAL_ROUNDS:
                    config = scoped.loc[
                        (scoped["temporal_view"] == temporal_view)
                        & (scoped["round"] == endpoint_round)
                    ].copy()
                    label = temporal_label(temporal_view, endpoint_round)
                    for held_out_domain in expected_domains:
                        train = config.loc[config["domain"] != held_out_domain]
                        test = config.loc[config["domain"] == held_out_domain]
                        common_audit = {
                            "analysis_population": population,
                            "observer_scope": scope,
                            "temporal_view": temporal_view,
                            "round": endpoint_round,
                            "round_or_prefix": label,
                            "source_rounds": "|".join(
                                str(value) for value in source_rounds(temporal_view, endpoint_round)
                            ),
                            "held_out_domain": held_out_domain,
                        }
                        if test.empty:
                            training_audits.append({
                                **common_audit,
                                "fit_status": "not_fit_no_held_out_observations",
                                "training_rows_available": len(train),
                                "test_rows": 0,
                            })
                            continue
                        predictions, diagnostics, audit = fit_sentinel_fold(
                            train, test, held_out_domain
                        )
                        for record in predictions:
                            record.update({
                                "analysis_population": population,
                                "observer_scope": scope,
                            })
                        for record in diagnostics:
                            record.update({
                                "analysis_population": population,
                                "observer_scope": scope,
                                "temporal_view": temporal_view,
                                "round": endpoint_round,
                                "round_or_prefix": label,
                                "source_rounds": common_audit["source_rounds"],
                            })
                        prediction_records.extend(predictions)
                        diagnostic_records.extend(diagnostics)
                        training_audits.append({
                            **common_audit,
                            "fit_status": "fit",
                            **audit,
                        })
    return (
        pd.DataFrame.from_records(prediction_records),
        pd.DataFrame.from_records(diagnostic_records),
        training_audits,
    )


def attach_paired_diagnostics(predictions, diagnostics):
    """Add paired score shifts and representation diagnostics to saved rows."""
    pair_columns = [
        "analysis_population",
        "observer_scope",
        "temporal_view",
        "round",
        "scenario_id",
        "agent_name",
        "generator_position",
        "fold",
        "model_fingerprint",
    ]
    score_pairs = predictions.pivot(
        index=pair_columns, columns="mode", values="sentinel_score"
    ).reset_index()
    score_pairs = score_pairs.rename(
        columns={"collusion": "paired_collusion_score", "control": "paired_control_score"}
    )
    if score_pairs[["paired_collusion_score", "paired_control_score"]].isna().any().any():
        raise ValueError("Sentinel output contains an unmatched score")
    score_pairs["paired_score_difference"] = (
        score_pairs["paired_collusion_score"] - score_pairs["paired_control_score"]
    )
    diagnostic_columns = pair_columns + [
        "delta_activation_l2",
        "paired_activation_cosine_similarity",
        "delta_projection_on_unit_sentinel_direction",
        "sentinel_decision_score_shift",
        "model_coefficient_l2_norm",
    ]
    pairs = score_pairs.merge(
        diagnostics[diagnostic_columns],
        on=pair_columns,
        how="left",
        validate="one_to_one",
    )
    output = predictions.merge(
        pairs,
        on=pair_columns,
        how="left",
        validate="many_to_one",
    )
    if output[
        ["paired_score_difference", "delta_activation_l2"]
    ].isna().any().any():
        raise ValueError("Sentinel paired diagnostics were not attached completely")
    return output


def _scenario_bootstrap(config, scenario_differences, seed, replicates):
    scenario_ids = sorted(config["scenario_id"].unique())
    blocks = {scenario: config.loc[config["scenario_id"] == scenario] for scenario in scenario_ids}
    difference_map = scenario_differences.set_index("scenario_id")["score_difference"].to_dict()
    rng = np.random.RandomState(seed)
    auc_values = np.empty(replicates, dtype=float)
    difference_values = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        selected = rng.randint(0, len(scenario_ids), size=len(scenario_ids))
        sampled_ids = [scenario_ids[index] for index in selected]
        sample = pd.concat([blocks[scenario] for scenario in sampled_ids], ignore_index=True)
        auc_values[replicate] = roc_auc_score(
            sample["environment_label"], sample["sentinel_score"]
        )
        difference_values[replicate] = np.mean(
            [difference_map[scenario] for scenario in sampled_ids]
        )
    return {
        "pooled_oof_auroc_ci_95": [
            float(np.quantile(auc_values, 0.025)),
            float(np.quantile(auc_values, 0.975)),
        ],
        "paired_mean_score_difference_ci_95": [
            float(np.quantile(difference_values, 0.025)),
            float(np.quantile(difference_values, 0.975)),
        ],
    }


def calculate_sentinel_metrics(predictions, expected_domains, bootstrap_replicates=1000, seed=42):
    """Calculate explicit-fold, pooled OOF, paired, and scenario-bootstrap metrics."""
    fold_records = []
    summary_records = []
    metric_issues = []
    config_columns = [
        "analysis_population",
        "observer_scope",
        "temporal_view",
        "round",
        "round_or_prefix",
    ]
    for keys, config in predictions.groupby(config_columns, sort=True):
        population, scope, temporal_view, endpoint_round, label = keys
        valid_aurocs = []
        for domain in expected_domains:
            fold = config.loc[config["fold"] == domain]
            counts = Counter(fold["environment_label"])
            issue = None
            if fold.empty:
                issue = "no retained observer pairs in held-out domain"
                auroc = None
            elif set(fold["environment_label"]) != {0, 1}:
                issue = "single-class fold"
                auroc = None
            else:
                auroc = float(roc_auc_score(fold["environment_label"], fold["sentinel_score"]))
                valid_aurocs.append(auroc)
            record = {
                "analysis_population": population,
                "observer_scope": scope,
                "temporal_view": temporal_view,
                "round": int(endpoint_round),
                "round_or_prefix": label,
                "fold": domain,
                "observation_count": len(fold),
                "control_count": counts.get(0, 0),
                "collusion_count": counts.get(1, 0),
                "scenario_count": fold["scenario_id"].nunique(),
                "auroc": auroc,
                "metric_issue": issue,
            }
            fold_records.append(record)
            if issue:
                metric_issues.append(record)

        if set(config["environment_label"]) != {0, 1}:
            raise ValueError("A sentinel configuration lacks a pooled class")
        pooled_auroc = float(
            roc_auc_score(config["environment_label"], config["sentinel_score"])
        )
        observer_pairs = config[
            [
                "scenario_id",
                "agent_name",
                "generator_position",
                "paired_score_difference",
            ]
        ].drop_duplicates()
        scenario_differences = (
            observer_pairs.groupby("scenario_id")["paired_score_difference"]
            .mean()
            .rename("score_difference")
            .reset_index()
        )
        bootstrap = _scenario_bootstrap(
            config, scenario_differences, seed, bootstrap_replicates
        )
        differences = scenario_differences["score_difference"].to_numpy(dtype=float)
        summary_records.append({
            "analysis_population": population,
            "observer_scope": scope,
            "temporal_view": temporal_view,
            "round": int(endpoint_round),
            "round_or_prefix": label,
            "observation_unit": "honest agent temporal representation",
            "pooled_observation_count": len(config),
            "pooled_control_count": int((config["environment_label"] == 0).sum()),
            "pooled_collusion_count": int((config["environment_label"] == 1).sum()),
            "observer_identity_pair_count": len(observer_pairs),
            "scenario_pair_count": len(scenario_differences),
            "pooled_oof_auroc": pooled_auroc,
            "pooled_oof_auroc_bootstrap_ci_95": bootstrap["pooled_oof_auroc_ci_95"],
            "fold_count_expected": len(expected_domains),
            "fold_count_valid": len(valid_aurocs),
            "fold_count_undefined": len(expected_domains) - len(valid_aurocs),
            "fold_macro_mean_auroc": float(np.mean(valid_aurocs)) if valid_aurocs else None,
            "fold_population_sd_auroc": float(np.std(valid_aurocs, ddof=0)) if valid_aurocs else None,
            "paired_mean_score_difference": float(np.mean(differences)),
            "paired_sample_sd_score_difference": float(np.std(differences, ddof=1)),
            "paired_median_score_difference": float(np.median(differences)),
            "paired_mean_score_difference_bootstrap_ci_95": bootstrap[
                "paired_mean_score_difference_ci_95"
            ],
            "bootstrap_unit": "matched scenario pair",
            "bootstrap_replicates": bootstrap_replicates,
        })
    return (
        pd.DataFrame.from_records(fold_records),
        pd.DataFrame.from_records(summary_records),
        metric_issues,
    )


def summarize_representation_shifts(diagnostics, bootstrap_replicates=1000, seed=42):
    """Summarize round-only OOF paired shifts using position-specific directions."""
    frame = diagnostics.loc[
        (diagnostics["temporal_view"] == "round_only")
        & (diagnostics["observer_scope"].isin(["position_2", "position_3"]))
    ].copy()
    metrics = [
        "delta_activation_l2",
        "paired_activation_cosine_similarity",
        "delta_projection_on_unit_sentinel_direction",
        "sentinel_decision_score_shift",
    ]
    records = []
    for keys, group in frame.groupby(
        ["analysis_population", "observer_scope", "round", "round_or_prefix"], sort=True
    ):
        population, scope, endpoint_round, label = keys
        record = {
            "analysis_population": population,
            "observer_scope": scope,
            "round": int(endpoint_round),
            "round_or_prefix": label,
            "observer_identity_pair_count": len(group),
            "scenario_pair_count": group["scenario_id"].nunique(),
            "projection_direction": "held-out-fold position-specific unit logistic coefficient",
            "delta_scaling_for_projection": "delta_activation / training-fold StandardScaler.scale_",
        }
        rng = np.random.RandomState(seed)
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            record[metric] = {
                "mean": float(np.mean(values)),
                "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "median": float(np.median(values)),
                "q1": float(np.quantile(values, 0.25)),
                "q3": float(np.quantile(values, 0.75)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
            bootstrap = np.empty(bootstrap_replicates, dtype=float)
            for replicate in range(bootstrap_replicates):
                selected = rng.randint(0, len(values), size=len(values))
                bootstrap[replicate] = np.mean(values[selected])
            record[metric]["mean_bootstrap_ci_95"] = [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ]
        records.append(record)
    return records
