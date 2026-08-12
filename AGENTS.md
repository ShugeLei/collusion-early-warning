# Project Instructions

Read `EXPERIMENT_PLAN.md` completely before making implementation decisions.

## Research goal

This project studies:

"When does collective internal evidence of multi-agent collusion
become actionable under a realistic false-intervention budget?"

The objective is NOT to build a new generic collusion classifier.

## Compute constraints

Machine:
- Apple M1
- 16 GB unified memory
- CPU-only for this project

Do NOT:
- run Qwen3 locally
- download Qwen model weights
- generate new agent trajectories
- use MLX
- fine-tune LLMs
- require CUDA

Use only the pre-extracted NARCBench activations.

## Experimental constraints

Preserve the NARCBench leave-one-domain-out protocol.

Never allow held-out test-domain data to influence:
- scaler fitting
- probe training
- threshold selection
- layer selection
- aggregation-method selection

Primary development layer:
- layer 28

Robustness layers:
- 26, 27, 28, 29, 30

Primary monitors:
1. Independent-local: Peak Suspicion
2. Collective: Deception Split

Primary operational metrics:
- TFIR: Trajectory-Level False Intervention Rate
- PIC: Pre-Final Intervention Coverage

Secondary metrics:
- AUROC by prefix
- first detection round
- intervention lead
- collective intervention gain

## Engineering rules

Work incrementally.

Before implementing a new experiment:
1. inspect the upstream NARCBench implementation
2. explain what will be reused
3. identify any deviation from the original methodology

Save raw predictions before plotting.

All figures must be reproducible from saved CSV/JSON outputs.

Use seed 42 where applicable.

Write tests for:
- train/test leakage
- prefix construction
- threshold calibration
- metric calculation

Do not silently exclude data.

If something in EXPERIMENT_PLAN.md conflicts with the actual
NARCBench dataset or implementation, stop and report the discrepancy
before changing the methodology.