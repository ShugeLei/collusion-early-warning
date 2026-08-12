# Phase 2: Temporal Detectability (Layer 28)

- Status: **PASS**
- Probe: one fixed upstream scaler/direction per leave-domain-out fold
- Folds: 14
- Metric issues: 0

| View | Time | Method | Fold mean AUROC | Fold SD | Pooled OOF AUROC |
|---|---|---|---:|---:|---:|
| prefix | R1 | deception_split | 1.000000 | 0.000000 | 1.000000 |
| prefix | R1 | peak_suspicion | 1.000000 | 0.000000 | 1.000000 |
| prefix | R1-R2 | deception_split | 1.000000 | 0.000000 | 1.000000 |
| prefix | R1-R2 | peak_suspicion | 1.000000 | 0.000000 | 1.000000 |
| prefix | R1-R2-R3 | deception_split | 1.000000 | 0.000000 | 1.000000 |
| prefix | R1-R2-R3 | peak_suspicion | 1.000000 | 0.000000 | 1.000000 |
| round_only | R1 | deception_split | 1.000000 | 0.000000 | 1.000000 |
| round_only | R1 | peak_suspicion | 1.000000 | 0.000000 | 1.000000 |
| round_only | R2 | deception_split | 1.000000 | 0.000000 | 1.000000 |
| round_only | R2 | peak_suspicion | 1.000000 | 0.000000 | 1.000000 |
| round_only | R3 | deception_split | 1.000000 | 0.000000 | 1.000000 |
| round_only | R3 | peak_suspicion | 1.000000 | 0.000000 | 1.000000 |

Predictions were saved before metrics and figures were computed. No final-round activation, threshold calibration, TFIR, PIC, or E2 intervention logic was used.
