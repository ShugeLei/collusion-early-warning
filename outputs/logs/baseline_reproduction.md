# NARCBench Core Baseline Reproduction

- Status: **PASS**
- Model: `Qwen/Qwen3-32B-AWQ`
- Dataset: `aaronrose227/narcbench` at `19cb14b269e052b19ba6d2bd2f4519afd3b10c57`
- Upstream commit: `85fea3be2a9d8b9f16e33a4bd65bd947c31fa07d`
- Layers: `26, 27, 28, 29, 30`
- Observed leave-domain-out folds: `14`
- Random seed implemented upstream: `42`

## AUROC

| Method | Reproduced mean | Layer std | Paper v2 | Difference |
|---|---:|---:|---:|---:|
| Sorted Concatenation | 1.000000 | 0.000000 | 1.00 | 0.000000 |
| Activation Alignment | 1.000000 | 0.000000 | 1.00 | 0.000000 |
| Peak Suspicion | 1.000000 | 0.000000 | 1.00 | 0.000000 |
| Deception Split | 1.000000 | 0.000000 | 1.00 | 0.000000 |
| Asymmetry Probe | 1.000000 | 0.000000 | 1.00 | 0.000000 |

Every detailed fold/layer aggregate matched the unchanged upstream `held_out_cv_layer()` result to numerical tolerance.

## Methodology

The wrapper imports upstream loading, grouping, probe training, scoring, and evaluation functions. The deception probe is fit on public discussion rows from collusion-condition training domains only, with `honest=0` and `colluder=1`; classes are balanced using upstream seed 42. `StandardScaler` precedes `LogisticRegression(C=1.0, max_iter=1000, random_state=42)`, and the normalized coefficient vector is used as the deception direction. Evaluation vectors average each agent over all public discussion rounds.

## Comparison and provenance caveats

The reproduced values match paper v2 Table 4 at its two-decimal reporting precision. The current released Qwen Core subset has 31 matched scenarios and 14 observed domains, whereas paper v2 describes 50 scenarios, 17 folds, and averaging over ten seeds. The checked-out repository implementation uses the released rows dynamically and hard-codes one seed (42). Because all reproduced Core values are exactly 1.0, these provenance differences do not create a numerical discrepancy here.

No temporal-prefix or intervention experiment was run.
