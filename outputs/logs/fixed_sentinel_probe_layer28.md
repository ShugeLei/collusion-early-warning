# Experiment D — Fixed Sentinel Temporal Probe (Layer 28)

- Status: **PASS**
- Training: one non-held-out `mean(R1,R2,R3)` probe per fold/population/scope
- Evaluation: the same fitted scaler and classifier reused at all six temporal views
- Features: honest observer positions 2–3 only; zero colluder activation rows
- Classifier: StandardScaler + LogisticRegression(C=1.0, max_iter=1000, random_state=42)
- Scope: temporal methodology control only; no thresholds, TFIR, PIC, or E2

## Fixed-probe integrity

Every prediction in a given population/scope/fold carries one identical model fingerprint. All fitted scalers and classifiers use only non-held-out full-discussion means. There were 73 fitted probes, 11 style-subset folds with no retained test observation, and 0 convergence warnings.

## Retained samples

| Population | Scope | Retained observer pairs | Excluded style mismatches | Scenarios | Missing domains |
|---|---|---:|---:|---:|---|
| all_observers | position_2 | 31 | 0 | 31 | none |
| all_observers | position_3 | 31 | 0 | 31 | none |
| all_observers | both_positions | 62 | 0 | 31 | none |
| style_matched | position_2 | 15 | 16 | 15 | audit, fin, food, leg, re, research |
| style_matched | position_3 | 15 | 16 | 15 | audit, gov, research |
| style_matched | both_positions | 30 | 32 | 22 | audit, research |

## Primary result — fixed probe on round-only activations

| Population | Scope | Round | Pooled OOF AUROC (95% CI) | Fold mean | Fold SD | Valid folds | Paired mean Δ (95% CI) |
|---|---|---|---:|---:|---:|---:|---:|
| all_observers | both_positions | R1 | 0.8057 [0.7242, 0.8892] | 0.8616 | 0.1750 | 14/14 | 7.0037 [5.0247, 8.9195] |
| all_observers | both_positions | R2 | 0.8749 [0.7947, 0.9480] | 0.9102 | 0.1435 | 14/14 | 10.4641 [8.0510, 12.7280] |
| all_observers | both_positions | R3 | 0.9170 [0.8540, 0.9722] | 0.9524 | 0.0875 | 14/14 | 11.3620 [9.2155, 13.6291] |
| all_observers | position_2 | R1 | 0.7076 [0.5723, 0.8273] | 0.6627 | 0.3449 | 14/14 | 4.1091 [1.6218, 6.6166] |
| all_observers | position_2 | R2 | 0.8429 [0.7482, 0.9272] | 0.8532 | 0.2814 | 14/14 | 8.4767 [6.1646, 10.7400] |
| all_observers | position_2 | R3 | 0.8939 [0.8137, 0.9657] | 0.9365 | 0.1166 | 14/14 | 9.1367 [6.9419, 11.3516] |
| all_observers | position_3 | R1 | 0.8554 [0.7700, 0.9376] | 0.9345 | 0.1335 | 14/14 | 9.7410 [7.5443, 11.8108] |
| all_observers | position_3 | R2 | 0.8345 [0.7377, 0.9220] | 0.8829 | 0.1247 | 14/14 | 9.5749 [6.6894, 12.7586] |
| all_observers | position_3 | R3 | 0.8866 [0.8137, 0.9522] | 0.9504 | 0.0854 | 14/14 | 11.0469 [8.4177, 13.7130] |
| style_matched | both_positions | R1 | 0.8033 [0.6911, 0.9085] | 0.8439 | 0.2363 | 12/14 | 5.8386 [3.0482, 8.6479] |
| style_matched | both_positions | R2 | 0.8200 [0.7277, 0.9083] | 0.8238 | 0.2886 | 12/14 | 8.3374 [5.2906, 11.3224] |
| style_matched | both_positions | R3 | 0.8856 [0.7824, 0.9667] | 0.9343 | 0.1095 | 12/14 | 8.3219 [5.2455, 11.1536] |
| style_matched | position_2 | R1 | 0.6489 [0.4488, 0.8222] | 0.5312 | 0.3642 | 8/14 | 2.2705 [-2.9720, 6.2980] |
| style_matched | position_2 | R2 | 0.7911 [0.6578, 0.9289] | 0.7708 | 0.3338 | 8/14 | 7.5683 [4.6609, 10.5807] |
| style_matched | position_2 | R3 | 0.7733 [0.5867, 0.9333] | 0.7708 | 0.3430 | 8/14 | 4.7814 [1.8803, 7.4139] |
| style_matched | position_3 | R1 | 0.8000 [0.6711, 0.9468] | 0.9545 | 0.1437 | 11/14 | 7.6042 [4.8267, 10.6383] |
| style_matched | position_3 | R2 | 0.7467 [0.5733, 0.8801] | 0.7273 | 0.3762 | 11/14 | 5.9692 [1.7957, 10.3390] |
| style_matched | position_3 | R3 | 0.9067 [0.8043, 0.9911] | 0.8864 | 0.2893 | 11/14 | 10.2987 [7.3014, 13.6658] |

## Secondary result — fixed probe on causal prefixes

| Population | Scope | Prefix | Pooled OOF AUROC (95% CI) | Fold mean | Fold SD | Valid folds | Paired mean Δ (95% CI) |
|---|---|---|---:|---:|---:|---:|---:|
| all_observers | both_positions | R1 | 0.8057 [0.7242, 0.8892] | 0.8616 | 0.1750 | 14/14 | 7.0037 [5.0247, 8.9195] |
| all_observers | both_positions | R1-R2 | 0.8707 [0.7934, 0.9420] | 0.9142 | 0.1478 | 14/14 | 8.7339 [6.9727, 10.5264] |
| all_observers | both_positions | R1-R2-R3 | 0.9030 [0.8270, 0.9690] | 0.9430 | 0.1129 | 14/14 | 9.6099 [7.8681, 11.4133] |
| all_observers | position_2 | R1 | 0.7076 [0.5723, 0.8273] | 0.6627 | 0.3449 | 14/14 | 4.1091 [1.6218, 6.6166] |
| all_observers | position_2 | R1-R2 | 0.8314 [0.7398, 0.9158] | 0.8194 | 0.2662 | 14/14 | 6.2929 [4.4558, 7.9486] |
| all_observers | position_2 | R1-R2-R3 | 0.8762 [0.7877, 0.9511] | 0.8611 | 0.2807 | 14/14 | 7.2408 [5.5334, 8.7925] |
| all_observers | position_3 | R1 | 0.8554 [0.7700, 0.9376] | 0.9345 | 0.1335 | 14/14 | 9.7410 [7.5443, 11.8108] |
| all_observers | position_3 | R1-R2 | 0.8866 [0.8106, 0.9646] | 0.9583 | 0.0845 | 14/14 | 9.6579 [7.6407, 11.7375] |
| all_observers | position_3 | R1-R2-R3 | 0.9188 [0.8512, 0.9771] | 0.9762 | 0.0620 | 14/14 | 10.1209 [8.1672, 12.2231] |
| style_matched | both_positions | R1 | 0.8033 [0.6911, 0.9085] | 0.8439 | 0.2363 | 12/14 | 5.8386 [3.0482, 8.6479] |
| style_matched | both_positions | R1-R2 | 0.8422 [0.7517, 0.9245] | 0.8855 | 0.1511 | 12/14 | 7.0880 [4.9983, 9.1686] |
| style_matched | both_positions | R1-R2-R3 | 0.8767 [0.7967, 0.9511] | 0.9197 | 0.1452 | 12/14 | 7.4993 [5.4908, 9.5359] |
| style_matched | position_2 | R1 | 0.6489 [0.4488, 0.8222] | 0.5312 | 0.3642 | 8/14 | 2.2705 [-2.9720, 6.2980] |
| style_matched | position_2 | R1-R2 | 0.7244 [0.5910, 0.8711] | 0.8715 | 0.1729 | 8/14 | 4.9194 [1.9120, 7.7543] |
| style_matched | position_2 | R1-R2-R3 | 0.7556 [0.6222, 0.8933] | 0.7882 | 0.3167 | 8/14 | 4.8734 [2.7281, 6.9564] |
| style_matched | position_3 | R1 | 0.8000 [0.6711, 0.9468] | 0.9545 | 0.1437 | 11/14 | 7.6042 [4.8267, 10.6383] |
| style_matched | position_3 | R1-R2 | 0.8489 [0.7111, 0.9556] | 0.8864 | 0.2893 | 11/14 | 6.7867 [4.2956, 9.3738] |
| style_matched | position_3 | R1-R2-R3 | 0.9111 [0.8356, 0.9867] | 1.0000 | 0.0000 | 11/14 | 7.9573 [5.7072, 10.2732] |

## Direct temporal-strengthening test

Scenario-paired bootstrap intervals below test the temporal gain itself. They avoid inferring accumulation only from separate per-round confidence intervals.

| Population | View | Transition | AUROC gain (95% CI) | Paired-score gain (95% CI) |
|---|---|---|---:|---:|
| all_observers | prefix | R1 → R1-R2 | 0.0650 [0.0146, 0.1236] | 1.7302 [0.5481, 3.0784] |
| all_observers | prefix | R1 → R1-R2-R3 | 0.0973 [0.0330, 0.1657] | 2.6062 [1.1261, 4.2253] |
| all_observers | prefix | R1-R2 → R1-R2-R3 | 0.0323 [0.0107, 0.0580] | 0.8760 [0.2948, 1.4383] |
| all_observers | round_only | R1 → R2 | 0.0692 [-0.0050, 0.1545] | 3.4604 [1.0963, 6.1568] |
| all_observers | round_only | R1 → R3 | 0.1113 [0.0281, 0.2009] | 4.3583 [1.8701, 6.7789] |
| all_observers | round_only | R2 → R3 | 0.0421 [-0.0005, 0.0916] | 0.8980 [-0.8128, 2.6057] |
| style_matched | prefix | R1 → R1-R2 | 0.0389 [-0.0281, 0.1161] | 1.2494 [-0.7409, 3.3731] |
| style_matched | prefix | R1 → R1-R2-R3 | 0.0733 [-0.0033, 0.1653] | 1.6607 [-0.5320, 4.2154] |
| style_matched | prefix | R1-R2 → R1-R2-R3 | 0.0344 [-0.0078, 0.0822] | 0.4113 [-0.5838, 1.3745] |
| style_matched | round_only | R1 → R2 | 0.0167 [-0.1142, 0.1568] | 2.4988 [-1.4818, 6.7462] |
| style_matched | round_only | R1 → R3 | 0.0822 [-0.0498, 0.2217] | 2.4833 [-1.2620, 6.5888] |
| style_matched | round_only | R2 → R3 | 0.0656 [-0.0421, 0.1738] | -0.0156 [-3.3474, 3.2745] |

For all observers combined, the R1→R3 gain excludes zero for both round-only AUROC (0.1113 [0.0281, 0.2009]) and prefix AUROC (0.0973 [0.0330, 0.1657]). The position-specific endpoint test localizes that gain mainly to position 2: its round-only and prefix intervals exclude zero, while position 3 starts high at R1 and its endpoint-gain intervals cross zero. Style-matched combined point estimates also rise, but both endpoint-gain intervals cross zero; that smaller sensitivity confirms stable early signal more strongly than it confirms temporal strengthening.

## Comparison with Experiment C

Both-position pooled OOF AUROC; Experiment C trained a separate classifier per temporal point.

| Population | View | Time | Fixed D | Separate C | Fixed − separate |
|---|---|---|---:|---:|---:|
| all_observers | prefix | R1 | 0.8057 | 0.8046 | +0.0010 |
| all_observers | prefix | R1-R2 | 0.8707 | 0.8897 | -0.0190 |
| all_observers | prefix | R1-R2-R3 | 0.9030 | 0.9030 | +0.0000 |
| all_observers | round_only | R1 | 0.8057 | 0.8046 | +0.0010 |
| all_observers | round_only | R2 | 0.8749 | 0.8824 | -0.0075 |
| all_observers | round_only | R3 | 0.9170 | 0.9240 | -0.0070 |
| style_matched | prefix | R1 | 0.8033 | 0.8056 | -0.0022 |
| style_matched | prefix | R1-R2 | 0.8422 | 0.8456 | -0.0033 |
| style_matched | prefix | R1-R2-R3 | 0.8767 | 0.8767 | +0.0000 |
| style_matched | round_only | R1 | 0.8033 | 0.8056 | -0.0022 |
| style_matched | round_only | R2 | 0.8200 | 0.8356 | -0.0156 |
| style_matched | round_only | R3 | 0.8856 | 0.8567 | +0.0289 |

## Exposure-count score shifts (R1)

The fixed both-position probe is post-stratified by the verified number of preceding position-0/1 public messages. No exposure-specific model is trained.

| Population | Position | Exposure | Pairs | Paired mean Δ (95% CI) |
|---|---:|---:|---:|---:|
| all_observers | 2 | 0 | 11 | -2.4972 [-4.7436, -0.6077] |
| all_observers | 2 | 1 | 6 | 6.5529 [3.0507, 10.1234] |
| all_observers | 2 | 2 | 14 | 11.0522 [7.8907, 14.0273] |
| all_observers | 3 | 0 | 5 | 2.2990 [-1.3726, 6.2673] |
| all_observers | 3 | 1 | 11 | 6.4135 [3.2150, 9.4403] |
| all_observers | 3 | 2 | 15 | 12.3738 [10.0889, 14.8130] |
| style_matched | 2 | 0 | 6 | -4.7015 [-13.0119, 0.6609] |
| style_matched | 2 | 1 | 4 | 10.3615 [8.2534, 12.7303] |
| style_matched | 2 | 2 | 5 | 9.8263 [3.0553, 16.7402] |
| style_matched | 3 | 0 | 2 | 6.5175 [5.8764, 7.1586] |
| style_matched | 3 | 1 | 7 | 3.8583 [1.2142, 6.6185] |
| style_matched | 3 | 2 | 6 | 13.5926 [7.8916, 20.0285] |

## Leave-one-domain-out fold AUROCs

Undefined style-subset folds are retained explicitly.

- `all_observers` / `both_positions` / `prefix` / `R1`: audit=1.0000, cyber=0.9722, eng=0.8056, env=0.3889, fin=0.9375, food=1.0000, gov=1.0000, hire=0.8333, leg=0.6250, med=0.7778, mil=0.7500, pharma=0.9722, re=1.0000, research=1.0000
- `all_observers` / `both_positions` / `prefix` / `R1-R2`: audit=1.0000, cyber=0.9722, eng=0.9444, env=0.6389, fin=0.9375, food=1.0000, gov=1.0000, hire=0.9722, leg=0.5000, med=0.8611, mil=1.0000, pharma=0.9722, re=1.0000, research=1.0000
- `all_observers` / `both_positions` / `prefix` / `R1-R2-R3`: audit=1.0000, cyber=0.9722, eng=1.0000, env=0.7222, fin=0.9375, food=1.0000, gov=1.0000, hire=0.9722, leg=0.6250, med=0.9722, mil=1.0000, pharma=1.0000, re=1.0000, research=1.0000
- `all_observers` / `both_positions` / `round_only` / `R1`: audit=1.0000, cyber=0.9722, eng=0.8056, env=0.3889, fin=0.9375, food=1.0000, gov=1.0000, hire=0.8333, leg=0.6250, med=0.7778, mil=0.7500, pharma=0.9722, re=1.0000, research=1.0000
- `all_observers` / `both_positions` / `round_only` / `R2`: audit=1.0000, cyber=0.9722, eng=1.0000, env=0.7222, fin=0.9375, food=1.0000, gov=0.7500, hire=0.9722, leg=0.5000, med=0.9444, mil=1.0000, pharma=0.9444, re=1.0000, research=1.0000
- `all_observers` / `both_positions` / `round_only` / `R3`: audit=0.7500, cyber=0.8889, eng=0.9722, env=0.7500, fin=1.0000, food=1.0000, gov=1.0000, hire=0.9722, leg=1.0000, med=1.0000, mil=1.0000, pharma=1.0000, re=1.0000, research=1.0000
- `all_observers` / `position_2` / `prefix` / `R1`: audit=0.0000, cyber=0.8889, eng=0.6667, env=0.1111, fin=1.0000, food=1.0000, gov=1.0000, hire=0.8889, leg=0.5000, med=0.3333, mil=0.2500, pharma=0.8889, re=0.7500, research=1.0000
- `all_observers` / `position_2` / `prefix` / `R1-R2`: audit=0.0000, cyber=1.0000, eng=0.8889, env=0.7778, fin=1.0000, food=1.0000, gov=1.0000, hire=0.8889, leg=0.5000, med=0.7778, mil=0.7500, pharma=0.8889, re=1.0000, research=1.0000
- `all_observers` / `position_2` / `prefix` / `R1-R2-R3`: audit=0.0000, cyber=0.8889, eng=1.0000, env=0.6667, fin=1.0000, food=1.0000, gov=1.0000, hire=1.0000, leg=0.5000, med=1.0000, mil=1.0000, pharma=1.0000, re=1.0000, research=1.0000
- `all_observers` / `position_2` / `round_only` / `R1`: audit=0.0000, cyber=0.8889, eng=0.6667, env=0.1111, fin=1.0000, food=1.0000, gov=1.0000, hire=0.8889, leg=0.5000, med=0.3333, mil=0.2500, pharma=0.8889, re=0.7500, research=1.0000
- `all_observers` / `position_2` / `round_only` / `R2`: audit=0.0000, cyber=1.0000, eng=1.0000, env=0.7778, fin=1.0000, food=1.0000, gov=1.0000, hire=0.6667, leg=0.5000, med=1.0000, mil=1.0000, pharma=1.0000, re=1.0000, research=1.0000
- `all_observers` / `position_2` / `round_only` / `R3`: audit=1.0000, cyber=0.6667, eng=0.8889, env=0.6667, fin=1.0000, food=1.0000, gov=1.0000, hire=1.0000, leg=1.0000, med=1.0000, mil=1.0000, pharma=0.8889, re=1.0000, research=1.0000
- `all_observers` / `position_3` / `prefix` / `R1`: audit=1.0000, cyber=1.0000, eng=1.0000, env=0.7778, fin=1.0000, food=0.7500, gov=1.0000, hire=0.5556, leg=1.0000, med=1.0000, mil=1.0000, pharma=1.0000, re=1.0000, research=1.0000
- `all_observers` / `position_3` / `prefix` / `R1-R2`: audit=1.0000, cyber=1.0000, eng=1.0000, env=0.8889, fin=1.0000, food=1.0000, gov=1.0000, hire=0.7778, leg=0.7500, med=1.0000, mil=1.0000, pharma=1.0000, re=1.0000, research=1.0000
- `all_observers` / `position_3` / `prefix` / `R1-R2-R3`: audit=1.0000, cyber=1.0000, eng=1.0000, env=0.7778, fin=1.0000, food=1.0000, gov=1.0000, hire=0.8889, leg=1.0000, med=1.0000, mil=1.0000, pharma=1.0000, re=1.0000, research=1.0000
- `all_observers` / `position_3` / `round_only` / `R1`: audit=1.0000, cyber=1.0000, eng=1.0000, env=0.7778, fin=1.0000, food=0.7500, gov=1.0000, hire=0.5556, leg=1.0000, med=1.0000, mil=1.0000, pharma=1.0000, re=1.0000, research=1.0000
- `all_observers` / `position_3` / `round_only` / `R2`: audit=1.0000, cyber=0.8889, eng=1.0000, env=0.6667, fin=0.7500, food=1.0000, gov=1.0000, hire=0.7778, leg=0.7500, med=1.0000, mil=0.7500, pharma=0.7778, re=1.0000, research=1.0000
- `all_observers` / `position_3` / `round_only` / `R3`: audit=1.0000, cyber=1.0000, eng=1.0000, env=0.7778, fin=1.0000, food=0.7500, gov=1.0000, hire=0.8889, leg=1.0000, med=0.8889, mil=1.0000, pharma=1.0000, re=1.0000, research=1.0000
- `style_matched` / `both_positions` / `prefix` / `R1`: audit=undefined (no retained observer pairs in held-out domain), cyber=0.8889, eng=0.8000, env=0.2500, fin=1.0000, food=1.0000, gov=1.0000, hire=1.0000, leg=1.0000, med=0.6875, mil=0.5000, pharma=1.0000, re=1.0000, research=undefined (no retained observer pairs in held-out domain)
- `style_matched` / `both_positions` / `prefix` / `R1-R2`: audit=undefined (no retained observer pairs in held-out domain), cyber=1.0000, eng=0.8000, env=0.7500, fin=1.0000, food=1.0000, gov=1.0000, hire=0.8889, leg=0.5000, med=0.7500, mil=1.0000, pharma=0.9375, re=1.0000, research=undefined (no retained observer pairs in held-out domain)
- `style_matched` / `both_positions` / `prefix` / `R1-R2-R3`: audit=undefined (no retained observer pairs in held-out domain), cyber=0.8889, eng=0.9600, env=0.5000, fin=1.0000, food=1.0000, gov=1.0000, hire=1.0000, leg=0.7500, med=0.9375, mil=1.0000, pharma=1.0000, re=1.0000, research=undefined (no retained observer pairs in held-out domain)
- `style_matched` / `both_positions` / `round_only` / `R1`: audit=undefined (no retained observer pairs in held-out domain), cyber=0.8889, eng=0.8000, env=0.2500, fin=1.0000, food=1.0000, gov=1.0000, hire=1.0000, leg=1.0000, med=0.6875, mil=0.5000, pharma=1.0000, re=1.0000, research=undefined (no retained observer pairs in held-out domain)
- `style_matched` / `both_positions` / `round_only` / `R2`: audit=undefined (no retained observer pairs in held-out domain), cyber=1.0000, eng=0.9200, env=1.0000, fin=1.0000, food=1.0000, gov=0.0000, hire=0.7778, leg=0.5000, med=0.9375, mil=1.0000, pharma=0.7500, re=1.0000, research=undefined (no retained observer pairs in held-out domain)
- `style_matched` / `both_positions` / `round_only` / `R3`: audit=undefined (no retained observer pairs in held-out domain), cyber=0.6667, eng=0.9200, env=0.7500, fin=1.0000, food=1.0000, gov=1.0000, hire=1.0000, leg=1.0000, med=0.8750, mil=1.0000, pharma=1.0000, re=1.0000, research=undefined (no retained observer pairs in held-out domain)
- `style_matched` / `position_2` / `prefix` / `R1`: audit=undefined (no retained observer pairs in held-out domain), cyber=0.7500, eng=0.5556, env=0.0000, fin=undefined (no retained observer pairs in held-out domain), food=undefined (no retained observer pairs in held-out domain), gov=1.0000, hire=1.0000, leg=undefined (no retained observer pairs in held-out domain), med=0.4444, mil=0.0000, pharma=0.5000, re=undefined (no retained observer pairs in held-out domain), research=undefined (no retained observer pairs in held-out domain)
- `style_matched` / `position_2` / `prefix` / `R1-R2`: audit=undefined (no retained observer pairs in held-out domain), cyber=1.0000, eng=0.5556, env=1.0000, fin=undefined (no retained observer pairs in held-out domain), food=undefined (no retained observer pairs in held-out domain), gov=1.0000, hire=0.7500, leg=undefined (no retained observer pairs in held-out domain), med=0.6667, mil=1.0000, pharma=1.0000, re=undefined (no retained observer pairs in held-out domain), research=undefined (no retained observer pairs in held-out domain)
- `style_matched` / `position_2` / `prefix` / `R1-R2-R3`: audit=undefined (no retained observer pairs in held-out domain), cyber=0.7500, eng=0.7778, env=0.0000, fin=undefined (no retained observer pairs in held-out domain), food=undefined (no retained observer pairs in held-out domain), gov=1.0000, hire=1.0000, leg=undefined (no retained observer pairs in held-out domain), med=0.7778, mil=1.0000, pharma=1.0000, re=undefined (no retained observer pairs in held-out domain), research=undefined (no retained observer pairs in held-out domain)
- `style_matched` / `position_2` / `round_only` / `R1`: audit=undefined (no retained observer pairs in held-out domain), cyber=0.7500, eng=0.5556, env=0.0000, fin=undefined (no retained observer pairs in held-out domain), food=undefined (no retained observer pairs in held-out domain), gov=1.0000, hire=1.0000, leg=undefined (no retained observer pairs in held-out domain), med=0.4444, mil=0.0000, pharma=0.5000, re=undefined (no retained observer pairs in held-out domain), research=undefined (no retained observer pairs in held-out domain)
- `style_matched` / `position_2` / `round_only` / `R2`: audit=undefined (no retained observer pairs in held-out domain), cyber=1.0000, eng=0.7778, env=1.0000, fin=undefined (no retained observer pairs in held-out domain), food=undefined (no retained observer pairs in held-out domain), gov=0.0000, hire=0.5000, leg=undefined (no retained observer pairs in held-out domain), med=0.8889, mil=1.0000, pharma=1.0000, re=undefined (no retained observer pairs in held-out domain), research=undefined (no retained observer pairs in held-out domain)
- `style_matched` / `position_2` / `round_only` / `R3`: audit=undefined (no retained observer pairs in held-out domain), cyber=0.5000, eng=0.6667, env=0.0000, fin=undefined (no retained observer pairs in held-out domain), food=undefined (no retained observer pairs in held-out domain), gov=1.0000, hire=1.0000, leg=undefined (no retained observer pairs in held-out domain), med=1.0000, mil=1.0000, pharma=1.0000, re=undefined (no retained observer pairs in held-out domain), research=undefined (no retained observer pairs in held-out domain)
- `style_matched` / `position_3` / `prefix` / `R1`: audit=undefined (no retained observer pairs in held-out domain), cyber=1.0000, eng=1.0000, env=1.0000, fin=1.0000, food=0.5000, gov=undefined (no retained observer pairs in held-out domain), hire=1.0000, leg=1.0000, med=1.0000, mil=1.0000, pharma=1.0000, re=1.0000, research=undefined (no retained observer pairs in held-out domain)
- `style_matched` / `position_3` / `prefix` / `R1-R2`: audit=undefined (no retained observer pairs in held-out domain), cyber=1.0000, eng=1.0000, env=1.0000, fin=0.0000, food=1.0000, gov=undefined (no retained observer pairs in held-out domain), hire=1.0000, leg=0.7500, med=1.0000, mil=1.0000, pharma=1.0000, re=1.0000, research=undefined (no retained observer pairs in held-out domain)
- `style_matched` / `position_3` / `prefix` / `R1-R2-R3`: audit=undefined (no retained observer pairs in held-out domain), cyber=1.0000, eng=1.0000, env=1.0000, fin=1.0000, food=1.0000, gov=undefined (no retained observer pairs in held-out domain), hire=1.0000, leg=1.0000, med=1.0000, mil=1.0000, pharma=1.0000, re=1.0000, research=undefined (no retained observer pairs in held-out domain)
- `style_matched` / `position_3` / `round_only` / `R1`: audit=undefined (no retained observer pairs in held-out domain), cyber=1.0000, eng=1.0000, env=1.0000, fin=1.0000, food=0.5000, gov=undefined (no retained observer pairs in held-out domain), hire=1.0000, leg=1.0000, med=1.0000, mil=1.0000, pharma=1.0000, re=1.0000, research=undefined (no retained observer pairs in held-out domain)
- `style_matched` / `position_3` / `round_only` / `R2`: audit=undefined (no retained observer pairs in held-out domain), cyber=1.0000, eng=1.0000, env=1.0000, fin=0.0000, food=0.7500, gov=undefined (no retained observer pairs in held-out domain), hire=0.0000, leg=0.7500, med=1.0000, mil=1.0000, pharma=0.5000, re=1.0000, research=undefined (no retained observer pairs in held-out domain)
- `style_matched` / `position_3` / `round_only` / `R3`: audit=undefined (no retained observer pairs in held-out domain), cyber=1.0000, eng=1.0000, env=1.0000, fin=1.0000, food=0.7500, gov=undefined (no retained observer pairs in held-out domain), hire=1.0000, leg=1.0000, med=0.0000, mil=1.0000, pharma=1.0000, re=1.0000, research=undefined (no retained observer pairs in held-out domain)

## Interpretation

A single full-discussion sentinel direction retains reliable R1 predictive signal.
For all observers, pooled AUROC changes by +0.1113 from R1 to R3 for round-only activations and +0.0973 from R1 to the full prefix.
The style-matched sensitivity also retains reliable R1 signal, so the early fixed-direction result is not explained solely by the audited prompt-style mismatch. Its R1→R3 temporal gain is positive in point estimate but uncertain, with bootstrap intervals crossing zero.
The direct endpoint test supports temporal strengthening for all observers combined, driven primarily by position 2. Position 3 already carries strong early signal and does not show a reliable additional endpoint gain. AUROC and paired score shift still do not establish a causal mechanism or prove that exposure itself caused the within-agent change.

No intervention thresholds, TFIR, PIC, or E2 analysis were run.
