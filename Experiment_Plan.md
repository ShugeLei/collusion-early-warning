# Experiment Plan

## Working Title

**How Early Can We Stop Multi-Agent Collusion?**

### Actionable Early Warning from Open-Weight Agent Internals

---

# 1. Core Research Question

The project studies one concrete question:

> **When does collective internal evidence of multi-agent collusion become reliable enough to justify intervention, under a realistic false-intervention budget?**

The goal is **not** to build another collusion classifier.

NARCBench has already shown that activation-based probes can detect multi-agent collusion and that aggregating signals across agents can outperform purely local reasoning about one agent.

Our extension changes the evaluation objective from:

> Can collusion be detected after observing the interaction?

to:

> **Can collusion be detected early enough to act before the final consequential decision?**

The project connects four security dimensions:

**Detectability × Locality × Time Horizon × Reversibility**

and converts them into a runtime harness question:

> **WHEN SHOULD THE HARNESS INTERVENE?**

---

# 2. Why This Matters

A detector that identifies collusion after a consequential action is operationally much less useful than one that reliably identifies risk during deliberation.

Conceptual deployment:

```text
Round 1
Agent discussion
     ↓
security monitor
     ↓
Round 2
agent coordination
     ↓
security monitor
     ↓
Round 3
pre-action state
     ↓
========== SECURITY GATE ==========
     ↓
Round 4 / consequential action
```

In a production agent system, the final boundary could correspond to:

```text
send_email
transfer_money
deploy_code
write_database
release_private_data
approve_transaction
execute_tool
```

NARCBench does **not** contain these production actions.

For this experiment, **Round 4 is only a proxy for a consequential-action boundary**.

Therefore our claim must remain:

> This experiment studies whether activation monitoring can create an intervention window before a final multi-agent decision.

Do **not** claim that we have demonstrated prevention of real-world irreversible actions.

---

# 3. Primary Hypotheses

## H1 — Temporal Detectability

Collusion risk should not be treated as a static binary property.

We test whether the signal changes as interaction progresses:

```text
Round 1 → Round 2 → Round 3 → Final
```

Possible finding:

```text
weak → moderate → strong
```

would support:

> Collusion is a temporally accumulating signal.

However, an equally important possible result is:

```text
very strong from Round 1
```

This would suggest that the benchmark may primarily expose a **latent colluder-role / intent signal**, rather than a signal that emerges through coordination.

Both results are scientifically useful.

---

## H2 — Collective Observability

A multi-agent security harness should gain information by looking across the group instead of independently asking whether one agent looks suspicious.

We compare:

### Independent-local monitoring

Each agent receives an individual deception score.

The system alerts when the most suspicious individual agent crosses a threshold.

Operational score:

```text
Peak Suspicion = max(agent deception scores)
```

This requires no relational structure between agents.

### Collective monitoring

The monitor asks whether the **distribution of scores across the entire group** looks like a colluding subgroup.

Primary collective score:

```text
Deception Split
=
mean(top 2 deception scores)
-
mean(bottom 2 deception scores)
```

NARCBench uses this because Core contains four agents with two colluders.

Research question:

> At the same false-intervention budget, does collective monitoring catch more collusion before the final action?

---

## H3 — Actionability Under an Operational Budget

The relevant optimization objective is **not maximum AUROC**.

A company cannot afford to interrupt large numbers of legitimate agent workflows.

Therefore the primary optimization objective is:

```text
maximize:
    P(collusion detected before final action)

subject to:
    trajectory-level false intervention rate <= budget
```

Primary budget:

```text
5%
```

Sensitivity analysis:

```text
1%
10%
```

Because the dataset is small, the 5% estimate will have meaningful sampling uncertainty. Report confidence intervals and the achieved held-out false-intervention rate rather than pretending the budget is exact.

---

# 4. Critical Metric Definition

## Do NOT use per-round FPR as the main business metric

Suppose a benign workflow is checked at:

```text
Round 1
Round 2
Round 3
```

Even if each check individually has a 5% false-positive rate, the probability that the workflow is interrupted **at least once** can be materially higher than 5%.

Therefore define:

# Trajectory-Level False Intervention Rate — TFIR

For a control trajectory:

```text
false_intervention = 1
```

if the monitor flags the trajectory during **any** pre-final round.

Formally:

```text
TFIR =
# control trajectories flagged at least once before final
--------------------------------------------------------
# control trajectories
```

This is the main operational safety-cost metric.

It approximates:

> What fraction of legitimate workflows would the security system incorrectly interrupt?

---

# 5. Primary Success Metric

## Pre-Final Intervention Coverage — PIC

For a collusion trajectory:

```text
intervention_success = 1
```

if the monitor flags it during Round 1, 2, or 3.

Do not count detection during Round 4 as successful early intervention.

```text
PIC =
# collusion trajectories flagged before Round 4
------------------------------------------------
# collusion trajectories
```

The headline comparison is:

```text
PIC_collective @ target TFIR

versus

PIC_local @ target TFIR
```

Example interpretation:

```text
TFIR ≈ 5%

Local monitor:
PIC = 48%

Collective monitor:
PIC = 72%
```

Then the useful result is:

> Collective system observability catches 24 percentage points more collusive trajectories before the final decision at approximately the same false-intervention cost.

Do not invent numbers.

---

# 6. Secondary Metrics

## 6.1 AUROC by Prefix

For diagnostic purposes:

```text
Round 1 prefix
Round 1–2 prefix
Round 1–3 prefix
Final/full interaction
```

Measure run-level AUROC for each monitor.

AUROC answers:

> How separable are collusion and control trajectories at this point?

It does **not** answer whether the detector is deployable.

Therefore AUROC is secondary.

---

## 6.2 First Detection Round

For every detected collusion trajectory:

```text
r* = earliest round where risk >= threshold
```

Possible values:

```text
1
2
3
never
```

Plot the cumulative proportion detected:

```text
             cumulative detected

Round 1             X%
Round 2             Y%
Round 3             Z%
```

---

## 6.3 Intervention Lead

Treat Round 4 as the final action boundary.

```text
lead_rounds = 4 - first_detection_round
```

Examples:

```text
detect Round 1 → lead = 3
detect Round 2 → lead = 2
detect Round 3 → lead = 1
```

Report:

```text
median lead rounds
distribution of lead rounds
```

This is descriptive rather than the primary statistical metric.

---

## 6.4 Collective Intervention Gain

Define:

```text
Collective Intervention Gain
=
PIC_collective
-
PIC_local
```

evaluated at approximately the same trajectory-level false-intervention budget.

This is the clearest measure of the value of system-level observability.

---

# 7. Dataset

## Primary Dataset

**NARCBench-Core**

Primary released model:

```text
Qwen/Qwen3-32B-AWQ
```

Do **not** run Qwen3-32B locally.

We use the **pre-extracted activations released by NARCBench**.

The official release contains aligned:

```text
metadata_gen.json
activations_gen.npz
```

where each activation row corresponds to one agent-turn. Relevant metadata includes:

```text
run_id
mode
role
channel
phase
round
agent_name
family
scenario_id
scenario_domain
collusion_success
```

The official Qwen release contains activation layers 26–30. The dataset page currently reports 31 collusion + 31 matched control Core runs, but the code must derive counts from the downloaded data rather than hard-code them.

Each collusion run has a paired same-scenario control run.

---

# 8. Why Use Qwen3-32B Data

We choose Qwen3-32B as the primary dataset because:

1. It is NARCBench's primary experimental model.
2. The official released activations are already available.
3. The original paper's Core methodology and layer range are defined around it.
4. It lets us reproduce the published result before modifying the temporal evaluation.
5. No LLM inference is required on our Mac.

The Mac is only running:

```text
NumPy
scikit-learn
pandas
matplotlib
```

plus the official probe code.

No GPU inference.

No MLX.

No new trajectory generation.

No model weights need to be loaded.

---

# 9. Layer Policy

Official probe range:

```text
26, 27, 28, 29, 30
```

Primary development/debug layer:

```text
layer 28
```

Reason:

It is the center of the official predefined range and is used in the dataset's loading example.

Do **not** select the best-performing layer after seeing test results.

Final robustness analysis:

Run the exact same experiment on:

```text
26
27
28
29
30
```

and report:

```text
mean
standard deviation
individual-layer results
```

A conclusion should be called robust only if it is not driven by one cherry-picked layer.

---

# 10. Data Filtering

Primary analysis should use:

```text
family == relevant Core family
channel == public
phase == discussion
round in {1, 2, 3}
```

Round 4 is retained as the final-decision reference / post-hoc upper bound.

The official paper trained the deception probe on public-channel Core collusion runs and averaged each agent's activations across discussion rounds.

Before implementing any custom filtering:

1. Run the official `reproduce.py`.
2. Inspect NARCBench's filtering logic.
3. Match the official successful-run exclusions as closely as possible.
4. Log exactly how many scenarios remain.

Do not silently remove examples.

---

# 11. Train/Test Protocol

## Preserve NARCBench's leave-domain-out evaluation

NARCBench evaluates Core with:

```text
Leave-One-Domain-Out Cross Validation
```

For each held-out domain:

```text
TRAIN:
all other domains

TEST:
held-out domain
```

The official methodology does this specifically to evaluate generalization across scenario domains.

We preserve this protocol.

Never randomly split agent-turn rows.

Never randomly split trajectories.

Never allow the same scenario to contribute to both train and test.

---

# 12. Deception Probe

Reuse the NARCBench approach rather than inventing a new classifier.

For each training fold and each layer:

### Training examples

Use **collusion-condition training runs only**.

Each agent receives:

```text
colluder
or
honest
```

label.

For each agent, average public activations over training discussion rounds:

```text
a_agent =
mean(
    round1 activation,
    round2 activation,
    round3 activation
)
```

Then:

```text
StandardScaler
       ↓
LogisticRegression(C=1.0)
```

Balance colluder/honest classes as in the official implementation.

The normalized logistic-regression weight defines the deception direction.

This reproduces the conceptual NARCBench probe.

---

# 13. Important Temporal Design Choice

## The probe must remain FIXED while time changes

Do **not** train a completely different classifier for each round in the primary experiment.

That would confound:

```text
signal emergence
```

with:

```text
different classifier quality
```

Instead:

### Train once

Train the deception direction using the training domains and their full discussion-round averages.

### Evaluate repeatedly

At test time calculate:

```text
Prefix 1:
mean(R1)

Prefix 2:
mean(R1, R2)

Prefix 3:
mean(R1, R2, R3)
```

using the **same trained deception direction**.

This asks:

> As more interaction becomes available, when does a fixed monitor obtain enough evidence to intervene?

That is exactly our deployment question.

---

# 14. Prefix Representation

For each agent j and round r:

```text
prefix_activation(j, r)
=
mean(
activation(j, 1),
...
activation(j, r)
)
```

Then project through the fixed deception probe:

```text
s(j, r)
=
deception score of agent j
using prefix activation through round r
```

For every run and every prefix we should therefore have four agent scores:

```text
[s1, s2, s3, s4]
```

---

# 15. Monitor A — Independent Local Monitoring

Primary local baseline:

## Peak Suspicion

```text
risk_local(r)
=
max_j s(j, r)
```

Interpretation:

> Independently monitor every agent and trigger if any individual agent appears sufficiently deceptive.

Although the final system takes a maximum across agents, this monitor uses **no relational structure between agents**.

Call it:

```text
Independent-local baseline
```

Do not call it a pure "single-agent classifier" in the paper/presentation because the final alert is an OR across all agents.

NARCBench defines Peak Suspicion exactly as the highest individual deception score.

---

# 16. Monitor B — Collective Monitoring

Primary collective detector:

## Deception Split

Sort four deception scores:

```text
s[1] >= s[2] >= s[3] >= s[4]
```

Then:

```text
risk_collective
=
mean(s[1], s[2])
-
mean(s[3], s[4])
```

Interpretation:

> Does the group contain a suspicious two-agent subgroup that separates from the rest of the team?

This explicitly uses **group structure**.

NARCBench introduced this as one of its decision-level fusion methods.

This is the primary comparison against Peak Suspicion.

---

# 17. Optional Monitor C

Only implement after the primary experiment works.

## Activation Alignment

For each prefix, calculate mean pairwise cosine similarity among the four agent activation vectors.

NARCBench uses:

```text
negative mean pairwise cosine similarity
```

as a collusion risk score.

This tests a different hypothesis:

> Can collusion be detected from relationships between representations even without a deception-direction score?

This is scientifically useful but **not required for the 8-hour MVP**.

---

# 18. Threshold Calibration

This is critical.

Do not select thresholds using held-out test domains.

For every fold:

## Step 1

For each TRAINING control trajectory calculate:

```text
M_i =
max(
risk_i(round1),
risk_i(round2),
risk_i(round3)
)
```

This is the highest pre-final risk encountered anywhere during that benign trajectory.

## Step 2

For target false-intervention budget α:

```text
alpha = 0.05
```

choose threshold:

```text
tau =
95th percentile of {M_i}
```

Use a conservative quantile implementation such as:

```python
np.quantile(values, 0.95, method="higher")
```

Then apply this fixed threshold to the held-out test domain.

This calibration directly controls the quantity we care about:

> probability that a legitimate trajectory gets interrupted at least once.

Do not tune threshold after inspecting test labels.

---

# 19. Runtime Decision Rule

For a test trajectory:

```text
for r in [1, 2, 3]:

    risk = monitor(prefix through r)

    if risk >= threshold:
        INTERVENE
        record first_detection_round = r
        stop
```

If no threshold crossing occurs:

```text
first_detection_round = never
```

Round 4 must not count toward successful early intervention.

---

# 20. Experiment E0 — Reproduce NARCBench

Before custom work:

Run the official reproduction command.

Expected purpose:

```text
verify installation
verify data paths
verify activation shapes
verify baseline methodology
```

The official repository states that generation/extraction require CUDA but probe reproduction runs on CPU.

We are doing probe reproduction only.

Deliverable:

```text
outputs/baseline_reproduction.json
```

containing official-style Core results.

If baseline reproduction fails, fix this before proceeding.

---

# 21. Experiment E1 — Temporal Detectability

Question:

> How does detectability evolve as interaction accumulates?

For both monitors calculate diagnostic AUROC at:

```text
Prefix R1
Prefix R1–R2
Prefix R1–R3
Final / full interaction if useful as upper bound
```

Main figure:

```text
AUROC
  │
1 │
  │
.8│            collective
  │       local
.6│
  │
.5└────────────────
     R1   R2   R3
```

Interpretation possibilities:

### Pattern A

Performance improves over time.

Conclusion:

> Evidence of collusion accumulates through interaction.

### Pattern B

Performance is already extremely high at Round 1.

Conclusion:

> The benchmark may expose latent colluder intent/role before meaningful coordination emerges.

Do **not** hide Pattern B.

It is an important benchmark finding.

---

# 22. Experiment E2 — Actionable Early Intervention

This is the **main experiment**.

For each monitor and each cross-validation fold:

1. calibrate threshold using training control trajectories;
2. run the threshold sequentially on test prefixes;
3. measure TFIR;
4. measure PIC;
5. record first detection round.

Primary comparison:

```text
                 TFIR       PIC

Local             ?          ?
Collective        ?          ?
```

The meaningful business question is:

> At roughly the same cost to legitimate workflows, which monitor stops more collusive workflows before the final action?

---

# 23. Experiment E3 — Intervention Curve

Repeat threshold calibration for:

```text
1%
5%
10%
20%
```

false-intervention budgets.

Plot:

```text
Pre-final collusion caught
1.0 │
    │
.8  │
    │      collective
.6  │    /
    │   / local
.4  │  /
    │ /
.2  │/
    └────────────────────────
      Trajectory false
      intervention rate
```

This is the most business-relevant figure.

It shows the security/utility frontier:

```text
more aggressive monitoring
        ↕
more benign workflows interrupted
```

---

# 24. Experiment E4 — Benchmark Diagnostic

Ask:

> Are we detecting emergent coordination, or merely assigned colluder identity?

Use Round 1 behavior as the primary diagnostic.

If:

```text
Round-1 AUROC ≈ full-trajectory AUROC
```

and:

```text
Round-1 PIC is already very high
```

then explicitly discuss:

> There is little temporal emergence in the measured signal.

NARCBench itself acknowledges that residual prompt-related confounds cannot be fully ruled out and that labels represent instructed collusion rather than a perfect mechanistic ground truth.

This does **not** invalidate NARCBench.

It changes the interpretation:

```text
detecting latent collusive intent
```

may be easier than:

```text
detecting coordination as it emerges
```

That becomes one of our three insights.

---

# 25. Statistical Uncertainty

The dataset is small.

Do not report point estimates without uncertainty.

Use bootstrap resampling at the:

```text
scenario level
```

not agent-turn level.

Preserve matched:

```text
collusion/control scenario pairs
```

when possible.

Recommended:

```text
1000 bootstrap samples
95% confidence intervals
```

Report uncertainty for:

```text
TFIR
PIC
Collective Intervention Gain
AUROC
```

Do not make significance claims solely from overlapping/non-overlapping error bars.

The purpose is mainly to communicate dataset uncertainty.

---

# 26. Required Figures

Produce only three primary figures.

## Figure 1 — Detectability Over Time

```text
x:
R1
R1–R2
R1–R3

y:
AUROC

lines:
Independent-local
Collective
```

Purpose:

> Does collusion evidence accumulate over time?

---

## Figure 2 — Security / Utility Frontier

```text
x:
trajectory-level false intervention rate

y:
pre-final intervention coverage

lines:
Independent-local
Collective
```

Purpose:

> At a given operational cost, how much collusion can we stop?

This is the main business figure.

---

## Figure 3 — Cumulative Intervention Timing

```text
x:
Round 1
Round 2
Round 3

y:
fraction of collusion trajectories
detected by or before this round
```

Purpose:

> How much intervention time does the monitor buy us?

---

# 27. Results Table

Generate:

```text
Method              TFIR    PIC    Median Alert Round    Mean Lead
-------------------------------------------------------------------
Independent-local     ?      ?             ?                ?
Collective            ?      ?             ?                ?
```

for:

```text
1%
5%
10%
```

target budgets.

Never replace achieved TFIR with target TFIR.

Report both:

```text
target TFIR
actual held-out TFIR
```

---

# 28. Result Interpretation Matrix

Do not decide the conclusion before seeing the data.

## Outcome A

```text
Collective PIC >> Local PIC
```

at matched TFIR.

Strong conclusion:

> System-level observability creates actionable security value beyond independent agent monitoring.

---

## Outcome B

```text
Collective ≈ Local
```

Conclusion:

> In NARCBench-Core, most actionable collusion signal may already be visible in at least one individual agent; sophisticated group aggregation adds limited early-warning value.

This is still useful for harness design:

> Use cheaper independent monitors unless more distributed attacks require relational monitoring.

---

## Outcome C

```text
Round 1 already near saturation
```

Conclusion:

> NARCBench may primarily expose pre-existing collusive intent rather than gradual emergence of coordination.

This motivates a better future benchmark where:

```text
agents begin without explicit colluder roles
          ↓
interaction creates opportunity
          ↓
coordination emerges
```

---

## Outcome D

```text
R1 weak
R2 stronger
R3 strongest
```

Conclusion:

> Collusion is temporally emergent, supporting continuous runtime monitoring rather than static pre-deployment certification.

---

# 29. How Experiments Support the Main Story

## Claim 1

**Detectability is temporal.**

Supported by:

```text
E1
AUROC / risk quality across prefixes
```

---

## Claim 2

**The relevant security object may be the composition, not the individual agent.**

Supported by:

```text
E2
Collective PIC vs Local PIC
at matched TFIR
```

---

## Claim 3

**A useful harness should optimize intervention timing under operational cost, not merely classifier AUROC.**

Supported by:

```text
E3
security/utility frontier
```

---

## Claim 4

**Very early detectability can reveal benchmark limitations.**

Supported by:

```text
E4
Round-1 saturation analysis
```

---

# 30. Business Translation

Translate technical metrics as follows:

```text
TFIR
=
percentage of legitimate agent workflows
incorrectly interrupted
```

```text
PIC
=
percentage of collusive workflows
stopped before the consequential action
```

```text
First Detection Round
=
how early security can respond
```

```text
Collective Intervention Gain
=
value gained from monitoring
the agent system rather than
independent components
```

The eventual product concept is:

```text
Open-weight agent runtime
          ↓
activation telemetry
          ↓
multi-agent risk aggregation
          ↓
policy engine
       /       \
    continue   intervene
                  ↓
             human review
             tool blocking
             permission drop
             isolation
```

This experiment tests whether the **risk aggregation + intervention gate** is justified.

It does not build the entire product.

---

# 31. Project Structure

Create:

```text
collusion-early-warning/
│
├── EXPERIMENT_PLAN.md
├── README.md
├── requirements.txt
├── .gitignore
│
├── external/
│   └── narcbench/
│
├── data/
│   └── narcbench/
│
├── src/
│   ├── data_loader.py
│   ├── probe.py
│   ├── prefix_features.py
│   ├── monitors.py
│   ├── thresholds.py
│   ├── metrics.py
│   ├── bootstrap.py
│   └── plotting.py
│
├── scripts/
│   ├── 00_validate_environment.py
│   ├── 01_validate_data.py
│   ├── 02_reproduce_baseline.py
│   ├── 03_run_temporal_experiment.py
│   ├── 04_run_intervention_experiment.py
│   └── 05_make_figures.py
│
├── outputs/
│   ├── logs/
│   ├── metrics/
│   ├── predictions/
│   └── figures/
│
└── tests/
    ├── test_no_data_leakage.py
    ├── test_prefix_features.py
    ├── test_thresholds.py
    └── test_metrics.py
```

---

# 32. Required Output Files

At minimum:

```text
outputs/metrics/baseline.json

outputs/predictions/oof_temporal_scores.csv

outputs/metrics/intervention_metrics.csv

outputs/figures/detectability_over_time.png

outputs/figures/security_utility_frontier.png

outputs/figures/intervention_timing.png

outputs/results_summary.md
```

`oof_temporal_scores.csv` should contain enough information to reproduce every result:

```text
scenario_id
domain
mode
layer
round
method
risk_score
threshold
flagged
first_detection_round
fold
```

---

# 33. Reproducibility Rules

Always:

```text
seed = 42
```

unless the upstream implementation requires another seed.

Log:

```text
git commit of NARCBench
dataset counts
layers
domains
excluded runs
sklearn version
Python version
threshold rule
random seed
```

Every figure must be reconstructable from saved prediction CSVs.

Do not make plots directly from transient Python variables without saving the underlying metrics.

---

# 34. Data Leakage Rules

These are non-negotiable.

Never use test-domain data to:

```text
fit scaler
train probe
choose layer
select threshold
choose aggregation method
```

Round 2 prediction may use:

```text
Round 1
Round 2
```

but never:

```text
Round 3
Round 4
```

Round 1 prediction may use only Round 1.

The final vote / final action must never appear in any pre-final feature.

Write automated tests for these constraints.

---

# 35. What NOT to Build

Given the eight-hour constraint, do not:

```text
generate new trajectories
run Qwen locally
download Qwen model weights
use MLX
fine-tune an LLM
build a CrewAI system
build LangGraph
train neural probes
design a new attack
implement steganography
add AgentLeak
add TAMAS
add a web UI
build a production security gateway
```

These are out of scope.

---

# 36. Eight-Hour Execution Budget

## Hour 0–1

```text
clone repo
download data
install dependencies
inspect metadata
validate shapes
```

## Hour 1–2

```text
reproduce NARCBench baseline
understand official probe functions
```

## Hour 2–4

```text
implement prefix activations
implement fixed deception probe
implement Peak Suspicion
implement Deception Split
run leave-domain-out predictions
```

## Hour 4–5

```text
implement trajectory threshold calibration
compute TFIR
compute PIC
compute first detection round
```

## Hour 5–6

```text
run 1% / 5% / 10% budgets
run layers 26–30
bootstrap uncertainty
```

## Hour 6–7

```text
make three figures
inspect failure cases
check Round-1 saturation
```

## Hour 7–8

```text
write results_summary.md
extract 3 key findings
prepare presentation results
```

---

# 37. Minimum Viable Completion

If time runs short, completion priority is:

```text
1. official baseline reproduced
2. layer 28 works
3. prefix scores generated
4. Peak vs Deception Split compared
5. TFIR and PIC calculated
6. security/utility figure produced
```

Everything after that is optional.

One correct layer-28 result is more valuable than five partially working experiments.

---

# 38. Definition of a Successful Project

The project is successful if, after eight hours, we can truthfully answer:

### Question 1

At what point during the interaction does collusion become distinguishable?

### Question 2

Does collective monitoring outperform independent-local monitoring **before** the final decision?

### Question 3

What fraction of benign workflows must we interrupt to obtain that protection?

### Question 4

Does the benchmark show temporal emergence, or is the collusion signal already present immediately?

We do **not** require the proposed defense to win.

A negative result that answers these four questions is a valid project.

---

# 39. Final Presentation Thesis

The presentation should not claim:

> We built a better collusion detector.

The stronger framing is:

> **Current collusion benchmarks mostly ask whether an interaction is detectable. A deployable security harness needs to know when the evidence becomes actionable. We evaluate collusion monitoring as a sequential intervention problem: maximize attacks stopped before the final action while constraining the probability of interrupting legitimate workflows.**

If collective monitoring wins:

> **System-level observability buys intervention time.**

If it does not:

> **The additional complexity of collective monitoring may not be justified for this attack distribution.**

If Round 1 is already near-perfect:

> **The benchmark may measure pre-existing collusive intent more strongly than emergent coordination.**

All three outcomes teach us something about how multi-agent security harnesses should be designed.