# ISRM projects: evidential ranking and sandbox test plan

Date: 2026-11-06. Scope: every ISRM-related folder on D:\ (about 25 projects, two
manuscripts, the Equation/Equations formal corpus). Method: six parallel read-only
audits, one per cluster, each extracting the central claim, evidence class, design
quality, result direction, red flags and falsifiable predictions; then an
independent sandbox (this folder) re-implementing the sharpest positive claim from
its written specification and testing it against matched comparators.

The question asked of every project was the same: **does a result isolate an
ISRM-specific mechanism (the three-term utility U = alpha E - beta dS - gamma dC,
its threshold policy, or the decision-debt term) with a positive, held-out effect
whose confidence interval excludes zero, against a comparator given the same
information and budget?**

## 1. Headline

1. **No project on the drive currently supplies that result.** The best-designed
   studies (preregistered, matched comparators, held-out seeds or flights,
   bootstrap intervals, replayable) report null or negative verdicts for the
   ISRM-specific terms, and the author's own documents say so.
2. **Every positive number is attached to an arm that is not the three-term ISRM.**
   The label drifted: in the expert-switching studies "ISRM" is a CUSUM change
   detector plus argmin serving; in the Isaac dispatch studies it is a net-benefit
   rule algebraically identical to the conventional twin; in the GPU work it is a
   static power cap; in AgentDojo it is a lexical scope filter; in the coupled
   planning study it is a quadratic ridge regressor.
3. **The sandbox replicated the sharpest positive claim and found its cause.** The
   expert-switching long-run gain (+0.36 to +0.40 pp in the original) reproduced in
   an independent world at **+0.42 pp [+0.34, +0.50], 12 of 12 worlds**, against the
   original's cost-selected comparator. Against a budget-matched scanner from a fine
   grid the event controller is **-0.04 pp [-0.07, +0.00]**, and applying the
   original's selection rule to the event family instead of the adaptive family
   reverses the sign to **-0.34 pp [-0.43, -0.24]**. The gain belongs to the
   selection rule (the comparator was allowed a 0.5 pp development deficit), not to
   the detector. Details in section 4.
4. **The decision-debt prediction is real but neither as large as printed nor
   unique.** Re-implemented from the manuscript's equations with its printed
   parameters, the debt agent escalates **2.9 trials** earlier after deferred
   preparation [+2.5, +3.3] (**7.0 trials** [+6.4, +7.6] under the original
   simulation's protocol with debt frozen during the test), against the printed
   12.7. The printed number is reached only with lambda_eps = 1.0, which is the
   value the original code used, not the printed 0.5. Two conventional controllers
   with no debt term that merely carry state across the context reset, a
   hazard-learning filter and a leaky accumulator, are also history dependent
   (+1.2 and +3.8 trials). The matched comparator's "exactly zero" is guaranteed by
   construction. As written, the discriminating experiment does not discriminate
   decision debt from ordinary state-carrying adaptive control (section 4).
5. **The three remaining synthetic claims replicate and each is attributed (T03, T08,
   T04, run 2026-11-06 evening).** The metacognition ordering U0 < FIXED_FAIR < MPC
   reproduces with a larger deficit (**U0 -0.33 J below the fixed schedule, 0 of 64
   blocks**) and the receding-horizon deferral mechanism is visible tick by tick; U0 and
   its weighted-sum twin never disagree. The "dual weights" controller is the textbook
   drift-plus-penalty rule (**0 differing actions in 192,000 ticks**), no constant triple
   matches it and the zero-reserve trap reproduces. The Total ISRM v3 gain reproduces
   (**+1.33 s** against adaptive energy-only, original +1.35) but a dwell-and-margin
   comparator does not close it, and the ablations show the gain is a brake on a harmful
   switching architecture (no-switching +4.7 s and the fixed predictor +6.4 s over ISRM).
   Details in section 4.

## 2. Ranking

Two scores per project. **Design** is methodological quality (0-5: preregistered,
matched comparators, held-out, CIs, reproducible). **For ISRM** is how much the
project's evidence supports an ISRM-specific mechanism (0-5). Numbers are the
authors' own, quoted from their reports.

| # | Project | What "ISRM" is in the tested arm | Evidence | Design | Result for the ISRM-specific term | For ISRM | Sandbox |
|---|---|---|---|---|---|---|---|
| 1 | Multi ISRM / isrm_metacognition_benchmark | U0 three-term utility; U_PLUS extension | synthetic, preregistered hash, N=160 blocks, 7 comparators | 4 | U0 - MPC **-0.214 [-0.234, -0.193]**, below FIXED_FAIR by -0.063; U_PLUS - MPC +0.003 [-0.002, +0.009]; C adds no information (coefficient collapsed to 0) | 0 (confirmed by T03: -0.33 below the schedule) | **T03 run** |
| 2 | ISRM Test | U over budget forecast, dEntropy, dProbe-loss | synthetic, frozen contract, 8 paired seeds, ablations | 4 | **-11.96 jobs [-13.52, -10.56]** vs error-triggered PCGrad; coherence ablation inert (+0.08 [-0.56, +0.50]); never chooses Inspect by construction | 0 | none needed |
| 3 | Total ISRM / benchmarks (v2, v3) | three-term utility with model switching | synthetic arena, frozen protocol, 64-1,024 paired seeds, Holm | 4 | v2: 0 wins / 72 cells, energy -17.43 [-18.15, -16.72]; v3: **+1.347 s [+1.140, +1.552]** vs adaptive energy-only but -7.07 s vs fixed; ISRM switched 1.2x vs 13.5x per run (dwell suspected) | 1 (unchanged after T04: the gain is a brake on a harmful switching architecture) | **T04 run** |
| 4 | Flock ISRM (v3 Hybrid; PE v1/v2 Nagy 2010; PE Nagy 2013) | v3: threshold gate on E/dS/dC (U telemetry only); PE: CUSUM-triggered basin switching | real pigeon flocks, frozen hashed protocols, grouped folds, cluster-robust CIs | 3.5 | tuned Hybrid inert (occupancy 0.0), defaults harm +0.315 [+0.070, +0.560]; ISRM-PE vs continuous comparator **-0.069 [-0.099, -0.040]** (dev) and **-0.091 [-0.120, -0.062]** (clean held-out, 3 flocks); ISRM-PE = substrate (|d| <= 0.004) | 0 | T05 (spec, data hashed) |
| 5 | VA / tfm_isrm_experiment | P5 forecast gate with uncertainty, disagreement and pressure terms | real 1-minute perp panel (proprietary), preregistered, matched 9-cell tuning | 4 | P5 - P4 daily **+0.002 [-0.025, +0.037]**, p 0.47; pressure term tuned to 0, forced lambda=1 costs -0.18; persistence beats both | 0.5 | T06 (spec, synthetic data) |
| 6 | NeuralNets / isrm_lab | native three-term U arbitrating a multi-exit net | real Fashion-MNIST, frozen protocol, attribution control, NVML energy | 4 | NO_ADVANTAGE_DEMONSTRATED; attribution contrast +0.024 [+0.004, +0.044] but sign-flip p 0.125 (5 seeds), **+1.84 RU per success worse**, +2.85 J worse | 1 | T09 (spec, needs GPU) |
| 7 | Virtual Astrocytes (LIMITS, GAINS, LONGITUDINAL, EQUAL_TUNING, REGIME_MAP) | **CUSUM + argmin event controller; no E/S/C term** | synthetic cached streams, frozen contracts, 12 held-out seeds, simultaneous CIs | 3 | +0.36 to +0.40 pp vs cost-selected adaptive at p320/d2; -0.03 [-0.07, +0.01] vs budget-selected; -0.17 vs accuracy-first; **-1.64 [-2.28, -1.00]** at delay 8 | 1 -> **0 after T02** | **T02 run** |
| 8 | Multi ISRM dispatch studies (scaling, supervision, coupled planning, latency, intervention value) | net-benefit rule = CONVENTIONAL_NET twin; quadratic ridge | synthetic + Isaac timing noise, registered, Bonferroni | 2-3.5 | scaling 13.7% gain at B=256 shrinks to 5.0% under fixed costs, reverses in OVERLOAD; supervision +0.038 [+0.021, +0.055] net (1.3%); coupled planning **loses to linear ridge** +0.082 [+0.029, +0.137]; 0 prediction calls by any adaptive arm | 0 | none |
| 9 | Tokenless / AgentDojo | restriction-only scope filter | external benchmark, Qwen3-8B, post-hoc | 2 | injection success 15.3% -> 6.9%, 8.3 pts, CI [0, 17.5], p 0.125; no utility, no threshold policy | 0.5 | none |
| 10 | Blackwell, ISRM.DVFS, ISRM.GPU.Render, ISRM.Hypertuner, ISRM.GPU | static 250 W / 80% power cap; PI on a logged distance; degenerate gradient | real GPU telemetry, no preregistration, n=2-3 | 1-2 | Blackwell -11.5% / -21.3% J/token from a fixed cap (deployed policy had U = E, dS = dC = 0); DVFS perf/W -3.8%; Hypertuner weights collapse to bounds regardless of telemetry | 0 | formalism check |
| 11 | PID+ISRM, isrm_ardupilot_phase_compare | three-term sidecar over PID telemetry | ArduPilot SITL, n=3 vs 3, synthetic power, no matched gate | 1 | recovery alarm burden 19-32% vs 63-66%, but 100% LOITER in recovery and mission never resumed; energy-deficit throttle integral not reduced | 1 | T07 (spec) |
| 12 | ISRM Dual Weights | virtual-queue weights (drift-plus-penalty) | toy plant, protocol before run, seeds disjoint (v3) | 2 | no fixed triple matched within tolerance (worst mismatch 2.74); absorbing zero-reserve trap; NO_C ablation near-tautological | 1 -> **0 after T08** | **T08 run** |
| 13 | isrm_tahoe_demo_v3 (Tahoe-100M, PRISM) | three heuristic axes with hard thresholds | real data, frozen PRISM protocol | 2 | PRISM macro MAE +0.067% worse with ISRM features; all real-data labels abstain; dated subfolders are copies of Virtual Astrocytes sources | 0.5 | none |
| 14 | VA / rv_challenge | dS pressure composite; LightGBM + hysteresis rule | real data, **externally scored** by the counterparty | 3 | submission +0.238 vs persistence +0.663 at 5 bps; dS adoption criteria failed (lower bound -0.05); all 30 switching cells negative | 0.5 | none |
| 15 | ISRM.OS, Agentic ISRM, ISRM Agent, ISRM C Drive, ISRM Labs AB folder, AgentDojo runtime | schemas, replay kernels, hash evidence, k-selection with a k=4 prior | engineering only | n/a | no empirical ISRM claim (ISRM Agent's k-selection is circular by construction) | 0 | none |
| 16 | Equations (formal core, Sep 2026) | canonical three-mode regulator, product drive theorem, stochastic bounds, Lean core | hand proofs reviewed by AI agents, exact rational verifiers that fail closed, 13 elementary Lean theorems kernel-checked | 3 | exact and self-repaired, but about a hand-chosen certificate point of a toy automaton; the universal claims (impossible triad, cutoff theorem, drive implies stability) are refuted or withdrawn by the author; no empirical reach | 0 | formalism checks |
| 17 | Equation (ISRM_2.0 paper, Oct 2026) | same regulator plus Layer 4/5 argmax policies | hand proofs + synthetic simulation, scripts with no assertions | 2 | the Oct-21 text still prints theorems the author's own Sep-4 audit refuted; stress benchmark irreproducible (39 vs 67 jumps); Layer-4 ISRM argmax ties ALWAYS (+3.9 +/- 9.5) and loses under failure costs; the winner is a standard Bayesian audit policy | 0 | none |
| 18 | Equations real-data studies (UCI HAR, PenDigits, digits, Agent_Sim) | product-of-probabilities routing rules | real public data, frozen protocols, alias controls | 2-3 | HAR: FAILED_DESIGN gate, test never opened; PenDigits: +0.020 utility from a calibration tie-order artefact at 2.4x work; digits: NONDISCRIMINATING; Agent_Sim: PRIMARY_CRITERIA_NOT_MET (+0.0036 [-0.0007, +0.0085]) | 0 | none |
| 19 | ISRM_manuscript1.md (decision-debt controller, section 5.1) | escalation rule with lambda_eps x debt; matched twin lambda_eps = 0 | simulation of its own equations; code outside the research tree; 101/400 censored participants unreported | 1 | 12.7 trials reproduces only at lambda_eps = 1.0 (text says 0.5); matched zero by construction; the author's own uncited follow-up concedes debt is "an ordinary piece of controller state" | 1 -> **0.5 after T01** | **T01, T01b run** |

Reading the table: the projects that are most convincing *as science* are rows 1 to
6 and 16, and each is convincing precisely because it shows that the ISRM-specific
term did not help under fair conditions, or that a theorem holds only for a toy
point. The projects that *report gains* (rows 7 to 11) do so for arms whose ISRM
content is nominal. The one distinctive theoretical prediction (row 19) survives
the sandbox only in reduced, non-unique form.

## 3. The label-drift finding

| Study with a positive number | Arm called "ISRM" | What the arm actually computes | Conventional twin that matches it |
|---|---|---|---|
| Virtual Astrocytes LIMITS / GAINS / LONGITUDINAL | ISRM-style event regulator | discounted error observer, argmin serving, per-expert one-sided CUSUM (theta 0.5, gamma 0.8) | EWMA scanner (GAINS: -0.058 pp, CPU within 7%) ; budget-matched EWMA (sandbox T02: -0.04 pp) |
| Multi ISRM scaling / supervision | ISRM net-benefit supervisor | benefit - price x work_ms | CONVENTIONAL_NET: 14,592 / 14,592 and 49,152 / 49,152 identical decisions |
| Multi ISRM coupled planning | ISRM gain predictor | 595-term quadratic ridge | 34-term linear ridge wins (+0.082 [+0.029, +0.137]) |
| Multi ISRM active monitoring follow-up | U0 with new S/C | warning-driven selector | INNOVATION selector: +0.0000037 [-0.0002, +0.0002]; oracle Bayes beats both |
| Blackwell policy v1 | ISRM regulator | U = E, dS = dC = 0 in all six epochs | fixed 250 W cap |
| Tokenless AgentDojo | ISRM governance | trusted-prompt lexical scope closure | identity restriction + AgentDojo's stock tool_filter (untested) |
| Total ISRM v3 | three-term switching | utility-ranked model operations with 1.5 s dwell and margin | dwell-and-margin energy controller does **not** match it (sandbox T04: ISRM +1.4 s); the gain is fewer paid model operations in an arena where every model operation hurts (no-switching ablation +4.7 s, fixed predictor +6.4 s over ISRM) |

## 4. Sandbox results

### T02: expert-switching long-run gain

Spec `specs/T02_expert_switching_longrun_gain.json` (hash 5fd8823c3bc5), frozen
before any run. World: `isrm_sandbox/environments/frozen_bank.py`, written from the
audit's description, not from the original code: 8 regimes, 8 experts each trained
on one regime's nonlinear rule, hidden Markov regime process with mean persistence
320 batches, labels released 2 batches late, 4,096 batches, 8 inputs per batch.
Controllers: `CUSUMEventPolicy` and `EWMAScanPolicy` re-implemented from the written
rules; comparator selection on 4 development worlds disjoint from the 12
evaluation worlds; every arm sees the same evaluation worlds. Ledger:
`reports/ledger.jsonl` (chain intact).

| Contrast (percentage points of whole-stream accuracy) | Mean | 95% CI | Reading |
|---|---|---|---|
| event_fixed - adaptive_cost_selected (**primary**, the original comparison) | **+0.423** | [+0.345, +0.499], 12/0/0 | replicates the original +0.36 to +0.40 |
| event_fixed - adaptive_budget_fine (same work, fine grid) | -0.036 | [-0.075, +0.003] | no advantage at matched work |
| event_fixed - adaptive_budget_selected (same work, original grid) | -0.036 | [-0.076, +0.003] | same |
| event_fixed - adaptive_accuracy_first (8x work) | -0.161 | [-0.217, -0.114] | dense scanning wins, as in the original |
| event_cost_selected_sym - adaptive_accuracy_first (**selection symmetry**) | **-0.338** | [-0.428, -0.243] | the sign follows the rule, not the detector |
| event_fixed - periodic_budget_matched | +2.68 | [+2.20, +3.17] | event triggering beats a periodic schedule (replicates the robust conventional finding) |
| oracle - event_fixed | -0.147 | [-0.335, +0.024] | the event controller's extra scans repair noisy receipts |

Work per input: event_fixed 1.093, adaptive_cost_selected 1.026, adaptive_budget_fine
1.064, periodic 1.109. The cost-selected comparator chose the quietest cell in the
grid (margin 0.1, max gap 4,096: 15 scans in 4,096 batches) because the rule allowed
it to lose 0.5 pp; the gain is the deficit the rule permitted.

Verdict recorded: `SUPPORTED_IN_SANDBOX` on the primary contrast as frozen, with the
interpretation key in the spec notes satisfied in the direction of H0. In plain
terms: the original number is real and reproducible, and it measures the comparator
selection rule. The event controller contains no ISRM term to credit.

### T01 and T01b: decision-debt history dependence

Specs `T01_decision_debt_history_dependence` (hash 41c9533b2a1c) and
`T01b_decision_debt_frozen_test_protocol` (hash de3bcfcd79b0), both frozen before
running. World `isrm_sandbox/environments/changepoint_task.py` and controller
`isrm_sandbox/controllers/isrm_debt.py`, written from manuscript sections 2.2 and
10.7: change-point filter (hazard 0.05, accuracy 0.62), escalation when
dL + lambda_eps x debt exceeds the broadcast cost (6.0 at full capacity), Stage 2
arbitration with a capacity veto (CHECK needs 0.25, REVISE needs 0.55), the
non-negative debt equation with settlement on verified revision, six preparation
episodes at capacity 1.0 (resolved) or 0.35 (deferred), then a test episode at
capacity 1.0 after a belief reset. 400 simulated participants; both conditions and
all arms share each participant's streams. T01 lets the debt equation run during
the test as the manuscript's equations state; T01b freezes debt during the test as
the original simulation did.

| Arm (history effect in trials, resolved minus deferred escalation) | T01, debt evolves in test | T01b, debt frozen in test |
|---|---|---|
| isrm_debt, lambda_eps = 0.5 as printed (**primary**) | **+2.87 [+2.47, +3.31]**, 347/0/53 | **+6.98 [+6.43, +7.57]**, 395/0/5 |
| matched_no_debt, lambda_eps = 0 | 0 exactly (by construction) | 0 exactly |
| isrm_debt, lambda_eps = 1.0 (the original code's value) | +6.12 | **+11.20 [+10.45, +11.98]** |
| leaky_accumulator_no_debt (debt without settlement, no lambda_eps) | +1.72 [+1.37, +2.13] | +3.76 [+3.32, +4.24] |
| adaptive_hazard_no_debt (hazard learned from surprise, carried across reset) | +1.19 [+1.00, +1.39] | +1.24 [+1.04, +1.48] |
| isrm_debt, rho = 0.05 (manuscript: about 19) | +7.27 | not run |
| isrm_debt, rho = 0.80 (manuscript: about 1.4) | 0 | not run |
| isrm_debt, two preparation episodes (manuscript: flat) | +2.79, difference to six episodes +0.09 [-0.02, +0.25] | not run |

Verdicts recorded: `POSITIVE_BELOW_PRACTICAL_THRESHOLD` for both (thresholds 5 and
10 trials, chosen from the printed 12.7 with the auditor's 8.3 in view). Readings:

- The history effect exists and has the predicted shape (larger at slower debt
  decay, absent at fast decay, flat in the number of deferred episodes).
- Its size at the printed parameters is a quarter (T01) to a half (T01b) of the
  printed 12.7; the printed size needs lambda_eps = 1.0, which the original code
  used while section 10.7 says 0.5. The theory audit reached the same conclusion by
  re-deriving the original's 1,600 trial counts exactly at 1.0 and only 1,246 of them
  at 0.5.
- The effect is not unique to decision debt. A leaky accumulator with no settlement
  clause produces more than half of it, and a filter that merely learns the hazard
  from surprise produces a third of it, because both carry state across the "context
  reset" that the manuscript assumes leaves an adaptive controller in an identical
  state. The matched comparator cannot differ because the test stream is identical
  and nothing it carries is read at test; its zero is an implementation check, as the
  manuscript itself notes. The author's own later mechanism study (not cited in any
  manuscript version) found the same thing and describes debt as "an ordinary piece
  of controller state".
- Magnitudes depend on choices the manuscript leaves open (one-sided versus
  two-sided change-point filter, whether dL includes the revision cost): this
  sandbox follows the text; the original code differs on both. The ordering of arms
  is the same under either protocol.

For the human experiment this means: a positive history effect would not by itself
support decision debt over any controller with a slowly decaying state variable;
the design needs a comparator that carries state, and an independent estimate of
the decay rate, which the manuscript already asks for.

### T03: shared-budget metacognition, the ordering U0 < FIXED_FAIR < MPC and the twin identity

Spec `T03_metacognition_u0_ordering` (hash 0522e9c1712d), frozen before running. World
`isrm_sandbox/environments/metacog_budget.py` and controllers
`isrm_sandbox/controllers/metacog_policies.py`, written from the original's mathematical
contract: six services with AR(1) latents (a = 0.97), scalar Kalman observation, a public
DAG with signed coefficients and coupling 0.35, deadlines with periods {8, 12, 16} and
values {1, 3, 9}, computation lengths {1, 2, 3}, budget B0 = 6, B_max = 12, refill 1.5
(scarce) or 6 (ample), the S score (belief variance under each artifact's consumed
evidence) and the C score (parent-revision incompatibility), and one shared expected-value
forecaster with horizon 4 that every planner uses. Calibration on 12 development blocks
only, with equal budget per family (every configuration scored on the same paired dev
episodes): U0 on the 28-point coefficient simplex, MPC on five lambda_B values, U_PLUS on
40 (alpha, beta, gamma) triples whose beta = gamma = 0 slice is the MPC grid. 64 fresh
blocks, each run in the four scarce-coupled cells ({stable, changing} x {linear,
nonlinear}) with J averaged per block, plus the ample-coupled stable linear cell as an
extra. 25 minutes; ledger chain intact.

Calibration outcome (the ledger's selection field): the development data chose alpha = 1/6,
beta = 5/6, gamma = 0 for U0, the same point the original's calibration chose, with the
whole U0 family between J 0.00 and 0.12 on dev; MPC chose lambda_B = 0, and U_PLUS chose
(0, 0, 0), the pure value planner, over every configuration with a positive S or C weight;
the nonlinear tolerance ratio came out at 0.979 (original 0.9795) and the P(correct)
recalibration slope at 0.89 (original 0.81).

| Contrast (J on the scarce-coupled cells, n = 64 paired blocks) | Sandbox | Original (N = 160 blocks) |
|---|---|---|
| ISRM_U0 - FIXED_FAIR (**primary**) | **-0.326 [-0.347, -0.304]**, U0 lower in 64/64 blocks | -0.063 [-0.085, -0.043] |
| ISRM_U0 - CONVENTIONAL_MPC | **-0.403 [-0.422, -0.383]** | -0.214 [-0.234, -0.193] |
| U_PLUS - MPC | **0.000 exactly** (calibration set beta = gamma = 0 and the same lambda_B) | +0.003 [-0.002, +0.009] (calibrated alpha 2, beta 4, gamma 0) |
| ISRM_U0 vs weighted-sum twin | **0 decision mismatches in 16,384 decisions** | 0 in 2,560 episodes |
| ISRM_U0 - HOLD_ALL | +0.091 [+0.073, +0.111] | HOLD_ALL = 0 |
| mean J: U0 / FIXED_FAIR / MPC | 0.091 / 0.417 / 0.494 | 0.346 / 0.409 / 0.560 |
| missing-answer rate: U0 / MPC / FIXED_FAIR | 0.64 / 0.09 / 0.002 | 0.31 / 0.02 / - |
| ample minus scarce J: U0 / MPC / FIXED_FAIR | -0.064 / +0.028 / +0.162 | difference in differences +0.095 [0.065, 0.126]: U0 worse when the budget is ample |

Verdict recorded: `HARMFUL_IN_TESTED_RANGE`. Readings:

- The ordering reproduces and the U0 deficit is larger than the original's: U0 scores 0.33 J
  below a fixed cyclic schedule and 0.40 below the conventional planner, in every one of 64
  blocks. The sandbox's fixed schedule serves tasks in cyclic order with as many unit
  operations per tick as the budget affords; it reaches 0.42 J where the original's reached
  0.41.
- The mechanism is the one the original diagnosed, and the planner traces show it tick by
  tick: with no value or deadline term, U0 prefers "observe now, compute later" over
  "compute now" whenever the budget forecast saturates at B_max, because one more
  observation lowers the snapshot variance of a future artifact while the saturated E term
  makes the extra credit free. It defers publication tick after tick (missing-answer rate
  0.64) and does worse when the budget is ample (-0.064 J from scarce to ample, while every
  other controller gains). In this world the deferral is nearly total because publishing a
  never-observed task cannot lower S at all (its snapshot variance is the prior), so nothing
  in U0 ever rewards getting an answer out before a deadline.
- U0 is decision-identical to the weighted-sum twin on every tick, as the algebra requires
  (contract section 7.2; `tests/test_formalism.py`): the ISRM utility makes no decision a
  weighted sum of forecast features would not.
- U_PLUS adds nothing to MPC here: given the same forecaster and the same lambda_B grid,
  the development data preferred beta = gamma = 0, so the extended utility collapsed onto
  the conventional planner and the equivalence holds trivially. Where the original found a
  +0.003 edge for (alpha 2, beta 4), this world finds no positive S or C weight that beats
  the pure value planner on development blocks.

### T08: dual weights, the matching problem and the absorbing trap

Spec `T08_dual_weights_matching_and_trap` (hash 48ad448b7446), frozen before running.
Plant `isrm_sandbox/environments/reserve_supply_plant.py` and controllers
`isrm_sandbox/controllers/dual_weights.py`, written from the V3 protocol and reports:
reserve R with 0.10 supply per tick and 40-tick deficits every 400 +/- 40 ticks, effort
cost 0.15 kappa with kappa jumping to 0.6 or 1.6 (R2: 150-250 ticks after a deficit
onset; R3: at the onset), reserve-funded Sample (0.01) and Update (0.10, an oracle refit),
effort curtailment, meters E = 0.30 - R, staleness S (threshold 0.20) and estimated
consistency C (threshold 0.10). ADAPTIVE: Q <- max(0, Q + r) on the signed realised
residuals, action = argmin[-work + sum_i Q_i rhat_i]. FIXED: constant (alpha, beta, gamma)
fitted to ADAPTIVE's R2 violation depths on four training worlds by the protocol's bounded
search (1,000 coarse triples plus two refinement rounds, 1,495 distinct candidates).
Sixteen fresh paired worlds, each run in R1, R2 and R3 by every arm; metric = R2 work per
tick (lower is better). Ledger chain intact.

| Quantity (full run, n = 16 paired worlds) | Sandbox | Original (V3_REPORT.md) |
|---|---|---|
| FIXED - ADAPTIVE work, R2 (**primary**) | **+0.294 [+0.259, +0.330]**, ADAPTIVE lower in 16/16 worlds | +0.295 [0.257, 0.334] |
| FIXED - ADAPTIVE work, R3 | +0.280 | +0.316 [0.273, 0.350] |
| joint match of FIXED to ADAPTIVE's R2 depths | **none**: worst normalised mismatch 4.39 on the training worlds (0 of 1,495 candidates within tolerance), 8.2 held out | none: 2.741 (0 of 1,611), 5.74 held out |
| ADAPTIVE vs textbook drift-plus-penalty twin (independent implementation) | **0 differing actions in 192,000 ticks** | expected identical (formalism test) |
| NO_C - ADAPTIVE work, R2 | -0.000 [-0.007, +0.006]; C_true 0.464 vs 0.111 | +0.018; C_true +0.290 [0.231, 0.340] |
| R2 depths E / S / C, ADAPTIVE vs FIXED | 0.086 / 0.218 / 0.143 vs 0.109 / 0.305 / 0.110 | 0.066 / 0.171 / 0.018 vs 0.056 / 0.220 / 0.028 |
| R1 absorbing state (R = 0, zero stored supply estimate, supply positive) | 86% of post-deficit positive-supply ticks (ADAPTIVE), 100% (FIXED) | 3,062 such ticks in the saved trace (about 90% after supply resumed) |

Verdict recorded: `SUPPORTED_IN_SANDBOX` on the primary contrast as frozen (ADAPTIVE does
at least 0.03 less work per tick than FIXED). The spec's own reading key says that verdict
is meaningful only if a joint match within tolerance exists, and none does: FIXED sits at a
different operating point (deeper E and S violations, shallower C), so the work difference
is a trade-off, not a saving at equal constraint pressure, which is what the original
concluded. Readings:

- The "dual weights" controller is the textbook drift-plus-penalty rule. A Lyapunov
  virtual-queue implementation written separately in the Neely form (minimise V x penalty
  plus queue-weighted net arrivals) chose the same action as ADAPTIVE on every one of the
  192,000 evaluated ticks. The method contains no ISRM term; the label is the only novelty.
- The absorbing zero-reserve trap reproduces: with reserve-funded actions, once R reaches 0
  while the stored supply estimate is 0, every feasible action predicts the same constraint
  terms and the work reward keeps selecting Act1, which the plant curtails to consume all
  incoming supply; sensing never resumes. ADAPTIVE spends 86% of its post-deficit
  positive-supply ticks in R1 there. The oversampling incentive is also present: Sample on
  57% of R2 ticks.
- Removing the consistency queue changes work by nothing measurable and quadruples the true
  model error (0.111 to 0.464) because Update is the only route to the oracle refit: the
  near-tautology the ranking already noted.

### T04: Total ISRM v3, is the switching gain switching hygiene?

Spec `T04_total_isrm_dwell_attribution` (hash d6aad761dd5e), frozen before running.
Arena `isrm_sandbox/environments/ball_arena.py` and governors
`isrm_sandbox/controllers/arena_governors.py`, re-created from the v3 benchmark's
documented rules (circular arena, 60 Hz, paid 0.012 observations, 25 candidate commands
ranked by forecast energy, an inertial and a learned turn-diffusion predictor, paid
assessments every 0.5 s, Switch 0.025, Recalibrate 0.08 with a 0.3 s hold, 1.5 s dwell,
hysteresis 0.0005, frozen weights 0.4 / 0.35 / 0.25). Nothing is tuned. Replicate = one
fresh world whose cell is drawn uniformly from the 72-cell stationary grid and whose whole
hazard trajectory is drawn once, so every arm faces identical balls. The decisive new arm
is the dwell-only comparator: the adaptive energy-only governor plus a switching margin
equal to ISRM's constant coherence-identity penalty (0.25 x 0.1 in U units, 0.0625 in
energy-value units) on every model-identity change; dwell and hysteresis are already part
of every adaptive governor, as in the original. 128 paired worlds, 22 minutes.

| Contrast (survival seconds, 120 s cap, n = 128 paired worlds) | Sandbox | Original (v3, 72 cells x 64 seeds) |
|---|---|---|
| isrm_full - adaptive_energy_dwell (**primary**) | **+1.43 [+0.07, +2.81]**, W/L/T 41/28/59, sign-flip p 0.043 | not run |
| isrm_full - adaptive_energy_only | **+1.33 [+0.03, +2.65]** | +1.347 [+1.140, +1.552] |
| isrm_full - fixed_energy_only | **-6.37 [-8.40, -4.59]** | -7.073 [-7.308, -6.838] |
| isrm_no_switching - isrm_full | **+4.70 [+3.00, +6.64]** | disabling switching: +4.08 energy at the default, +21.2 s at the harsh cycle anchor |
| mean survival: isrm / dwell / adaptive / fixed / no-switch | 76.0 / 74.6 / 74.7 / 82.4 / 80.7 | 72.1 / - / 70.8 / 79.2 / - |
| terminal energy: isrm / dwell / adaptive / fixed / no-switch | 14.0 / 13.3 / 13.5 / 22.0 / 19.7 | ISRM +0.61 vs adaptive, -5.42 vs fixed |
| model-family switches per run: isrm / dwell / adaptive | 1.14 / 0.83 / 8.84 | 1.22 / - / 13.50 |
| paid recalibrations per run: isrm / dwell / adaptive | 1.75 / 3.07 / 1.77 | 5.07 / - / 0.10 |

Verdict recorded: `POSITIVE_BELOW_PRACTICAL_THRESHOLD` (threshold 0.5 s; the lower bound
+0.07 clears zero, not 0.5; 59 of 128 worlds are ties, mostly both agents alive at 120 s).
Readings:

- The original's numbers reproduce in an independently written arena: +1.33 s against the
  plain adaptive energy-only governor (original +1.35), -6.4 s against the fixed predictor
  (original -7.1), survival levels within a few seconds of the original's, and the same
  switching profile (about 1 switch per run for ISRM against 9 for the plain governor).
- The frozen H0 is not supported as stated: the dwell-and-margin comparator reproduces
  ISRM's switch suppression (0.83 switches per run) but recalibrates almost twice as often
  (3.07 against 1.75) and does not close the gap (+1.43 against it, no smaller than +1.33
  against the plain governor). ISRM's margin on identity changes is not where its seconds
  come from.
- Where they do come from: every model operation is harmful in this arena. The ISRM governor
  forbidden to switch beats the full governor by 4.7 s and the fixed predictor with no model
  bank beats it by 6.4 s while retaining 57% more energy. The three-term U is a brake on that
  harmful machinery: it performs the fewest paid model operations of the three adaptive
  governors (2.9 switches plus recalibrations per run against 3.9 and 10.6), and each
  recalibration it declines is a 0.3 s stop among moving balls that the energy term alone
  would have bought. The S and C terms therefore earn a real but small advantage over an
  equally capable energy optimiser by doing less of what the architecture does, which is the
  original report's own conclusion ("the original simpler controller remains the stronger
  overall reference"). It is not evidence that uncertainty or coherence measurement improves
  control over the conventional baseline.

Formalism checks (`tests/test_formalism.py`, all passing, no simulation):

- U0 equals a weighted sum of forecasts plus a candidate-independent constant, so its
  argmax is identical to the linear twin's (0 mismatches in 2,000 random candidate
  sets). The ISRM utility cannot make a decision a weighted sum would not.
- With E_after = E_now - cost, the budget level E_now never changes a decision; only a
  saturating clip of E can, and then only when the clip binds differently across
  candidates. The energy channel as written cannot pace spending.
- U_PLUS with beta = gamma = 0 is bit-identical to CONVENTIONAL_MPC.
- The Hypertuner's gradient on its penalty weights is never positive, so the weights
  fall to their bounds whatever the GPU does.
- The flock Hybrid's coherence gate saturates below its threshold at 60 Hz
  (dC < 0.55 for any heading dynamics within the motion limits): "dC never fired" is
  arithmetic.
- The Dual Weights rule picks the same action as textbook drift-plus-penalty on
  1,000 random instances.

## 5. Test plan: frozen specs

All specs are content-hashed in `specs/`; `python -m isrm_sandbox list` shows which
have suites. Suggested order by cost and information value.

| Spec | Claim under test | Status | Cost | Why it matters |
|---|---|---|---|---|
| T02 | expert-switching gain | **run, full** | 1 min | done; see section 4 |
| T01 | decision-debt history dependence, equations as written | **run, full** | 10 s | done; see section 4 |
| T01b | decision debt under the original frozen-debt protocol | **run, full** | 10 s | done; see section 4 |
| T03 | U0 < FIXED_FAIR < MPC ~ U_PLUS; U0 = twin; procrastination under ample refill | **run, full** | 25 min | done; see section 4: ordering reproduced, U0 -0.33 J below the schedule in 64/64 blocks, twin identical, deferral mechanism traced |
| T08 | dual weights: matching impossible, drift-plus-penalty identity, absorbing trap | **run, full** | 1 min | done; see section 4: no match, 0 twin mismatches, trap reproduced |
| T04 | Total ISRM v3 gain vs dwell-only comparator | **run, full** | 22 min | done; see section 4: +1.3 s reproduced, dwell comparator does not close it, every model operation harmful |
| T06 | P5 vs P4 forecast gate on synthetic basis/carry | spec only | medium | pressure-term ablation on public or synthetic data |
| T07 | ISRM sidecar vs PID-gated hold on an energy-limited plant | spec only | medium | mission completion scored, which the original never did |
| T05 | CONT vs ISRM-PE on Nagy 2013 (hashed local data) | spec only | high (replay engine) | real-data negative; data-only pre-checks are cheap |
| T09 | isrm_lab attribution contrast, >= 10 seeds | spec only | high (torch, GPU) | only real-data study with a positive attribution contrast |

Design rules every suite follows: no imports from the original projects (enforced),
no network (enforced), seeds derived from the spec hash, paired worlds across arms,
comparators selected on development worlds only, verdict recorded in the ledger and
never asserted by pytest.

## 6. Theory cluster

- **Formal core (D:/Equations).** The strongest formal item on the drive. The
  product-drive representation theorem is sound under its homogeneity axioms; the
  canonical point-ISS bound and the "Recalibrate never exits at D <= 0.01" result are
  proved and verified with exact rational arithmetic that fails closed; 1,349 tests
  run from a manifest. The Lean file is genuinely kernel-checked (13 theorems, standard
  axioms only, no sorry), but its content is elementary integer arithmetic about a toy
  gate benchmark; none of the hybrid dynamics, the drive theorem or the stochastic
  results is formalised. The author's own 2026-09-06 assessment refutes or withdraws
  the universal claims (impossible triad, Lagrangian equivalence of persistence, the
  cutoff theorem under a first-order-stochastic-dominance kernel) and states that
  threshold level sets cannot identify the Cobb-Douglas drive.
- **The October paper (D:/Equation).** Written six weeks after that assessment, it
  still prints the refuted triad, the Lagrangian sentence and the cutoff theorem as
  proved, reports a stress benchmark (39 jumps) that the current scripts do not
  reproduce (67 jumps), and ships four scripts with no assertions. Its Layer 4/5
  experiments contain no debt term: the ISRM argmax ties the always-audit policy and
  degrades under failure costs, while a standard Bayesian change-point audit policy
  wins by a wide margin.
- **The manuscript's decision-debt result.** The simulation code is not in the
  research tree; it lives in the manuscript's artifact store. The code ran with
  lambda_eps = 1.0 while section 10.7 prints 0.5; 101 of 400 resolved and 17 of 400
  deferred participants never escalated within the 40-trial cap and were coded 41,
  which the manuscript does not mention; the test episode freezes debt and selects
  no actions, so the forecasting step and bounded update that are described as
  "shared" never execute. The matched comparator differs by nothing that is read at
  test, so its zero is a construction. The author's later, uncited mechanism study
  compares debt with a leaky evidence accumulator, an error counter and a learned
  cost and concludes that debt is "an ordinary piece of controller state", which
  contradicts the manuscript's claim that a controller whose state is belief,
  constraints and costs "cannot represent that history". The sandbox result in
  section 4 agrees.
- **Real-data studies in D:/Equations.** UCI HAR (FAILED_DESIGN gate, the official
  test partition was never opened), PenDigits (the +0.020 utility gap is a calibration
  tie-order artefact and the ISRM rule used 2.4 times the work), sklearn digits
  (NONDISCRIMINATING, always-cheap wins) and Agent_Sim (PRIMARY_CRITERIA_NOT_MET).
  All are honestly reported; none supports an ISRM-specific mechanism.

## 7. How to read a sandbox verdict

`SUPPORTED_IN_SANDBOX` means the primary contrast's lower confidence bound cleared the
frozen threshold. It does not by itself mean the ISRM mechanism is responsible; that
is what the secondary contrasts (budget-matched comparator, symmetric selection,
ablation of the ISRM-specific term) are for, and the spec's notes say in advance how
to read them. A null result is kept in the ledger with the same standing as a
positive one.
