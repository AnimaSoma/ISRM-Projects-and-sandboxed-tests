"""Freeze the sandbox's test specifications.

Run from the sandbox root:  .venv\\Scripts\\python tools\\freeze_specs.py
Existing frozen files are never overwritten; change the id or add a revision
suffix to register a changed design.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from isrm_sandbox.spec import Arm, TestSpec  # noqa: E402

SPECS = ROOT / "specs"


def T01() -> TestSpec:
    return TestSpec(
        id="T01_decision_debt_history_dependence",
        title="Decision debt: history dependence at matched belief and capacity (manuscript section 5.1)",
        source_project="D:/ISRM_manuscript1.md sections 2.2, 5.1 and 10.7 (the simulation code is not on D:/; it lives in the manuscript's artifact store)",
        source_claim=(
            "With the parameters listed in Section 10.7 and 400 simulated participants per cell, the ISRM agent reaches "
            "escalation after 24.5 trials following resolved preparation and 11.7 after deferred preparation, a difference "
            "of 12.7 trials (d = 1.12). The matched controller shows a difference of exactly 0.0 trials, as it must. ... "
            "it is 19 trials at rho = 0.05 and 1.4 trials at rho = 0.80, and it is essentially flat in the number of "
            "deferred episodes once debt has reached equilibrium."
        ),
        hypothesis_h1=(
            "Re-implemented from the written equations with the printed parameters (lambda_eps 0.5, rho 0.10, hazard 0.05, "
            "accuracy 0.62, horizon 8, broadcast cost 6.0, capacities 1.0 / 0.35), the history effect (trials to escalation "
            "after resolved minus after deferred preparation) exceeds the matched controller's by at least 5 trials."
        ),
        hypothesis_h0=(
            "The effect is absent or small at the printed parameters; or conventional state-carrying controllers that "
            "contain no debt term (a hazard-learning filter; a leaky accumulator without settlement) produce a history effect "
            "of the same size, in which case the design does not discriminate decision debt from ordinary adaptive control."
        ),
        metric="history effect: trials to escalation after resolved preparation minus after deferred preparation (same test stream)",
        metric_higher_is_better=True,
        primary_contrast=["isrm_debt", "matched_no_debt"],
        direction="greater",
        threshold=5.0,
        equivalence_margin=2.0,
        alpha=0.05,
        n_replicates=400,
        quick_replicates=100,
        secondary_contrasts=[
            ["adaptive_hazard_no_debt", "matched_no_debt"],
            ["leaky_accumulator_no_debt", "matched_no_debt"],
            ["isrm_debt", "leaky_accumulator_no_debt"],
            ["isrm_debt_lambda1", "isrm_debt"],
            ["isrm_debt_rho005", "isrm_debt"],
            ["isrm_debt", "isrm_debt_rho080"],
            ["isrm_debt", "isrm_debt_2episodes"],
        ],
        arms=[
            Arm("isrm_debt", "treatment", "manuscript controller, lambda_eps = 0.5, rho = 0.10, six preparation episodes; debt equation runs on every trial including the test"),
            Arm("matched_no_debt", "comparator", "identical controller with lambda_eps = 0 (the manuscript's matched comparator)"),
            Arm("adaptive_hazard_no_debt", "control", "lambda_eps = 0; the filter's hazard is learned from how often the adopted model is surprised and persists across the context reset"),
            Arm("leaky_accumulator_no_debt", "control", "lambda_eps = 0; escalation test adds 0.5 x a leaky sum of max(0, q - 1/2) that is never settled by revision"),
            Arm("isrm_debt_lambda1", "control", "lambda_eps = 1.0, the value the original simulation code used"),
            Arm("isrm_debt_rho005", "control", "lambda_eps = 0.5, rho = 0.05 (manuscript predicts a larger effect, about 19 trials)"),
            Arm("isrm_debt_rho080", "control", "lambda_eps = 0.5, rho = 0.80 (manuscript predicts about 1.4 trials)"),
            Arm("isrm_debt_2episodes", "control", "lambda_eps = 0.5 with two preparation episodes instead of six (manuscript predicts no change once debt is at equilibrium)"),
        ],
        environment={
            "name": "changepoint_task",
            "task": {"accuracy": 0.62, "world_hazard": 0.0, "prep_episodes": 6, "prep_trials": 40, "test_trials": 40, "initial_rule": 1},
            "controller": {"hazard": 0.05, "horizon": 8, "u": 1.0, "c_broadcast": 6.0, "broadcast_slope": 0.5, "c_check": 0.35, "c_revise": 1.0, "c_hold": 0.25, "e_check": 0.25, "e_revise": 0.55, "settle_frac": 1.0},
            "capacities": {"resolved": 1.0, "deferred": 0.35},
        },
        notes=(
            "Replicate = simulated participant; both conditions and all arms share the participant's streams, so the matched "
            "arm's effect is exactly zero by construction (an implementation check, as the manuscript says). Interpretation "
            "key: the claim is distinctive only if isrm_debt beats matched_no_debt AND the two no-debt state-carrying controls "
            "do not. Magnitude context from the theory audit: the printed 12.7 was produced with lambda_eps = 1.0 and frozen "
            "debt during the test; at the printed lambda_eps = 0.5 the auditor's re-derivation gives about 8.3 trials. Record "
            "per-condition escalation trials, censoring at 41 and carried debt as extras."
        ),
    )


def T01b() -> TestSpec:
    return TestSpec(
        id="T01b_decision_debt_frozen_test_protocol",
        title="Decision debt under the original simulation protocol (debt frozen during the test): do the printed parameters give 12.7 trials?",
        source_project="D:/ISRM_manuscript1.md section 10.7 and the original sim_discriminating.py protocol as described by the theory audit",
        source_claim=(
            "With the parameters listed in Section 10.7 ... the ISRM agent reaches escalation after 24.5 trials following "
            "resolved preparation and 11.7 after deferred preparation, a difference of 12.7 trials (d = 1.12). Section 10.7: "
            "'Debt decays at rho = 0.10 and enters escalation with lambda_eps = 0.5 in the ISRM agent and lambda_eps = 0 in the "
            "matched controller, which is the only difference between them.'"
        ),
        hypothesis_h1="Under the original protocol (six wrong-from-the-start preparation episodes, test with belief 1/2, debt frozen during the test, 40-trial cap) and the printed lambda_eps = 0.5, the history effect exceeds the matched controller's by at least 10 trials.",
        hypothesis_h0="At lambda_eps = 0.5 the effect is about 8 trials (the printed number requires lambda_eps = 1.0, the value used in the original code); the leaky accumulator without settlement produces an effect of comparable size.",
        metric="history effect: trials to escalation after resolved preparation minus after deferred preparation (same test stream)",
        metric_higher_is_better=True,
        primary_contrast=["isrm_debt", "matched_no_debt"],
        direction="greater",
        threshold=10.0,
        equivalence_margin=2.0,
        alpha=0.05,
        n_replicates=400,
        quick_replicates=100,
        secondary_contrasts=[
            ["isrm_debt_lambda1", "matched_no_debt"],
            ["isrm_debt_lambda1", "isrm_debt"],
            ["leaky_accumulator_no_debt", "matched_no_debt"],
            ["adaptive_hazard_no_debt", "matched_no_debt"],
        ],
        arms=[
            Arm("isrm_debt", "treatment", "lambda_eps = 0.5 as printed; debt frozen during the test episode as in the original code"),
            Arm("matched_no_debt", "comparator", "lambda_eps = 0"),
            Arm("isrm_debt_lambda1", "control", "lambda_eps = 1.0, the value the original simulation code used"),
            Arm("leaky_accumulator_no_debt", "control", "lambda_eps = 0; leaky accumulator of consequential discrepancy without settlement"),
            Arm("adaptive_hazard_no_debt", "control", "lambda_eps = 0; hazard learned from surprise, carried across the reset"),
        ],
        environment={
            "name": "changepoint_task",
            "task": {"accuracy": 0.62, "world_hazard": 0.0, "prep_episodes": 6, "prep_trials": 40, "test_trials": 40, "initial_rule": 1},
            "controller": {"hazard": 0.05, "horizon": 8, "u": 1.0, "c_broadcast": 6.0, "broadcast_slope": 0.5, "c_check": 0.35, "c_revise": 1.0, "c_hold": 0.25, "e_check": 0.25, "e_revise": 0.55, "settle_frac": 1.0, "freeze_debt_in_test": True},
            "capacities": {"resolved": 1.0, "deferred": 0.35},
        },
        notes="Companion to T01: identical world and controllers, but debt is frozen during the test as in the original simulation, so this spec tests whether the printed number follows from the printed parameters. Censoring at trial 41 is recorded as an extra (the original had 101/400 resolved and 17/400 deferred censored, unreported).",
    )


def T02() -> TestSpec:
    return TestSpec(
        id="T02_expert_switching_longrun_gain",
        title="Expert switching: event controller long-run accuracy gain vs cost-selected adaptive scanner",
        source_project="D:/Virtual Astrocytes (ISRM_LIMITS_RESULTS.md, ISRM_GAINS.md, ISRM_REGIME_MAP_RESULTS.md)",
        source_claim=(
            "At persistence 320, delay 2, 4,096 batches, all four widths show positive whole-stream accuracy "
            "gains against the cost-selected adaptive controller: +0.358 to +0.400 pp (lower CI bounds +0.22 to "
            "+0.27) with +0.46% to +5.47% extra work. 'The long-run accuracy gain is real, with a measurable price.'"
        ),
        hypothesis_h1=(
            "In an independently generated frozen-bank world (8 regimes, persistence 320, delay 2, 4,096 batches), "
            "the fixed CUSUM event controller (theta 0.5, gamma 0.8) beats the cost-selected EWMA scanner by at least "
            "0.25 percentage points of whole-stream accuracy."
        ),
        hypothesis_h0=(
            "The gain is absent, or it is an artefact of the asymmetric selection rule: applying the same cost rule to "
            "the event family (symmetric arm) reverses the sign, and a fine-grid budget-matched EWMA scanner is within "
            "0.2 pp of the event controller."
        ),
        metric="whole-stream accuracy (percentage points)",
        metric_higher_is_better=True,
        primary_contrast=["event_fixed", "adaptive_cost_selected"],
        direction="greater",
        threshold=0.25,
        equivalence_margin=0.2,
        alpha=0.05,
        n_replicates=12,
        quick_replicates=6,
        secondary_contrasts=[
            ["event_fixed", "adaptive_budget_fine"],
            ["event_fixed", "adaptive_budget_selected"],
            ["event_fixed", "adaptive_accuracy_first"],
            ["event_cost_selected_sym", "adaptive_accuracy_first"],
            ["event_budget_selected_sym", "adaptive_accuracy_first"],
            ["event_fixed", "periodic_budget_matched"],
            ["oracle", "event_fixed"],
            ["dense_scan", "event_fixed"],
        ],
        arms=[
            Arm("event_fixed", "treatment", "CUSUM event controller, theta=0.5, gamma=0.8, refractory delay+1 (the original fixed configuration)"),
            Arm("adaptive_cost_selected", "comparator", "EWMA scanner from the 26-candidate grid, selected on 4 dev worlds by the original cost rule relative to event_fixed (allowed 0.5 pp deficit)"),
            Arm("adaptive_budget_selected", "comparator", "EWMA scanner from the 26-candidate grid with highest dev accuracy at work <= event_fixed"),
            Arm("adaptive_accuracy_first", "comparator", "EWMA scanner from the 26-candidate grid with highest dev accuracy"),
            Arm("adaptive_budget_fine", "comparator", "EWMA scanner from a 96-point fine grid with highest dev accuracy at work <= event_fixed"),
            Arm("event_cost_selected_sym", "control", "Event family (24 theta x gamma) selected by the same cost rule relative to adaptive_accuracy_first: the selection-symmetry control"),
            Arm("event_budget_selected_sym", "control", "Event family selected by highest dev accuracy at work <= adaptive_accuracy_first"),
            Arm("periodic_budget_matched", "comparator", "Periodic scan with the largest period whose dev work <= event_fixed"),
            Arm("dense_scan", "control", "Scan every batch: the accuracy ceiling for observer-based serving"),
            Arm("never_scan", "control", "Initial scan only: the persistence floor"),
            Arm("oracle", "control", "Scans exactly when a regime change becomes observable (uses the hidden schedule)"),
        ],
        environment={
            "name": "frozen_bank",
            "bank": {
                "n_regimes": 8,
                "dim": 8,
                "batch_size": 8,
                "persistence": 320,
                "delay": 2,
                "horizon": 4096,
                "nonlinearity": 0.4,
                "train_samples": 2048,
                "train_steps": 300,
                "lr": 0.5,
                "calib_samples": 4096,
            },
            "n_dev_worlds": 4,
            "families": {
                "event": "theta in {0, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2} x gamma in {0.8, 0.95, 0.99}",
                "adaptive": "gamma in {0.8,0.95,0.99} x (beta, margin, max_gap) in 8 cells, plus strong (0.8,0.95,0.03,64) and quiet (0.8,0.95,1,4096)",
                "fine": "beta in {0.8,0.95} x margin in {0,.01,.02,.03,.05,.1} x max_gap in {16,...,4096}",
                "periodic": "P in {2,3,4,6,8,12,16,24,32,48,64}",
            },
        },
        notes=(
            "Verdict is on the primary contrast only. The interpretation key is the pair of secondary contrasts "
            "(event_fixed - adaptive_budget_fine) and (event_cost_selected_sym - adaptive_accuracy_first): if the first "
            "is within +/-0.2 pp and the second is negative, the original gain is attributed to the selection rule, "
            "not to the detector. Work per input, final-quarter accuracy and scan counts are recorded as extras."
        ),
    )


def T03() -> TestSpec:
    return TestSpec(
        id="T03_metacognition_u0_ordering",
        title="Shared-budget metacognition: ordering U0 < FIXED_FAIR < MPC ~ U_PLUS and the U0 = weighted-sum identity",
        source_project="D:/Multi ISRM/isrm_metacognition_benchmark (VERDICT.md, MATHEMATICAL_CONTRACT.md)",
        source_claim=(
            "ISRM_U0 - CONVENTIONAL_MPC = -0.214 [-0.234, -0.193] (HARMFUL_IN_TESTED_RANGE); U0 is -0.063 "
            "[-0.085, -0.043] below the cyclic FIXED_FAIR schedule; ISRM_U_PLUS - MPC = +0.003 [-0.002, +0.009]; "
            "U0 and LINEAR_EQUIVALENT: 0 decision mismatches in 2,560 episodes."
        ),
        hypothesis_h1=(
            "In an independently built 6-service shared-budget simulator following the written contract (AR(1) "
            "latents, Kalman observation, DAG coupling, deadline-weighted J), receding-horizon U0 with a calibrated "
            "simplex scores at least FIXED_FAIR + 0.03 on scarce-coupled cells."
        ),
        hypothesis_h0=(
            "U0 scores below FIXED_FAIR (the deficit is a property of an objective with no value or deadline term: "
            "horizon-end dS rewards deferral), U_PLUS is within +/-0.03 of MPC, and U0 is decision-identical to the "
            "weighted-sum twin."
        ),
        metric="J = value-weighted share of correct answers at deadlines (scarce-coupled cells)",
        metric_higher_is_better=True,
        primary_contrast=["isrm_u0", "fixed_fair"],
        direction="greater",
        threshold=0.03,
        equivalence_margin=0.03,
        alpha=0.05,
        n_replicates=64,
        quick_replicates=16,
        secondary_contrasts=[["isrm_u0", "conventional_mpc"], ["u_plus", "conventional_mpc"], ["isrm_u0", "linear_twin"], ["isrm_u0", "hold_all"]],
        arms=[
            Arm("isrm_u0", "treatment", "U0(a) = alpha E_after - beta (S_after - S_now) - gamma (C_after - C_now), simplex calibrated on dev blocks, receding horizon H=4, execute first step"),
            Arm("fixed_fair", "comparator", "cyclic full-service schedule"),
            Arm("conventional_mpc", "comparator", "V_H + V_cont + lambda_B E_after with the shared forecaster"),
            Arm("u_plus", "comparator", "V_H + V_cont + alpha E_after - beta dS - gamma dC (extended utility)"),
            Arm("linear_twin", "control", "w_E E_after - w_S S_after - w_C C_after with the same weights as U0 (must be decision-identical)"),
            Arm("hold_all", "control", "never acts"),
        ],
        environment={"name": "metacog_budget", "status": "to be implemented from the audit's compact spec", "tasks": 6, "ticks": 256, "horizon": 4, "budget": {"B0": 6, "Bmax": 12, "refill_scarce": 1.5, "refill_ample": 6}},
        notes="Replicate = base-seed block (paired across arms). Also record missing-answer rate and the ample-vs-scarce difference-in-differences as extras to test the procrastination mechanism.",
    )


def T04() -> TestSpec:
    return TestSpec(
        id="T04_total_isrm_dwell_attribution",
        title="Ball-avoidance arena: is the v3 ISRM switching gain over adaptive energy-only explained by dwell/hysteresis?",
        source_project="D:/Total ISRM/benchmarks (v3 report)",
        source_claim=(
            "v3: ISRM vs adaptive energy-only +1.347 s survival [+1.140, +1.552] on the stationary grid; vs fixed "
            "energy-only -7.073 s [-7.308, -6.838]; ISRM switched 1.22x per run vs 13.50x for the adaptive comparator; "
            "disabling switching improved default energy +4.08 [2.42, 5.80]."
        ),
        hypothesis_h1="ISRM three-term switching beats an adaptive energy-only controller that is given ISRM's own dwell and margin (a dwell-only comparator) by at least 0.5 s survival on the stationary grid.",
        hypothesis_h0="The dwell-only comparator closes at least 80% of the +1.347 s gap; the three-term utility adds nothing beyond switching hygiene.",
        metric="survival seconds (120 s cap)",
        metric_higher_is_better=True,
        primary_contrast=["isrm_full", "adaptive_energy_dwell"],
        direction="greater",
        threshold=0.5,
        alpha=0.05,
        n_replicates=128,
        quick_replicates=32,
        secondary_contrasts=[["isrm_full", "adaptive_energy_only"], ["isrm_full", "fixed_energy_only"], ["isrm_no_switching", "isrm_full"]],
        arms=[
            Arm("isrm_full", "treatment", "three-term utility with model switching and paid recalibration (alpha 0.4, beta 0.35, gamma 0.25 frozen)"),
            Arm("adaptive_energy_dwell", "comparator", "adaptive energy-only switching plus ISRM's 1.5 s dwell and switching margin"),
            Arm("adaptive_energy_only", "comparator", "adaptive energy-only switching as in the original v3 comparator"),
            Arm("fixed_energy_only", "comparator", "fixed energy-only controller"),
            Arm("isrm_no_switching", "ablation", "ISRM utility with switching disabled"),
        ],
        environment={"name": "ball_avoidance_arena", "status": "to be implemented independently (2D arena, paired seeds, stationary grid regime)"},
        notes="The original highlighted cells were selected after inspecting all results; use fresh paired seeds only.",
    )


def T05() -> TestSpec:
    from isrm_sandbox.spec import InputRef

    return TestSpec(
        id="T05_flock_nagy2013_cont_vs_isrm_pe",
        title="Pigeon flocks (Nagy 2013): continuous salience comparator vs ISRM prediction-error-regulated switching",
        source_project="D:/Flock ISRM (reports/pilot_pigeon_nagy2013_pe_v1_findings.md)",
        source_claim=(
            "d = loss(CONT) - loss(ISRM-PE) = -0.091 [-0.120, -0.062] (CR1 t14), CONT better in 14 of 15 flights and in all "
            "three flocks; ISRM-PE vs own substrate +0.0002 [-0.0003, +0.0008]; NO_ADVANTAGE_DEMONSTRATED."
        ),
        hypothesis_h1="Re-implemented from the written controller notes on the raw trajectoriesD2.zip, ISRM-PE's onset-window loss is lower than CONT's by at least 0.05 energy-score units.",
        hypothesis_h0="CONT is lower (d about -0.09 +/- 0.03), ISRM-PE is within 0.01 of its substrate, and the regulator is nearly inert at 10 Hz because the heading-innovation floor (sigma_th0 ~0.004 rad) silences the CUSUM.",
        metric="onset-window energy-score loss (lower is better)",
        metric_higher_is_better=False,
        primary_contrast=["isrm_pe", "cont"],
        direction="greater",
        threshold=0.05,
        alpha=0.05,
        n_replicates=15,
        quick_replicates=5,
        secondary_contrasts=[["isrm_pe", "substrate_local_enhanced"], ["cont", "substrate_local_enhanced"], ["isrm_pe", "raw_trigger"], ["isrm_pe", "mms"], ["substrate_local_enhanced", "persistence_cv"]],
        arms=[
            Arm("isrm_pe", "treatment", "prediction-error-regulated switching (CUSUM on standardised max innovation, basins FOLLOW/COPY/IGNORE, budget E, re-implemented from the audit's compact spec)"),
            Arm("cont", "comparator", "continuous salience down-weighting of poorly predicted neighbours (the original winner)"),
            Arm("substrate_local_enhanced", "comparator", "topological local flocking model without any supervisor"),
            Arm("raw_trigger", "ablation", "same regulator on raw rate residuals"),
            Arm("mms", "ablation", "minimum-surprise model switching without CUSUM/budget"),
            Arm("persistence_cv", "control", "constant-velocity extrapolation"),
        ],
        environment={"name": "pigeon_replay_track_r", "status": "to be implemented: independent loader (11 whitespace columns, grid snap, reliability, airborne > 5 m/s), outcome-blind onset detector, Track R replay", "sample_hz": 10, "replicate_unit": "held-out flight"},
        inputs=[
            InputRef("D:/Flock ISRM/data/raw/pigeon_nagy2013/trajectoriesD2.zip", "715a3b91d31c0678be8ab5170ce824d25480ce061c1665b1f71733bc52384f34", "Nagy et al. 2013 pigeon flock trajectories (3 flocks x 10 birds, 15 flights, 10 Hz)"),
            InputRef("D:/Flock ISRM/data/raw/pigeon_nagy2010/pigeonflocks_trajectories.zip", "7b7f6636c0d8eba49dfb788a86da841216018cddf579984d974682e8affe7d2d", "Nagy et al. 2010 pigeon flock trajectories (1 flock, 15 flights, 5 Hz) for the sampling-rate comparison"),
        ],
        notes="Replicate = held-out flight (paired across arms). Cheap data-only pre-checks to record as extras: Stage-P1 lambda ~0.96 and sigma_th0 ~0.004 rad at 10 Hz versus 0.48 and 0.021 rad at 5 Hz; detector yield 85-97 onsets per fold.",
    )


def T06() -> TestSpec:
    return TestSpec(
        id="T06_tfm_isrm_gate_vs_quantile_gate",
        title="Forecast-gated trading: ISRM three-term gate (P5) vs uncertainty-threshold gate (P4) on synthetic basis/carry",
        source_project="D:/VA/tfm_isrm_experiment (RESULTS.md, PREREGISTRATION.md)",
        source_claim=(
            "P5 - P4 = +0.0305 units net at 5 bps/leg; mean daily +0.0020, 95% CI [-0.0253, +0.0374], p = 0.4725; -0.274 at "
            "1 bps; the pressure term lambda was tuned to 0 and forced lambda = 1 cost -0.18; NO DEMONSTRATED IMPROVEMENT."
        ),
        hypothesis_h1="Over a frozen quantile forecaster on synthetic AR(1) basis plus carry with realistic costs, the ISRM gate P5 beats the quantile gate P4 by at least 0.10 net units per block at 5 bps/leg over 60 out-of-sample day-blocks.",
        hypothesis_h0="|P5 - P4| < 0.1 |P4 - P3| and the pressure ablation (lambda = 1) never helps; the only effective ingredient is the uncertainty penalty P4 already has.",
        metric="net units per day-block at 5 bps/leg",
        metric_higher_is_better=True,
        primary_contrast=["p5_isrm_gate", "p4_quantile_gate"],
        direction="greater",
        threshold=0.10,
        alpha=0.05,
        n_replicates=60,
        quick_replicates=20,
        secondary_contrasts=[["p5_isrm_gate", "p3_point_rule"], ["p5_lambda1", "p5_isrm_gate"], ["p4_quantile_gate", "p3_point_rule"], ["p1_persistence", "p5_isrm_gate"]],
        arms=[
            Arm("p5_isrm_gate", "treatment", "U(a) = a y_hat - beta |a| u (1 + kappa min(D/u,3)) (1 + lambda min(max(R,0)/ln2, 3)) - c |a - x|, argmax, grid beta x lambda on dev"),
            Arm("p4_quantile_gate", "comparator", "enter +1 if q0.2 > c + m, -1 if q0.8 < -(c + m); exit on sign flip; same 9-cell tuning budget"),
            Arm("p3_point_rule", "comparator", "trade on the sign of the point forecast net of cost"),
            Arm("p5_lambda1", "ablation", "P5 with the pressure multiplier forced to lambda = 1"),
            Arm("p1_persistence", "control", "hold the previous position"),
            Arm("p0_flat", "control", "never trade"),
        ],
        environment={"name": "synthetic_basis_carry", "status": "to be implemented: AR(1) basis with regime shifts, carry, half-spread costs, frozen conformal quantile forecaster", "fee_bps_per_leg": [5, 1]},
        notes="Public venue data could replace the synthetic world; the original data (ValidAlpha pairs_1m) is proprietary and must not be used here.",
    )


def T07() -> TestSpec:
    return TestSpec(
        id="T07_pid_sidecar_energy_plant",
        title="ISRM sidecar over PID: ISRM-gated hold vs PID-score-gated hold on a synthetic energy-limited plant",
        source_project="D:/PID+ISRM, D:/isrm_ardupilot_phase_compare",
        source_claim=(
            "Recovery-phase PID alarm burden 19.2-32.1% (ACTIVE) vs 62.6-66.3% (SHADOW); throttle integral in the energy "
            "deficit not reduced (5664.7/7318.2/5594.1 vs 5653.3/5554.2/5675.9 %-s); the only intervention was a sustained "
            "CONSERVE_CRITICAL LOITER after which the mission was never resumed."
        ),
        hypothesis_h1="On an independent energy-limited plant with engine-multiplier steps, an ISRM-gated hold (U <= -0.75 sustained 3 s) reduces the recovery throttle integral by at least 10% relative to a PID-score-gated hold (score >= 1.35 sustained 3 s) at equal mission completion.",
        hypothesis_h0="The two gates are within +/-5% on throttle integral and alarm burden; a 3 s dwell on the PID alarm alone removes >= 25% of alarms; ISRM's channels add no lead time.",
        metric="recovery throttle integral as a fraction of the no-intervention run (lower is better)",
        metric_higher_is_better=False,
        primary_contrast=["isrm_gated_hold", "pid_gated_hold"],
        direction="greater",
        threshold=0.10,
        alpha=0.05,
        n_replicates=24,
        quick_replicates=8,
        secondary_contrasts=[["isrm_gated_hold", "pid_dwell_only"], ["pid_gated_hold", "no_intervention"]],
        arms=[
            Arm("isrm_gated_hold", "treatment", "E = 0.45(1 - thr) + 0.35 clip((aspd - 12)/8) + 0.20 batt; dS = mean of capped predictor and nav residuals; dC = 0.45 osc + 0.25 sat + 0.20 low + 0.10 conf; U = E - 0.7 dS - 0.7 dC; hold when U <= -0.75 for 3 s"),
            Arm("pid_gated_hold", "comparator", "hold when the PID health score (0.25 alt + 0.20 aspd + 0.20 xtrack + 0.15 low + 0.10 sat + 0.10 conf, with integral and derivative terms) >= 1.35 for 3 s"),
            Arm("pid_dwell_only", "ablation", "PID alarm with 3 s dwell and hysteresis, no ISRM channels"),
            Arm("no_intervention", "control", "PID flies the mission unaided"),
        ],
        environment={"name": "energy_limited_plant", "status": "to be implemented: second-order airspeed/altitude plant with throttle-limited energy, engine-multiplier disturbance steps, mission completion scored"},
        notes="Mission completion must be scored; the original benefit coincided with abandoning the mission (100% LOITER in recovery).",
    )


def T08() -> TestSpec:
    return TestSpec(
        id="T08_dual_weights_matching_and_trap",
        title="Dual weights: can fixed weights be matched to the virtual-queue controller, and is the controller a textbook drift-plus-penalty rule?",
        source_project="D:/ISRM Dual Weights (V3_PROTOCOL.md, V3_REPORT.md)",
        source_claim=(
            "No setting achieved the predeclared joint matching tolerance (primary worst normalised mismatch 2.741); with "
            "reserve-funded actions the controller enters an absorbing zero-reserve state (3,062 positive-supply ticks "
            "with R = 0); FIXED - ADAPTIVE work +0.295 [0.257, 0.334] in R2."
        ),
        hypothesis_h1="At matched E/S/C violation depths (all three within tolerance), the virtual-queue ADAPTIVE controller uses at least 0.03 less work per tick than FIXED weights.",
        hypothesis_h0="No fixed triple matches all three depths; a Lyapunov drift-plus-penalty twin reproduces ADAPTIVE's action sequence tick for tick; in R1 with reserve-funded actions the fraction of post-deficit ticks with R = 0 and stale supply exceeds 70%.",
        metric="mean work per tick (lower is better)",
        metric_higher_is_better=False,
        primary_contrast=["adaptive_queues", "fixed_matched"],
        direction="greater",
        threshold=0.03,
        alpha=0.05,
        n_replicates=16,
        quick_replicates=8,
        secondary_contrasts=[["adaptive_queues", "no_c"], ["adaptive_queues", "dpp_twin"]],
        arms=[
            Arm("adaptive_queues", "treatment", "Q_i <- max(0, Q_i + r_i); action = argmin[-V work(a) + sum_i Q_i r_hat_i(a)]"),
            Arm("fixed_matched", "comparator", "constant (alpha, beta, gamma) fitted to ADAPTIVE's R2 depth targets on training seeds"),
            Arm("no_c", "ablation", "consistency queue removed"),
            Arm("dpp_twin", "control", "textbook drift-plus-penalty with the same V and queues (expected identical)"),
        ],
        environment={"name": "reserve_supply_plant", "status": "to be implemented from the audit's plant description (reserve R, supply 0.10/tick, 40-tick deficits, cost 0.15 kappa, Sample/Update actions)"},
        notes="Report the joint-match score and the absorbing-state fraction as extras; the primary verdict is only meaningful if a match within tolerance exists.",
    )


def T09() -> TestSpec:
    return TestSpec(
        id="T09_isrm_lab_attribution_fashion_mnist",
        title="Native three-term U on Fashion-MNIST: attribution contrast against the matched arbitration control",
        source_project="D:/NeuralNets/isrm_lab (RESULTS.md, CRITICAL_REVIEW)",
        source_claim=(
            "isrm_native - matched_native_control accuracy +0.0241 [+0.0039, +0.0442] (5-cluster bootstrap) but exact "
            "sign-flip p = 0.125 and t(4) CI [-0.008, +0.056]; cost per completed task +1.837 [+0.683, +2.992] worse; "
            "verdict NO_ADVANTAGE_DEMONSTRATED."
        ),
        hypothesis_h1="With at least 10 training seeds, a 4-exit residual network on Fashion-MNIST arbitrated by the three-term U beats the matched control (same psi/phi, conventional arbitration) by at least 0.01 sealed accuracy without higher cost per completed task.",
        hypothesis_h0="The accuracy difference is within [-0.01, +0.05] with sign-flip p > 0.05 and cost per completed task is 1.0-2.5 RU higher; a dev-tuned confidence router beats the U-router in the binding-budget regime.",
        metric="sealed-partition accuracy",
        metric_higher_is_better=True,
        primary_contrast=["isrm_native", "matched_native_control"],
        direction="greater",
        threshold=0.01,
        alpha=0.05,
        n_replicates=10,
        quick_replicates=4,
        secondary_contrasts=[["isrm_native", "fixed_depth_selected"], ["isrm_native_noupdate", "isrm_native"], ["confidence_router", "isrm_native"]],
        arms=[
            Arm("isrm_native", "treatment", "three-term U arbitration over exits with learned plasticity"),
            Arm("matched_native_control", "comparator", "same psi/phi, conventional arbitration"),
            Arm("fixed_depth_selected", "comparator", "best fixed exit depth selected on dev"),
            Arm("isrm_native_noupdate", "ablation", "U arbitration without self-model updates"),
            Arm("confidence_router", "comparator", "one-line confidence threshold router tuned on dev"),
        ],
        environment={"name": "fashion_mnist_multi_exit", "status": "requires torch and the public Fashion-MNIST release; data manifest to be added with SHA-256 of the four raw files", "budget_ru": 1200, "feedback_delay": 8},
        notes="Only real-data study with a positive attribution contrast; underpowered at 5 seeds. GPU recommended (about 15 min per seed on an RTX 5070 in the original).",
    )


ALL = [T01, T01b, T02, T03, T04, T05, T06, T07, T08, T09]


def main() -> int:
    SPECS.mkdir(exist_ok=True)
    for build in ALL:
        spec = build()
        path = SPECS / f"{spec.id}.json"
        if path.exists():
            existing = TestSpec.load_frozen(path)
            print(f"keep   {spec.id}  {existing.frozen_hash[:12]}")
            continue
        h = spec.freeze(path)
        print(f"frozen {spec.id}  {h[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
