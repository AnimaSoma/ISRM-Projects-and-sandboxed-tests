# Re-implementation specifications for the pending suites

Compact specifications extracted by the 2026-11-06 audits from the original
projects' written contracts and code. They exist so that suites T03 to T09 can be
written inside the sandbox without importing or reading the original code again.
Where a spec records a parameter the original used but did not document, it says
so. Numbers in brackets are the original results the sandbox is checking against.

---

## T03 Shared-budget metacognition benchmark (D:/Multi ISRM/isrm_metacognition_benchmark)

World. n = 6 tasks, T = 256 ticks. Private latent z_i(t+1) = 0.97 z_i(t) + eps_i + shock_i,
sigma_eps in {0.02, 0.05}; in "changing" cells shocks ~ Bernoulli(0.01) x N(0, 0.3^2).
Observation z_i + sigma_obs n, sigma_obs in {0.05, 0.15}. Public DAG with at most 2
parents per task, signed weights a_ij with sum_j |a_ij| = 1, coupling kappa in
{0, 0.35}; target y_i = z_i + kappa sum_j a_ij y_j (linear family) or tanh of the same
(nonlinear family). Periods {8, 12, 16}, values {1, 3, 9}, compute lengths {1, 2, 3},
two tasks each, independent permutations. Deadlines at P + phi + kP. Tolerance
tol_i = max(0.02, 0.5 sd_stable_i). Correct iff |y_hat - y(t_d)| <= tol; a missing
answer is incorrect. Objective J = sum over due deadlines of V_i correct_i divided by
sum of V_i.

Operations and credits. OBSERVE 1 credit (scalar Kalman update, a = 0.97,
q = sigma_eps^2 + 0.01 x 0.09, r = sigma_obs^2); COMPUTE 1 credit per unit times L
(needs published parents; snapshots (m, v, t) and parent values; publishes on
completion); CONTINUE 1; CANCEL 0.5; HOLD 0. Budget B0 = 6, Bmax = 12, refill 1.5
(scarce) or 6 (ample) per tick, overflow discarded. Tick order: refill and predict,
observe, decide, execute, publish, score. Primary comparison at zero controller
overhead (the calibrated tariff makes every controller HOLD).

Measurements. E_t = B_t / Bmax. S = mean_i log(1 + v_i / v_ref_i) with
v_i = sum_j W_ij^2 var_j, W = (I - kappa A)^-1 (Jacobian for tanh), ancestor
snapshots propagated as mu_j = a^k m_j, var_j = a^2k v_j + q_j (1 - a^2k)/(1 - a^2),
unpublished tasks at prior variance (= v_ref). C = mean_i c_i with
c_i = min(1, |f_i(m_used, parents_used) - f_i(m_used, parents_latest)| / tol_i) for
published non-root tasks, else 0.

Shared forecaster. Deterministic expected-value rollout, horizon H = 4: variances
exact, means decay by a, a planned OBSERVE reduces variance with mean unchanged;
P(correct) = Phi((tol - b)/sd) - Phi((-tol - b)/sd) with b = y_hat - mu, recalibrated
by sigma(c0 + c1 logit) on dev (c1 about 0.81).

Candidates. HOLD; per task OBSERVE, COMPUTE to completion, CANCEL, CANCEL then
COMPUTE, OBSERVE then COMPUTE; parent-to-child chains within H; pairwise and bulk
first steps; at most 96 first steps and 128 plans; infeasible dropped.

Scores (argmax, execute the first step; ties by lower first-step cost, then
HOLD < OBSERVE < COMPUTE < CONTINUE < CANCEL):
U0(a) = alpha E_after - beta (S_after - S_t) - gamma (C_after - C_t), alpha + beta +
gamma = 1 (calibrated 1/6, 5/6, 0).
LINEAR_EQUIVALENT(a) = w_E E_after - w_S S_after - w_C C_after = U0 + constant.
U_PLUS(a) = V_H + V_cont + alpha E_after - beta dS - gamma dC (alpha 2, beta 4,
gamma 0), V_H = sum over deadlines in t..t+3 of V_i Pcal_i, V_cont = sum_i V_i Pcal
(next deadline after t+3 with no further action).
CONVENTIONAL_MPC(a) = V_H + V_cont + lambda_B E_after, lambda_B = 3.4145.
FIXED_FAIR = cyclic full-service schedule. HOLD_ALL never acts.

Original results. U0 - MPC = -0.214 [-0.234, -0.193]; U0 - FIXED_FAIR = -0.063
[-0.085, -0.043]; U_PLUS - MPC = +0.003 [-0.002, +0.009]; U0 vs LINEAR_EQUIVALENT
0 mismatches in 2,560 episodes; missing-answer rate U0 0.31 vs MPC 0.02; U0 worse in
ample than scarce cells (difference in differences +0.095 [0.065, 0.126]).

---

## T04 Ball-avoidance arena (D:/Total ISRM/benchmarks)

The original arena is a Node.js 2D ball-avoidance environment with a 120 s survival
cap; v3 adds model switching with paid recalibration. Frozen utility weights
alpha 0.4, beta 0.35, gamma 0.25. Arms: ISRM full, adaptive energy-only, fixed
energy-only, ISRM with switching disabled. The original reports ISRM switching 1.22
times per run versus 13.50 for the adaptive comparator, which is why a dwell-only
comparator (adaptive energy-only plus ISRM's 1.5 s dwell and switching margin) is the
decisive addition. The arena itself must be re-created independently (paired seeds,
stationary grid regime); the original source zips are in D:/Total ISRM and may be
read for the environment rules but not imported.

Original results. v3 ISRM - adaptive energy-only +1.347 s [+1.140, +1.552] on the
stationary grid; ISRM - fixed energy-only -7.073 s [-7.308, -6.838]; disabling
switching improved default energy +4.08 [2.42, 5.80] and harsh-cycle survival +21.16
[17.67, 24.66]. v2: 0 wins in 72 cells, terminal energy -17.43 [-18.15, -16.72].

---

## T05 Pigeon flocks (D:/Flock ISRM)

Data (hashed in the spec). Nagy 2013 `trajectoriesD2.zip`: 148 files
`flight/ff<F>n/ff<F>n_p<F>k`, flocks A/B/C, 5 flights each, 10 Hz. Nagy 2010
`pigeonflocks_trajectories.zip`: 150 files `<flight>/<flight>_<Pigeon>.txt`, ff1-ff11
and hf1-hf4, 5 Hz. Format: `#` header lines, then 11 whitespace-separated columns
t(centiseconds) X Y Z dX dY dZ d2X d2Y d2Z GPS_signal (m, m/s, m/s^2; GPS_signal 1
measured, 0 interpolated). D1 quirk: the last header line is not newline-terminated,
so split once on "GPS signal". D2 rows end with a trailing tab. Loader rules to
mirror: drop duplicated timestamps keeping the last; snap stamps within +/-2 cs to
the 20 cs (D1) or 10 cs (D2) grid, drop other off-grid rows; drop NaN positions;
frame = (t_cs - flight t0) / grid; reliability = at least 2 measured fixes within
+/-0.4 s; airborne = horizontal speed > 5 m/s; use x, y only. Positions are
author-smoothed (Gaussian sigma 0.4 s plus B-spline), so the published velocity
equals the 0.2 s central difference (loader self-check: median mismatch 0.0001 m/s).

Onset detector (outcome-blind). Per flight, event windows anchored at initiator
onsets: prefix 2.0 s, warm-up 6.0 s, gap 1.2 s before onset, 2.0 s scored, 0.8 s pad,
at least 5 followers, at most 12 birds; thresholds from training flights only
(D2: theta_D 44-47 degrees, theta_A about 4 degrees, phi_pre about 9 degrees);
cap 12 events per flight; expected yield 85-97 onsets per fold on D2.

Stage P (training flights, closed form). P1: OLS for lambda, k_soc, k_spd over all
airborne ordered pairs; sigma_th0, sigma_s0 as 1.4826 MAD [D2 fold 0: lambda 0.960,
k_soc 0.101/s, k_spd 0.069/s, sigma_th0 0.0042 rad, sigma_s0 0.147 m/s; D1 fold 0:
0.476, 0.247, 0.250, 0.0214 rad, 0.192]. P2: tau_h in {1, 2, 5, 10, 20} s by one-step
Gaussian predictive density. P3: null tables under always-FOLLOW on unselected anchor
windows: mu0, s0 and CUSUM quantiles h_on(p) [D2 fold 0 h(0.99) = 174.7; D1 55.4].

ISRM-PE regulator. Tick D = sample period. Per neighbour j with a fresh measured
sample: relative-frame one-step predictor phi_hat = phi_b + h[lambda (omega_j,b -
omega_ref,b) - k_soc phi_b], s_hat = s_b + h k_spd (s_ref,b - s_b); innovation
q = e_th^2 / (V_th h/D) + e_s^2 / (V_s h/D) with protected precision (update only in
FOLLOW, in band q < 9.21, a_h = 1 - exp(-D/tau_h), clip 9V, floor (0.5 sigma0)^2).
z = clip((log(1 + max_j q) - mu0)/s0, +/-8). CUSUM G <- max(0, G + z - 0.5) in FOLLOW
when not refractory; trigger at G >= h_on(lp_on) (default lp_on 2) and E >= 0.25.
Choose argmax_m [-S_m + gamma_C C_m - 0.25/max(E, 0.05) 1[m != FOLLOW] + beta_Q a_i
+ beta_F F_i (COPY)] over feasible {FOLLOW, COPY (majority deviants), IGNORE
(deviants), EVADE (threat)}; absorb if FOLLOW else switch and pay E -= 0.25; return
when dwell >= 0.4 s with 2 in-band ticks, or dwell >= T_max, or targets lost (then
refractory 1.0 s); re-select on a 0.5 utility margin; E regenerates 0.1/s in FOLLOW.
Steering: FOLLOW = substrate; IGNORE = zero alignment and cohesion weight on targets;
COPY = plus rho_c a_max (u_targets - u_i), clamped to a_max; shared motion limits
(0.5 and 99.5 percent airborne quantiles) and shared noise after.

CONT comparator (the winner). Same tracker; q_bar = EMA(q, tau_s);
sal = sigmoid((log(1 + q_bar) - l0)/s_l), l0 the training quantile at lp_sal;
wF = 1 + beta_F [front or side]; kappa_i = sum over majority-sign sal wF divided by
sum over sensed wF; steering every dt: alignment and cohesion weights
1 - rho_D sal (1 - kappa), plus rho_c kappa a_max (u_S - u_i) with u_S the
sal-weighted majority heading; no modes, no hysteresis; inert at rho_c = rho_D = 0.
Fitted D1 fold 0: lp_sal 0.40, s_l 0.68, tau_s 0.34 s, rho_c 0.54, rho_D 0.44,
beta_F 1.99.

Substrate `local_enhanced`: topological k nearest within r_max, rear blind sector,
reaction delay tau, weights w_coh, w_ali, w_sep, r_sep, cruise-speed term, a_max,
process noise (D1 fold 0 fit: k 9, tau 0.06 s, a_max 24.1). Use Ornstein-Uhlenbeck
acceleration noise (tau 0.5 s) and 99.9 percent limit quantiles, which removed the
limit-binding artefact found in pe_v1.

Original results. D2: d = loss(CONT) - loss(ISRM-PE) = -0.091 [-0.120, -0.062],
CONT better in 14 of 15 flights; ISRM-PE vs substrate +0.0002 [-0.0003, +0.0008];
occupancy of non-FOLLOW basins 1.0 percent at 10 Hz (6.6 percent at 5 Hz on D1).

---

## T06 Forecast gate (D:/VA/tfm_isrm_experiment)

Per series i, hourly origin t, action a in {-1, 0, +1}, current position x:
U(a) = a y_hat - beta |a| u (1 + kappa min(D/u, 3)) (1 + lambda min(max(R, 0)/ln 2, 3))
- c |a - x|, with y_hat the rolling-calibrated median 24 h forecast in bps (bias =
median residual, scale = 80 percent conformal inflation over trailing 72 h matured
outcomes, at least 2,000 samples), u = (q90 - q10)/2 calibrated, D = |y_hat_model_1 -
y_hat_model_2| (forecaster disagreement), R = 8 h change of the hourly mean of a
composite pressure S, c = 60-minute median half-spread (bps) + 2 x fee per leg;
a = argmax with ties keeping x; fallback hold; open positions capped at 25 percent of
the universe admitted by |y_hat| - c. Chosen cell beta 1, kappa 1, lambda 0 from the
grid beta in {0.25, 0.5, 1} x lambda in {0, 0.5, 1}.

P4 comparator: enter +1 if q_0.2 > c + m, enter -1 if q_0.8 < -(c + m) (tau 0.2, m 10
bps from the grid tau in {0.2, 0.3, 0.4} x m in {0, 5, 10}); exit when x y_hat < 0;
same cap, fallback, costs and forecast cache. P3 trades on the sign of the point
forecast net of cost. Both tuned on a development block by worst-of-two-halves net
at 5 bps.

Pressure composite S (for lambda > 0): five families (|basis|, |funding spread 8 h|,
|funding spread 5 min minus 8 h|, index residual, change in log open interest), each a
one-sided z against a trailing 3-day median/MAD refreshed hourly (floor 1e-4);
S = weighted RMS of log(1 + z); R = 8 h change of hourly mean S.

The original data is proprietary; use synthetic AR(1) basis with regime shifts plus
carry, or public venue data. Original results: P5 - P4 daily +0.0020 [-0.0253,
+0.0374], p 0.47; -0.274 at 1 bps; lambda = 1 forced: -0.18 with entries 70 -> 5;
persistence +0.663 beats both.

---

## T07 ISRM sidecar over PID (D:/isrm_ardupilot_phase_compare, D:/PID+ISRM)

Sidecar. E = 0.45 clip((100 - thr)/100) + 0.35 clip((aspd - 12)/8) + 0.20 clip(batt/100).
dS = mean of seven terms each capped at 3: |x - x_hat|/s for (airspeed, 3.0),
(climb, 1.5), (pitch, 0.08), (roll, 0.15) with x_hat = x_prev + v_ema dt (EMA alpha
0.08); |aspd_err|/4; |alt_err|/25; |xtrack|/25. dC = 0.45 osc + 0.25 sat + 0.20 low +
0.10 conf, osc = mean of clip(std_50(climb)/1.5, std(pitch)/0.12, std(roll)/0.25,
std(aspd)/3.0, 0, 2); sat = clip((thr - 85)/15, 0, 1.5); low = clip((14 - aspd)/4, 0,
1.5); conf = [thr > 80 and climb < 0.2] clip((80 - 20 climb)/80, 0, 1) + [alt_err > 15
and climb < 0.3] clip(alt_err/50, 0, 1). U = E - 0.7 dS - 0.7 dC; U <= -0.75
CONSERVE_CRITICAL, U <= -0.25 RECALIBRATE_WARN, else PROCEED; escalate immediately,
de-escalate only after a 1.5 s dwell. Intervene (hold/loiter) when critical is
sustained >= 3.0 s in the allowed phase.

PID health score. err = 0.25 |alt_err|/25 + 0.20 |aspd_err|/4 + 0.20 |xtrack|/25 +
0.15 low + 0.10 sat + 0.10 conf; score = err + 0.02 integral(err) (clamped at 10) +
0.20 max(0, EMA_0.2(d err/dt)); WARN >= 0.85, CRIT >= 1.35, 1.5 s dwell. Hybrid =
AND of PID critical and ISRM critical.

Plant to build: second-order airspeed and altitude dynamics with throttle-limited
energy, engine-multiplier disturbance steps, waypoint mission with completion scored.
Original (ArduPilot SITL, 3 vs 3 runs): recovery alarm burden 19-32 percent (active)
vs 63-66 percent (shadow), but 100 percent LOITER in recovery, mission never resumed,
energy-deficit throttle integral not reduced.

---

## T08 Dual weights (D:/ISRM Dual Weights)

Plant. Reserve R in [0, 1], start 0.8; supply 0.10 per tick with 40-tick deficits
every about 400 +/- 40 ticks (first at t = 200); true cost per unit effort 0.15 kappa_true,
kappa drifting to 0.6 or 1.6 (regime R1 never, R2 at deficit + 150..250, R3 at deficit
onset). Meters: E residual = E_MIN (0.30) - clip(R + s_known - predicted cost, 0, 1);
S = staleness of the supply estimate (+0.10 per tick, cap 1, reset by Sample;
threshold 0.20); C_est = 0.7 C + 0.3 min(1, |actual - predicted spend| / predicted)
(threshold 0.10; reset by Update, an oracle refit kappa_model := kappa_true);
C_true = min(1, |kappa_model - kappa_true| / kappa_0) reported separately.
Actions: Abstain (effort 0), Act 0.5, Act 1, Sample (cost 0.01; sets s_known to the
current supply, S := 0), Update (cost 0.10). v3 feasibility: Sample needs R >= 0.01,
Update needs R >= 0.10. 4,000 ticks per run.

Controller. Queues Q_i <- max(0, Q_i + r_i) on realised signed residuals; decision
argmin_a [-V effort(a) + sum_i Q_i r_hat_i(a)], V = 1, first-minimum tie rule.
FIXED = constant (alpha, beta, gamma) with matched residual treatment; NO_C sets
Q_C = 0. Matching: fit (alpha, beta, gamma) in [0, 64]^3 (1,000 coarse triples plus
two refinement rounds) to ADAPTIVE's R2 depth targets on training seeds 100-103 with
tolerance max(10 percent of target depth, 0.1 percent of the meter threshold);
evaluate on fresh seeds.

Original results. No triple matched (worst normalised mismatch 2.741); held-out
FIXED - ADAPTIVE work +0.295 [0.257, 0.334] (R2) and +0.316 [0.273, 0.350] (R3);
NO_C - ADAPTIVE C_true +0.290 [0.231, 0.340]; absorbing zero-reserve state in R1
(3,062 positive-supply ticks with R = 0). Textbook drift-plus-penalty with the same V
and queues is expected to be decision-identical (the sandbox's formalism test already
confirms this on random instances).

---

## T09 Native three-term U on Fashion-MNIST (D:/NeuralNets/isrm_lab)

Fashion-MNIST 70k, source-disjoint partitions (meta-train 62 percent, dev 16, sealed
22), held-out corruption family (occlusion) and severity. A 4-exit residual network;
episodes of 3 x 64 arrivals with budget B0 = 1200 resource units and feedback delay
8. Arms: isrm_native (three-term U arbitration over exits with learned plasticity
phi and estimator psi), matched_native_control (same psi and phi, conventional
arbitration), fixed_depth_selected, isrm_native_noupdate, confidence_router (one-line
threshold tuned on dev). Coefficient ablations in the original: alpha = 0 doubles
depth, gamma = 0 removes VERIFY, beta = 0 collapses to shallow. Original: 5 training
seeds, about 75 min on an RTX 5070; pinned torch 2.10 dev. Results: isrm_native -
matched_native_control accuracy +0.0241 [+0.0039, +0.0442] by 5-cluster bootstrap but
exact sign-flip p 0.125; cost per completed task +1.837 [+0.683, +2.992] worse;
confidence router 0.8147 vs fixed depth 0.7403 in the binding-budget regime while the
U-router lands at 0.6760.
