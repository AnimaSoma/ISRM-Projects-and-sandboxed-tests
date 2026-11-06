# T05 data pre-checks: independent pigeon loader and Stage-P1 (2026-11-06)

Data-only work for spec `specs/T05_flock_nagy2013_cont_vs_isrm_pe.json`. Nothing here tests the
claim; the T05 suite is not started (section 5). Numbers are in
`data/manifests/pigeon_stage_p1.json` (regenerate with `python -m isrm_sandbox.data.pigeon_stage_p1`).
Code: `isrm_sandbox/data/pigeons.py` (loader), `isrm_sandbox/data/pigeon_stage_p1.py` (estimator and
manifest), `tests/test_pigeon_loader.py` (19 protocol tests). Written from the archives' own readme and
headers and the loader rules in `reports/REIMPLEMENTATION_SPECS.md`; no code from `D:/Flock ISRM`.

## 1. Inputs and loader

Both archives match the SHA-256 in the frozen spec (`isolation.verify_inputs`, checked before every read).

| dataset | members | flights | birds | rows raw -> kept | cleaning actions fired | velocity self-check (median, p95) |
|---|---|---|---|---|---|---|
| Nagy 2013 (D2), 10 Hz | 148 | 15 (A, B, C x 5) | 30 (ffA1 has 8: A4, A8 absent) | 3,282,222 -> 3,282,222 | none | 0.00022 m/s, 0.00050 m/s |
| Nagy 2010 (D1), 5 Hz | 137 | 15 (ff1-ff11, hf1-hf4) | 13 (A-M, 7-10 per flight) | 2,426,294 -> 2,390,289 | 83 snapped, 13,969 duplicates, 22,036 NaN rows, 0 off-grid | 0.00011-0.00014 m/s, 0.00025 m/s |

* The published velocity equals the one-sample central difference of the published positions in every
  file (positions are printed to 0.1 mm, which sets the 1e-4 m/s floor), so the column semantics and the
  author smoothing are confirmed. The loader never uses the velocity columns otherwise.
* D1 cleaning matched the original data card file by file: ff2_C holds its series twice (13,877 duplicate
  rows, last kept), hf1_H opens with a 92-row duplicated out-of-order block, 83 stamps are 1 cs off the
  grid (ff4_D 2, ff7_C 75, ff8_L 2, ff9_D 2, ff9_L 2) and snap, 22,036 rows have no position (ff7_J 8,181).
* D2 needed no cleaning. Each file is a contiguous 0.1 s series; ffC4 starts one tick after its header's
  timeSTART, and A4 in ffA2 plus seven birds of ffC5 end one tick before their flockmates (8 absent
  cells on the flight grids in all). Interpolated samples are 11.9 percent overall but almost all on the
  ground: restricting to measured fixes removes 308 of 8.2 million airborne pairs. In D1 the interpolated
  share is 18-40 percent per flight and sits inside the airborne segments (section 4).
* Correction to `REIMPLEMENTATION_SPECS.md`: the D1 archive has 137 trajectory files, not 150. Its 155
  entries include 16 directory entries and 2 readmes. The per-flight bird counts (8, 10, 7, 10, 10, 7, 10,
  9, 10, 9, 10, 10, 9, 9, 9) sum to 137, as in the original QC report.

## 2. What Stage-P1 computes here

Relative-frame one-step predictor of the ISRM contract: observer i, neighbour j, backward-difference
kinematics at the native tick D; the reference of the pair is the circular mean heading of i's coherent
core (members within 0.5 rad of the leave-one-out mean) without j, held fixed across the base and scored
ticks; phi = wrap(theta_j - theta_ref); phi_hat(n+1) = phi + D[lambda (omega_j - omega_ref) - k_soc phi];
s_hat(n+1) = s_j + D k_spd (s_ref - s_j). OLS without intercept over every eligible ordered pair and tick;
sigma_th0, sigma_s0 = 1.4826 x MAD of the innovations per tick; bounds lambda [0, 1], k_soc [0, 5]/s,
k_spd [0, 2]/s (never active). Eligibility: present, reliable (2 measured fixes within 0.4 s) and airborne
(central-difference speed above 5 m/s) at frames n-2 to n+1, for both birds.

One stated departure: the original fitted over the substrate's sensed set (k = 9 nearest within r_max,
rear blind sector, reaction delay). Here the sensed set is every other eligible bird, as the task asked.

## 3. Results against the original values

| run | D | pairs | lambda | k_soc /s | k_spd /s | sigma_th0 rad | sigma_s0 m/s |
|---|---|---|---|---|---|---|---|
| Nagy 2013, native 10 Hz | 0.1 s | 8.20 M | **0.966** | 0.028 | 0.051 | **0.0043** | 0.143 |
| original, D2 fold 0 | 0.1 s | | 0.960 | 0.101 | 0.069 | 0.0042 | 0.147 |
| Nagy 2010, native 5 Hz | 0.2 s | 3.36 M | **0.943** | 0.014 | 0.114 | **0.0042** | 0.101 |
| original, D1 fold 0 | 0.2 s | | 0.476 | 0.247 | 0.250 | 0.0214 | 0.192 |
| Nagy 2013 decimated to 5 Hz, phase 0 | 0.2 s | 4.08 M | 0.889 | 0.053 | 0.084 | 0.0151 | 0.280 |
| Nagy 2013 decimated to 5 Hz, phase 1 | 0.2 s | 4.08 M | 0.889 | 0.053 | 0.085 | 0.0151 | 0.280 |
| Nagy 2010 decimated to 2.5 Hz (0.4 s tick), phase 0 / 1 | 0.4 s | 0.42 M / 0.45 M | 0.794 / 0.814 | 0.047 / 0.029 | 0.099 / 0.138 | 0.0158 / 0.0156 | 0.199 / 0.202 |

Per flight: D2 lambda 0.958-0.975 and sigma_th0 0.0037-0.0052 rad in all 15 flights, no flock stands
out; D1 free flights lambda 0.88-0.98 and sigma_th0 0.0046-0.0083 rad, D1 homing flights 0.96-0.99 and
0.0027-0.0044 rad. Sensitivities (manifest): defining omega_ref as the turn rate of the reference heading
instead of the mean member turn rate changes nothing to three decimals; the absolute (ballistic) frame
of the design document gives D1 lambda 0.988, k_soc 0.010/s, k_spd 0.126/s, which matches that
document's descriptive D1 values (0.985, 0.012, 0.12), so the kinematic conventions agree with the
original's; restricting D1 to neighbour ticks with four consecutive measured fixes leaves 18,271 pairs
(0.5 percent) with lambda 0.849 and sigma_th0 0.0085 rad.

## 4. Reading

1. **D2 at 10 Hz reproduces.** lambda, sigma_th0 and sigma_s0 land within 1-3 percent of the original
   fold-0 fit. k_soc is smaller (0.028 vs 0.101/s); the sensing mask and the fold composition differ, and
   both values say the social heading pull is weak against the constant-turn term.
2. **D1 at 5 Hz does not reproduce as a one-tick fit.** On all samples the D1 innovation floor is
   0.0042 rad, the same as D2's, not five times larger, and lambda is 0.94, not 0.48.
3. **Rate versus dataset.** Decimating D2 to 5 Hz multiplies sigma_th0 by 3.5 (0.0043 to 0.0151 rad) and
   lowers lambda to 0.89, so the sampling step alone produces most of a "five times smaller innovation at
   10 Hz". D1 itself is smoother than decimated D2 (its data card gives 1 Hz power at 2e-4 of the low band
   against 1e-3 for D2), so at a one-tick horizon it shows no large floor.
4. **Where the original D1 numbers come from.** The original tracker treats author-interpolated samples
   as unsensed and scores the predictor when the next measured fix arrives, over the elapsed horizon.
   In the airborne, reliable segments of D1, 79 percent of consecutive measured fixes are 2 ticks (0.4 s)
   apart and 21 percent 1 tick (313,520 gaps counted); in D2 every one of 1.10 million gaps is 1 tick.
   Scoring the same one-step predictor 2 ticks ahead on D1 gives lambda 0.874 and sigma_th0 0.0119 rad,
   and 3 ticks ahead 0.799 and 0.0220 rad; D2 scored 2 ticks (0.2 s) ahead gives 0.920 and 0.0123 rad,
   4 ticks (0.4 s) ahead 0.801 and 0.0351 rad. The original's D1 floor of 0.0214 rad is therefore what
   this smooth data shows at the 0.4-0.6 s horizon its tracker really works at on D1, not a property of
   the flock at 0.2 s; at an equal 0.4 s horizon D2 is the noisier dataset (0.035 against 0.012 rad).
   The horizon accounts for the floor but only for part of lambda (0.80 at 0.6 s against the original
   0.48); the rest presumably comes from how the original tracker forms omega across stale samples,
   which cannot be checked here without reading its code.
   Decimating D1 itself to a 0.4 s tick (every other frame, the tracker's effective tick on D1) gives
   lambda 0.79-0.81, sigma_th0 0.0157 rad and sigma_s0 0.20 m/s (original 0.192): the floor and the speed
   scale of the original D1 calibration are those of the fix spacing, while lambda stays near 0.8.
5. **Consequence for T05.** The spec's H0 (the regulator is nearly inert at 10 Hz because sigma_th0 of
   about 0.004 rad silences the CUSUM) rests on a per-tick floor that is a property of the sampling step
   and of the fix spacing, not of flock behaviour. A sandbox suite must therefore implement the
   imputed-unsensed rule and the h/D innovation scaling exactly as written, and record sigma_th0 under
   that rule as an extra, or the 10 Hz versus 5 Hz contrast the spec expects will not appear.

## 5. Not done

* The T05 suite is not started: it needs the outcome-blind onset detector, the Track R replay
  environment, the local_enhanced substrate, the ISRM-PE, CONT, RAW and MMS arms, the persistence control
  and the energy score, several sessions of work. `python -m isrm_sandbox list` shows T05 as spec only and
  the ledger has no T05 entry. The loader and the masks above are the inputs that suite will need.
* `reports/REIMPLEMENTATION_SPECS.md` still says "150 files" for D1 (should read 137); left for the audit
  session that owns that document.
