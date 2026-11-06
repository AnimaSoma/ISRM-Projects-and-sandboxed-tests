"""Stage-P1 pre-check for spec T05: closed-form fit of the relative-frame one-step
neighbour predictor on the observed pigeon tracks. Data only: no controller, no rollout.

What is fitted (reports/REIMPLEMENTATION_SPECS.md section T05; the Flock ISRM documents
docs/ISRM_CONTRACT.md 9.1 and docs/DESIGN_ISRM_SWITCHING.md section 3, read as text):

  kinematics at the native tick D, backward differences of the published positions:
      v_j(n) = (P_j(n) - P_j(n-1)) / D,  s_j = |v_j|,  theta_j = atan2(v_j),
      omega_j(n) = wrap(theta_j(n) - theta_j(n-1)) / D
  observer i senses M_i(n) = every other bird that is present, reliable and airborne at
  frames n-2 .. n+1; core(i) = members within phi_core = 0.5 rad of the leave-one-out
  circular mean of the other members; the reference of the ordered pair (i, j) is the
  circular mean over core(i) without j, with the SAME member set at ticks n-1, n, n+1;
      phi_ij(n)       = wrap(theta_j(n) - theta_ref_ij(n))
      phi_hat_ij(n+1) = phi_ij(n) + D [lambda (omega_j(n) - omega_ref_ij(n)) - k_soc phi_ij(n)]
      s_hat_ij(n+1)   = s_j(n) + D k_spd (s_ref_ij(n) - s_j(n))
  ordinary least squares, no intercept, of the one-step changes on their regressors over
  every valid ordered pair and tick; innovations e_th = wrap(phi(n+1) - phi_hat),
  e_s = s(n+1) - s_hat; sigma_th0, sigma_s0 = 1.4826 x MAD of the innovations (per tick).
  Parameter bounds as in the original: lambda in [0, 1], k_soc in [0, 5]/s, k_spd in [0, 2]/s.

Departure from the original Stage P1, stated: the original defined M_i with the fitted
substrate's sensing rule (topological k nearest within r_max, a rear blind sector and a
reaction delay). Here M_i is every other eligible bird, as the task asked ("all airborne
ordered pairs"). With k = 9 in the original fit and at most 10 birds per flight the two
sets differ only by the blind sector and the delay.

Variants: `omega_ref` as the mean turn rate of the reference set (default) or as the turn
rate of the reference heading itself; `frame = "absolute"` reproduces the design
document's earlier ballistic form, theta_hat = theta_j + D [lambda omega_j + k_soc
wrap(theta_bar_i - theta_j)] with the consensus over M_i plus the observer, as a second
published yardstick (its descriptive Nagy 2010 values were lambda ~0.985, k_soc ~0.012/s,
k_spd ~0.12/s); `measured_only` keeps only neighbour ticks whose four frames are measured
GPS fixes rather than author-interpolated points.

Run as a module to write data/manifests/pigeon_stage_p1.json:
    python -m isrm_sandbox.data.pigeon_stage_p1 [--spec ...] [--out ...] [--flights ffA1 ...]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .. import isolation
from ..spec import TestSpec
from . import pigeons as P

ROOT = isolation.SANDBOX_ROOT
DEFAULT_SPEC = ROOT / "specs" / "T05_flock_nagy2013_cont_vs_isrm_pe.json"
DEFAULT_OUT = ROOT / "data" / "manifests" / "pigeon_stage_p1.json"
MAD_TO_SIGMA = 1.4826
BOUNDS = {"lambda": (0.0, 1.0), "k_soc": (0.0, 5.0), "k_spd": (0.0, 2.0)}

ORIGINAL_VALUES = {
    "source": (
        "reports/REIMPLEMENTATION_SPECS.md section T05, quoting the Flock ISRM Stage P1 fits on the "
        "training flights of fold 0 (about 60 percent of the flights, substrate sensing mask)"
    ),
    "nagy2013_fold0": {"lambda": 0.960, "k_soc": 0.101, "k_spd": 0.069, "sigma_th0": 0.0042, "sigma_s0": 0.147},
    "nagy2010_fold0": {"lambda": 0.476, "k_soc": 0.247, "k_spd": 0.250, "sigma_th0": 0.0214, "sigma_s0": 0.192},
    "nagy2010_absolute_frame_descriptive": {
        "lambda": 0.985,
        "k_soc": 0.012,
        "k_spd": 0.12,
        "source": "D:/Flock ISRM/docs/DESIGN_ISRM_SWITCHING.md section 3, the controller input's descriptive D1 values",
    },
}


def wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# --------------------------------------------------------------------------- config
@dataclass(frozen=True)
class StageP1Config:
    frame: str = "relative"  # relative | absolute
    omega_ref: str = "mean_turn_rate"  # mean_turn_rate | reference_heading
    phi_core: float = 0.5  # rad
    min_speed: float = 5.0  # m/s, airborne
    reliability_window_s: float = 0.4
    min_fixes: int = 2
    measured_only: bool = False
    horizon_ticks: int = 1  # score the one-step predictor this many ticks ahead (h = horizon_ticks * D)
    chunk: int = 4096

    def __post_init__(self):
        if self.frame not in ("relative", "absolute"):
            raise ValueError("frame must be relative or absolute")
        if self.omega_ref not in ("mean_turn_rate", "reference_heading"):
            raise ValueError("omega_ref must be mean_turn_rate or reference_heading")
        if int(self.horizon_ticks) < 1:
            raise ValueError("horizon_ticks must be at least 1")


# --------------------------------------------------------------------------- kinematics
def eligibility(flight: P.Flight, cfg: StageP1Config) -> np.ndarray:
    ok = flight.present & P.reliability_mask(flight, cfg.reliability_window_s, cfg.min_fixes)
    ok &= P.airborne_mask(flight, cfg.min_speed)
    if cfg.measured_only:
        ok &= flight.measured
    return ok


def kinematics(flight: P.Flight) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward-difference heading, speed and turn rate at the native tick; NaN where undefined."""
    dt = flight.dt
    vx = np.full_like(flight.x, np.nan)
    vy = np.full_like(flight.y, np.nan)
    vx[:, 1:] = np.diff(flight.x, axis=1) / dt
    vy[:, 1:] = np.diff(flight.y, axis=1) / dt
    theta = np.arctan2(vy, vx)
    speed = np.hypot(vx, vy)
    omega = np.full_like(theta, np.nan)
    omega[:, 1:] = wrap(np.diff(theta, axis=1)) / dt
    return theta, speed, omega


# --------------------------------------------------------------------------- rows
def iter_pair_rows(flight: P.Flight, cfg: StageP1Config, ok: np.ndarray | None = None) -> Iterator[dict]:
    """Yield chunks of regression rows for every valid ordered pair (observer i, neighbour j).

    Each chunk is a dict of equal-length 1-D arrays: y_th (one-step heading change / D),
    x_lam, x_soc (its two regressors), y_s (one-step speed change / D) and x_spd.
    """
    n_birds, n_frames = flight.x.shape
    steps = int(cfg.horizon_ticks)
    if n_frames < steps + 3 or n_birds < 2:
        return
    if ok is None:
        ok = eligibility(flight, cfg)
    theta, speed, omega = kinematics(flight)
    dt = flight.dt
    h = steps * dt  # elapsed horizon between the base tick and the scored tick
    # base tick n needs frames n-2 .. n+steps eligible; index m of `window` is n - 2
    length = n_frames - (steps + 2)
    window = np.ones((n_birds, length), dtype=bool)
    for k in range(steps + 3):
        window &= ok[:, k : k + length]
    good = np.nonzero(window.sum(axis=0) >= 2)[0]
    eye = np.eye(n_birds, dtype=bool)
    for start in range(0, len(good), cfg.chunk):
        m = good[start : start + cfg.chunk]
        n = m + 2
        w = window[:, m].T  # [Nc, B]
        th0, th1, thm = theta[:, n].T, theta[:, n + steps].T, theta[:, n - 1].T
        om0, s0, s1 = omega[:, n].T, speed[:, n].T, speed[:, n + steps].T

        def z(a: np.ndarray) -> np.ndarray:  # zero where ineligible so NaN never leaks into sums
            return np.where(w, a, 0.0)

        c0, si0 = z(np.cos(th0)), z(np.sin(th0))
        omz, sz = z(om0), z(s0)
        sensed = w[:, :, None] & w[:, None, :] & ~eye  # [Nc, B(i), B(k)]
        sensed_f = sensed.astype(float)
        shape = sensed.shape

        def bc(a: np.ndarray) -> np.ndarray:
            return np.broadcast_to(a, shape)

        def sum_over(members_f: np.ndarray, a: np.ndarray) -> np.ndarray:  # [Nc, B]
            return (members_f * a[:, None, :]).sum(axis=-1)

        if cfg.frame == "absolute":
            own = w.astype(float)
            cons_c = sum_over(sensed_f, c0) + own * c0
            cons_s = sum_over(sensed_f, si0) + own * si0
            cnt = sensed_f.sum(axis=-1) + own
            theta_bar = np.arctan2(cons_s, cons_c)
            with np.errstate(divide="ignore", invalid="ignore"):
                s_bar = (sum_over(sensed_f, sz) + own * sz) / cnt
            valid = sensed
            y_th = bc(wrap(th1 - th0)[:, None, :] / h)
            x_lam = bc(om0[:, None, :])
            x_soc = wrap(theta_bar[:, :, None] - th0[:, None, :])
            y_s = bc((s1 - s0)[:, None, :] / h)
            x_spd = s_bar[:, :, None] - s0[:, None, :]
        else:
            c1, si1 = z(np.cos(th1)), z(np.sin(th1))
            cm, sim = z(np.cos(thm)), z(np.sin(thm))
            tot_c, tot_s = sum_over(sensed_f, c0), sum_over(sensed_f, si0)
            cnt = sensed_f.sum(axis=-1)
            loo_c = tot_c[:, :, None] - sensed_f * c0[:, None, :]
            loo_s = tot_s[:, :, None] - sensed_f * si0[:, None, :]
            loo_cnt = cnt[:, :, None] - sensed_f
            loo_mean = np.arctan2(loo_s, loo_c)
            core = sensed & (loo_cnt >= 1) & (np.abs(wrap(th0[:, None, :] - loo_mean)) <= cfg.phi_core)
            core_f = core.astype(float)
            cc0, cs0 = sum_over(core_f, c0), sum_over(core_f, si0)
            cc1, cs1 = sum_over(core_f, c1), sum_over(core_f, si1)
            ccm, csm = sum_over(core_f, cm), sum_over(core_f, sim)
            c_om, c_sp, c_cnt = sum_over(core_f, omz), sum_over(core_f, sz), core_f.sum(axis=-1)

            def without_j(total: np.ndarray, a: np.ndarray) -> np.ndarray:  # [Nc, B(i), B(j)]
                return total[:, :, None] - core_f * a[:, None, :]

            ref_cnt = without_j(c_cnt, np.ones_like(c0))
            valid = sensed & (ref_cnt >= 1)
            ref0 = np.arctan2(without_j(cs0, si0), without_j(cc0, c0))
            ref1 = np.arctan2(without_j(cs1, si1), without_j(cc1, c1))
            with np.errstate(divide="ignore", invalid="ignore"):
                if cfg.omega_ref == "mean_turn_rate":
                    omega_ref = without_j(c_om, omz) / ref_cnt
                else:
                    refm = np.arctan2(without_j(csm, sim), without_j(ccm, cm))
                    omega_ref = wrap(ref0 - refm) / dt
                s_ref = without_j(c_sp, sz) / ref_cnt
            phi0 = wrap(th0[:, None, :] - ref0)
            phi1 = wrap(th1[:, None, :] - ref1)
            y_th = wrap(phi1 - phi0) / h
            x_lam = om0[:, None, :] - omega_ref
            x_soc = -phi0
            y_s = bc((s1 - s0)[:, None, :] / h)
            x_spd = s_ref - s0[:, None, :]
        yield {
            "y_th": y_th[valid],
            "x_lam": x_lam[valid],
            "x_soc": x_soc[valid],
            "y_s": y_s[valid],
            "x_spd": x_spd[valid],
        }


# --------------------------------------------------------------------------- fitting
class _Normal:
    """Running normal equations for the two regressions."""

    def __init__(self) -> None:
        self.n = 0
        self.xtx = np.zeros((2, 2))
        self.xty = np.zeros(2)
        self.sxx = 0.0
        self.sxy = 0.0

    def add(self, rows: dict) -> None:
        X = np.stack([rows["x_lam"], rows["x_soc"]], axis=1)
        self.xtx += X.T @ X
        self.xty += X.T @ rows["y_th"]
        self.sxx += float(rows["x_spd"] @ rows["x_spd"])
        self.sxy += float(rows["x_spd"] @ rows["y_s"])
        self.n += int(len(rows["y_th"]))

    def solve(self) -> dict[str, float]:
        out = {"lambda": math.nan, "k_soc": math.nan, "k_spd": math.nan}
        if self.n >= 2:
            try:
                lam, k_soc = np.linalg.solve(self.xtx, self.xty)
                out["lambda"], out["k_soc"] = float(lam), float(k_soc)
            except np.linalg.LinAlgError:
                pass
            if self.sxx > 0:
                out["k_spd"] = float(self.sxy / self.sxx)
        return out


def clip_params(raw: dict[str, float]) -> tuple[dict[str, float], bool]:
    out = {}
    clipped = False
    for k, v in raw.items():
        lo, hi = BOUNDS[k]
        c = min(max(v, lo), hi) if math.isfinite(v) else v
        clipped |= c != v
        out[k] = c
    return out, clipped


def robust_scale(e: np.ndarray) -> float:
    if e.size == 0:
        return math.nan
    med = np.median(e)
    return float(MAD_TO_SIGMA * np.median(np.abs(e - med)))


def _innovation_stats(e_th: np.ndarray, e_s: np.ndarray) -> dict:
    def stats(e: np.ndarray) -> dict:
        if e.size == 0:
            return {"n": 0}
        return {
            "n": int(e.size),
            "robust_sigma": robust_scale(e),
            "median": float(np.median(e)),
            "rms": float(np.sqrt(np.mean(e * e))),
            "p99_abs": float(np.percentile(np.abs(e), 99)),
        }

    th, sp = stats(e_th), stats(e_s)
    return {
        "n": th.get("n", 0),
        "sigma_th0": th.get("robust_sigma", math.nan),
        "sigma_s0": sp.get("robust_sigma", math.nan),
        "heading": th,
        "speed": sp,
    }


def fit_stage_p1(flights: Sequence[P.Flight], cfg: StageP1Config = StageP1Config(), label: str = "") -> dict:
    """Pooled Stage-P1 fit over the given flights (same sample period), plus per-flight numbers."""
    started = time.time()
    flights = list(flights)
    if not flights:
        raise ValueError("no flights")
    dts = {round(fl.dt, 6) for fl in flights}
    if len(dts) != 1:
        raise ValueError(f"flights mix sample periods {sorted(dts)}")
    dt = flights[0].dt
    steps = int(cfg.horizon_ticks)
    h = steps * dt
    pooled = _Normal()
    per: dict[str, _Normal] = {}
    for fl in flights:
        acc = per.setdefault(fl.flight, _Normal())
        for rows in iter_pair_rows(fl, cfg):
            pooled.add(rows)
            acc.add(rows)
    raw = pooled.solve()
    params, clipped = clip_params(raw)
    # second pass: innovations over the horizon h under the pooled (bounded) parameters
    e_th_all: list[np.ndarray] = []
    e_s_all: list[np.ndarray] = []
    per_flight: dict[str, dict] = {}
    for fl in flights:
        eth_parts, es_parts = [], []
        for rows in iter_pair_rows(fl, cfg):
            eth_parts.append(wrap(h * (rows["y_th"] - params["lambda"] * rows["x_lam"] - params["k_soc"] * rows["x_soc"])))
            es_parts.append(h * (rows["y_s"] - params["k_spd"] * rows["x_spd"]))
        eth = np.concatenate(eth_parts) if eth_parts else np.array([])
        es = np.concatenate(es_parts) if es_parts else np.array([])
        own = per[fl.flight].solve()
        per_flight[fl.flight] = {
            "flock": fl.flock,
            "type": fl.flight_type,
            "n_pairs": per[fl.flight].n,
            "lambda_own": own["lambda"],
            "k_soc_own": own["k_soc"],
            "k_spd_own": own["k_spd"],
            "sigma_th0": robust_scale(eth),
            "sigma_s0": robust_scale(es),
        }
        e_th_all.append(eth)
        e_s_all.append(es)
    innov = _innovation_stats(
        np.concatenate(e_th_all) if e_th_all else np.array([]),
        np.concatenate(e_s_all) if e_s_all else np.array([]),
    )
    return {
        "label": label,
        "config": asdict(cfg),
        "dt": dt,
        "sample_hz": round(1.0 / dt, 6),
        "horizon_ticks": steps,
        "horizon_s": round(h, 6),
        "n_flights": len(flights),
        "flights": [fl.flight for fl in flights],
        "n_pairs": pooled.n,
        "ols_unbounded": raw,
        "params": params,
        "clipped_to_bounds": clipped,
        "innovations": innov,
        "sigma_th0": innov["sigma_th0"],
        "sigma_s0": innov["sigma_s0"],
        # per-tick equivalents of a multi-tick innovation under the two scalings the original
        # might use: variance proportional to h/D (random walk) or to (h/D)^2 (smooth drift)
        "sigma_th0_per_tick_if_variance_linear_in_h": innov["sigma_th0"] / math.sqrt(steps),
        "sigma_th0_per_tick_if_linear_in_h": innov["sigma_th0"] / steps,
        "per_flight": per_flight,
        "seconds": round(time.time() - started, 2),
    }


# --------------------------------------------------------------------------- manifest
def _sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def loader_report(ds: P.Dataset) -> dict:
    summary = ds.summary()
    stats = {}
    for fid, fl in ds.flights.items():
        st = P.flight_stats(fl)
        st["velocity_self_check"] = {k: v for k, v in st["velocity_self_check"].items() if k != "per_bird"}
        stats[fid] = st
    gaps = P.measured_gap_histogram(ds.flights.values())
    return {
        "summary": summary,
        "flight_stats": stats,
        "velocity_self_check_worst_median": max(s["velocity_self_check"]["median"] for s in stats.values()),
        "velocity_self_check_worst_p95": max(s["velocity_self_check"]["p95"] for s in stats.values()),
        "measured_gap_ticks_airborne": gaps,
    }


def build_manifest(spec: TestSpec, data: dict[str, P.Dataset], sensitivity: bool = True) -> dict:
    base = StageP1Config()
    d2 = data["nagy2013"]
    d1 = data["nagy2010"]
    runs: dict[str, dict] = {}
    runs["nagy2013_10hz"] = fit_stage_p1(d2.flights.values(), base, "Nagy 2013 (D2), native 10 Hz, relative frame")
    runs["nagy2010_5hz"] = fit_stage_p1(d1.flights.values(), base, "Nagy 2010 (D1), native 5 Hz, relative frame")
    for phase in (0, 1):
        dec = [P.decimate(fl, 2, phase) for fl in d2.flights.values()]
        runs[f"nagy2013_decimated_5hz_phase{phase}"] = fit_stage_p1(
            dec, base, f"Nagy 2013 decimated to 5 Hz (every other frame, phase {phase}), relative frame"
        )
    for phase in (0, 1):
        dec = [P.decimate(fl, 2, phase) for fl in d1.flights.values()]
        runs[f"nagy2010_decimated_2p5hz_phase{phase}"] = fit_stage_p1(
            dec, base,
            f"Nagy 2010 decimated to 2.5 Hz (every other frame, phase {phase}): the 0.4 s effective tick of a "
            "tracker that sees only measured fixes, since 79 percent of its airborne fixes are 2 ticks apart",
        )
    if sensitivity:
        # horizon sweep: the original tracker treats author-interpolated samples as unsensed and
        # scores the predictor when the next measured fix arrives, i.e. over several ticks on the
        # 2010 data; these runs show what the one-step fit becomes at those horizons
        for steps in (2, 3):
            runs[f"nagy2010_horizon_{steps}_ticks"] = fit_stage_p1(
                d1.flights.values(), replace(base, horizon_ticks=steps),
                f"Nagy 2010, predictor scored {steps} ticks ({0.2 * steps:.1f} s) ahead, relative frame",
            )
        for steps in (2, 4):
            runs[f"nagy2013_horizon_{steps}_ticks"] = fit_stage_p1(
                d2.flights.values(), replace(base, horizon_ticks=steps),
                f"Nagy 2013, predictor scored {steps} ticks ({0.1 * steps:.1f} s) ahead, relative frame",
            )
        for name, ds in (("nagy2013", d2), ("nagy2010", d1)):
            runs[f"{name}_omega_ref_heading"] = fit_stage_p1(
                ds.flights.values(), replace(base, omega_ref="reference_heading"),
                f"{name}: omega_ref = turn rate of the reference heading",
            )
            runs[f"{name}_measured_only"] = fit_stage_p1(
                ds.flights.values(), replace(base, measured_only=True),
                f"{name}: neighbour ticks restricted to measured GPS fixes",
            )
            runs[f"{name}_absolute_frame"] = fit_stage_p1(
                ds.flights.values(), replace(base, frame="absolute"),
                f"{name}: absolute (ballistic) frame of the design document",
            )
    return {
        "spec_id": spec.id,
        "spec_hash": spec.frozen_hash,
        "purpose": (
            "Data-only Stage-P1 pre-checks for T05: do the predictor parameters and the heading-innovation "
            "floor re-derive from the raw archives, and is the 5x smaller floor at 10 Hz a sampling-rate "
            "effect (decimated Nagy 2013) or a dataset effect?"
        ),
        "inputs": [{"path": ds.path, "sha256": ds.sha256, "dataset": ds.name} for ds in data.values()],
        "env": isolation.env_fingerprint(),
        "loader": {name: loader_report(ds) for name, ds in data.items()},
        "stage_p1": runs,
        "original_values": ORIGINAL_VALUES,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m isrm_sandbox.data.pigeon_stage_p1", description=__doc__.split("\n\n")[0])
    ap.add_argument("--spec", default=str(DEFAULT_SPEC))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--flights", nargs="*", default=None, help="restrict to these flight ids (debugging only)")
    ap.add_argument("--no-sensitivity", action="store_true", help="skip the omega_ref, measured-only and absolute-frame variants")
    args = ap.parse_args(argv)
    spec = TestSpec.load_frozen(args.spec)
    isolation.activate()
    started = time.time()
    data = P.load_spec_inputs(spec.inputs, flights=args.flights)
    print(f"loaded {', '.join(f'{k}: {len(v.flights)} flights' for k, v in data.items())} in {time.time() - started:.1f} s")
    manifest = build_manifest(spec, data, sensitivity=not args.no_sensitivity)
    manifest["seconds_total"] = round(time.time() - started, 1)
    if args.flights:
        manifest["restricted_to_flights"] = list(args.flights)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_sanitize(manifest), indent=2, sort_keys=True), encoding="utf-8")
    for key, run in manifest["stage_p1"].items():
        p = run["params"]
        print(
            f"{key:36s} dt={run['dt']:.1f}s n={run['n_pairs']:>9d}  lambda={p['lambda']:.3f} k_soc={p['k_soc']:.3f}/s "
            f"k_spd={p['k_spd']:.3f}/s  sigma_th0={run['sigma_th0']:.5f} rad  sigma_s0={run['sigma_s0']:.3f} m/s  ({run['seconds']} s)"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
