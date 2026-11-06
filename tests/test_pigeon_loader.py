"""Protocol tests for the independent pigeon-trajectory loader and the Stage-P1 estimator (spec T05).

Software invariants only: the archives parse, counts and cleaning actions match what the
archives and the frozen spec document, the published velocity equals the central difference
of the published positions, and the Stage-P1 estimator recovers planted parameters. Nothing
here says whether ISRM works.
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

from isrm_sandbox import isolation
from isrm_sandbox.data import pigeon_stage_p1 as SP
from isrm_sandbox.data import pigeons as P
from isrm_sandbox.spec import TestSpec

pytestmark = pytest.mark.protocol

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "specs" / "T05_flock_nagy2013_cont_vs_isrm_pe.json"

COLUMN_LINE = (
    "#t(centisec)\tX(m)\tY(m)\tZ(m)\tdX/dt(m/s)\tdY/dt(m/s)\tdZ/dt(m/s)"
    "\td^2X/dt^2(m/s^2)\td^2Y/dt^2(m/s^2)\td^2Z/dt^2(m/s^2)\t"
)
D1_HEADER = (
    "# TRAJECTORY DATA\n# Pigeon:\tA\n# Flight type and number:\tFree Flight 1\n"
    "# Date of measurement:\t26. 08. 2008.\n#\n" + COLUMN_LINE + "GPS signal"
)  # not newline-terminated: the first data row is glued to it, as in the archive
D2_HEADER = (
    "# TRAJECTORY DATA\n#date (YYYY-MM-DD) =\t2010-12-10\n#timeSTART =\t100000\n#timeEND =\t100040\n"
    "#GPS =\t12\n#PIGEON =\tA0\n#\n" + COLUMN_LINE + "GPS_signal\n"
)

# Documented per-flight bird counts of the 2010 archive (its readme: birds A-M, 7-10 per flight).
D1_BIRDS_PER_FLIGHT = {
    "ff1": 8, "ff2": 10, "ff3": 7, "ff4": 10, "ff5": 10, "ff6": 7, "ff7": 10, "ff8": 9, "ff9": 10,
    "ff10": 9, "ff11": 10, "hf1": 10, "hf2": 9, "hf3": 9, "hf4": 9,
}


def _row(t, x, y, gps=1, vx=0.0, vy=0.0):
    return [t, x, y, 0.0, vx, vy, 0.0, 0.0, 0.0, 0.0, gps]


def _d1_text(rows, eol="\n"):
    return D1_HEADER + eol.join("\t".join(str(v) for v in r) for r in rows) + eol


def _d2_text(rows):
    return D2_HEADER + "".join("\t".join(str(v) for v in r) + "\t\n" for r in rows)


def _flight(x, y, dt, measured=None, present=None, vx=None, vy=None, name="syn"):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    fl = P.Flight(
        dataset="synthetic",
        flight=name,
        flock="S",
        flight_type="free",
        date=None,
        grid_cs=int(round(dt * 100)),
        t0_cs=0,
        birds=[f"b{i}" for i in range(x.shape[0])],
        x=x,
        y=y,
        vx_pub=np.full_like(x, np.nan) if vx is None else vx,
        vy_pub=np.full_like(y, np.nan) if vy is None else vy,
        measured=np.ones(x.shape, bool) if measured is None else np.asarray(measured, bool),
        present=np.isfinite(x) if present is None else np.asarray(present, bool),
    )
    if vx is None:
        fl.vx_pub, fl.vy_pub = P.central_difference_velocity(fl)
    return fl


# --------------------------------------------------------------------------- parsing
def test_glued_header_is_split_and_crlf_tolerated():
    rows = [_row(100000, 1.0, 2.0), _row(100020, 1.1, 2.1, gps=0)]
    header, table = P.parse_member_text(_d1_text(rows, eol="\r\n"))
    assert header["pigeon"] == "A" and header["flight_label"] == "Free Flight 1"
    assert header["date"] == "26. 08. 2008."
    assert table.shape == (2, 11)
    assert table[0, 0] == 100000 and table[1, 10] == 0 and table[1, 1] == 1.1


def test_trailing_tabs_and_d2_header_fields():
    rows = [_row(100000, 1.0, 2.0), _row(100010, 1.1, 2.1), _row(100020, "nan", "nan", gps=0)]
    header, table = P.parse_member_text(_d2_text(rows))
    assert header["pigeon"] == "A0" and header["time_start_cs"] == 100000 and header["time_end_cs"] == 100040
    assert table.shape == (3, 11)
    assert np.isnan(table[2, 1]) and table[2, 0] == 100020 and table[2, 10] == 0


def test_corrupt_rows_are_refused():
    with pytest.raises(ValueError):
        P.parse_table("100000\t1.0\t2.0\n")  # short row: GPS_signal missing
    with pytest.raises(ValueError):
        P.parse_table("100000\t1\t2\t3\t4\t5\t6\t7\t8\t9\t2\n")  # GPS_signal not 0/1
    with pytest.raises(ValueError):
        P.split_header("# no column line\n1 2 3\n")


def test_cleaning_rules_in_documented_order():
    rows = [
        _row(100000, 0.0, 0.0),
        _row(100020, 1.0, 0.0),
        _row(100021, 1.5, 0.0),  # snaps onto 100020: duplicate, the LAST occurrence wins
        _row(100039, 2.0, 0.0),  # snaps up to 100040
        _row(100050, 9.0, 9.0),  # off grid by 10 cs: dropped
        _row(100060, "nan", "nan", gps=0),  # no position: dropped
        _row(100080, 4.0, 0.0),
    ]
    _, table = P.parse_member_text(_d1_text(rows))
    clean, counts = P.clean_table(table, grid_cs=20)
    assert clean[:, 0].tolist() == [100000, 100020, 100040, 100080]
    assert clean[:, 1].tolist() == [0.0, 1.5, 2.0, 4.0]
    assert counts.as_dict() == {
        "rows_raw": 7, "snapped": 2, "off_grid_dropped": 1, "duplicates_dropped": 1, "nan_dropped": 1, "rows_kept": 4,
    }


def test_assemble_aligns_birds_on_the_flight_grid():
    fmt = P.FORMATS["nagy2010"]
    a = P.track_from_text(_d1_text([_row(100000 + 20 * k, float(k), 0.0) for k in range(5)]), "A", "a", 20)
    b = P.track_from_text(_d1_text([_row(100040 + 20 * k, 10.0 + k, 0.0) for k in range(5)]), "B", "b", 20)
    fl = P.assemble_flight(fmt, "ff1", [b, a])
    assert fl.birds == ["A", "B"] and fl.t0_cs == 100000 and fl.n_frames == 7 and fl.dt == 0.2
    assert fl.present[0].tolist() == [True] * 5 + [False] * 2
    assert fl.present[1].tolist() == [False] * 2 + [True] * 5
    assert fl.x[1, 2] == 10.0 and np.isnan(fl.x[1, 0]) and fl.flight_type == "free" and fl.flock == "D1"


def test_reliability_and_airborne_masks():
    n = 9
    dt = 0.2
    fast = np.arange(n) * dt * 12.0  # 12 m/s along x
    slow = np.arange(n) * dt * 2.0
    measured = np.array([[1, 0, 0, 0, 0, 0, 0, 1, 1], [1] * n], bool)
    fl = _flight([fast, slow], [np.zeros(n), np.zeros(n)], dt, measured=measured)
    rel = P.reliability_mask(fl)  # at least 2 measured fixes within +/- 0.4 s = +/- 2 frames
    # frame 5 sees only frame 7 in [3, 7]; frame 6 sees frames 7 and 8 in [4, 8]
    assert rel[0].tolist() == [False, False, False, False, False, False, True, True, True]
    assert rel[1].all()
    air = P.airborne_mask(fl)
    assert air[0, 1:-1].all() and not air[0, 0] and not air[0, -1]  # ends have no central difference
    assert not air[1].any()
    check = P.velocity_self_check(fl)
    assert check["median"] < 1e-12  # the published velocity here IS the central difference


def test_decimate_halves_the_rate_and_recomputes_masks():
    n = 11
    x = np.arange(n) * 0.1 * 12.0
    fl = _flight([x], [np.zeros(n)], 0.1)
    d0 = P.decimate(fl, 2, 0)
    d1 = P.decimate(fl, 2, 1)
    assert d0.grid_cs == 20 and d0.n_frames == 6 and d0.t0_cs == 0
    assert d1.grid_cs == 20 and d1.n_frames == 5 and d1.t0_cs == 10
    assert np.allclose(d0.x[0], x[::2])
    speed = P.horizontal_speed(d0)
    assert np.allclose(speed[0, 1:-1], 12.0)
    with pytest.raises(ValueError):
        P.decimate(fl, 2, 2)


def test_archive_hash_is_checked_before_reading(tmp_path):
    zpath = tmp_path / "tiny.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("flight/ffA1/ffA1_pA0", _d2_text([_row(100000 + 10 * k, float(k), 0.0) for k in range(5)]))
        zf.writestr("flight/ffA1/ffA1_pA1", _d2_text([_row(100000 + 10 * k, float(k), 1.0) for k in range(5)]))
    with pytest.raises(isolation.SandboxViolation, match="hash mismatch"):
        P.load_archive(zpath, expected_sha256="0" * 64)
    ds = P.load_archive(zpath, expected_sha256=isolation.sha256_file(zpath))
    assert ds.name == "nagy2013" and list(ds.flights) == ["ffA1"] and ds.flights["ffA1"].birds == ["A0", "A1"]
    assert ds.flights["ffA1"].flock == "A" and ds.n_members == 2
    with pytest.raises(isolation.SandboxViolation):
        isolation.verify_inputs([{"path": str(zpath), "sha256": "1" * 64}])


# --------------------------------------------------------------------------- Stage-P1 estimator
def _synthetic_flock(n_birds=8, n_frames=4000, dt=0.1, lam=0.7, k_soc=0.5, k_spd=0.3, sigma_e=0.01, seed=0):
    """Headings follow the relative-frame predictor exactly plus white noise; positions are
    integrated so that backward differences return the planted headings and speeds."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames) * dt
    theta_ref = np.cumsum(0.3 * np.sin(2 * np.pi * t / 40.0)) * dt  # a common smooth turn
    phi = np.zeros((n_birds, n_frames))
    phi[:, 0] = rng.normal(0, 0.05, n_birds)
    phi[:, 1] = phi[:, 0] + rng.normal(0, 0.005, n_birds)
    for n in range(1, n_frames - 1):
        dphi = (phi[:, n] - phi[:, n - 1]) / dt
        phi[:, n + 1] = phi[:, n] + dt * (lam * dphi - k_soc * phi[:, n]) + sigma_e * rng.normal(size=n_birds)
    theta = theta_ref[None, :] + phi
    s = np.zeros((n_birds, n_frames))
    s[:, 0] = 12.0 + rng.normal(0, 0.5, n_birds)
    for n in range(n_frames - 1):
        others = (s[:, n].sum() - s[:, n]) / (n_birds - 1)
        s[:, n + 1] = s[:, n] + dt * k_spd * (others - s[:, n]) + 0.05 * rng.normal(size=n_birds)
    x = np.zeros((n_birds, n_frames))
    y = np.zeros((n_birds, n_frames))
    x[:, 0] = rng.uniform(-5, 5, n_birds)
    y[:, 0] = rng.uniform(-5, 5, n_birds)
    for n in range(1, n_frames):
        x[:, n] = x[:, n - 1] + dt * s[:, n] * np.cos(theta[:, n])
        y[:, n] = y[:, n - 1] + dt * s[:, n] * np.sin(theta[:, n])
    return _flight(x, y, dt), dict(lam=lam, k_soc=k_soc, k_spd=k_spd, sigma_e=sigma_e)


def test_stage_p1_recovers_planted_parameters():
    fl, truth = _synthetic_flock()
    res = SP.fit_stage_p1([fl], SP.StageP1Config())
    assert res["n_pairs"] > 100_000
    p = res["ols_unbounded"]
    assert abs(p["lambda"] - truth["lam"]) < 0.05, p
    assert abs(p["k_soc"] - truth["k_soc"]) < 0.10, p
    assert abs(p["k_spd"] - truth["k_spd"]) < 0.05, p
    # innovation of the pair = e_j - mean of 7 others: sd = sigma_e * sqrt(1 + 1/7)
    expected_sigma = truth["sigma_e"] * np.sqrt(1 + 1 / 7)
    assert abs(res["sigma_th0"] / expected_sigma - 1) < 0.2, res["sigma_th0"]
    assert not res["clipped_to_bounds"]
    # the other omega_ref definition agrees on this smooth reference
    res2 = SP.fit_stage_p1([fl], SP.StageP1Config(omega_ref="reference_heading"))
    assert abs(res2["ols_unbounded"]["lambda"] - truth["lam"]) < 0.05
    assert abs(res2["ols_unbounded"]["k_soc"] - truth["k_soc"]) < 0.10


def test_stage_p1_variants_run_and_mixed_rates_are_refused():
    fl, _ = _synthetic_flock(n_frames=800)
    for cfg in (SP.StageP1Config(frame="absolute"), SP.StageP1Config(measured_only=True)):
        res = SP.fit_stage_p1([fl], cfg)
        assert res["n_pairs"] > 0 and np.isfinite(res["sigma_th0"]) and np.isfinite(res["params"]["lambda"])
    with pytest.raises(ValueError, match="sample periods"):
        SP.fit_stage_p1([fl, P.decimate(fl, 2, 0)], SP.StageP1Config())
    with pytest.raises(ValueError):
        SP.StageP1Config(frame="sideways")


def test_decimation_keeps_the_estimator_consistent():
    fl, truth = _synthetic_flock(n_frames=6000)
    dec = P.decimate(fl, 2, 0)
    res = SP.fit_stage_p1([dec], SP.StageP1Config())
    assert res["dt"] == 0.2 and res["n_pairs"] > 0
    # at half the rate the planted AR(2) is no longer exactly one-step, so only the sign and
    # bounds are checked here; the heading innovation must grow with the step
    full = SP.fit_stage_p1([fl], SP.StageP1Config())
    assert res["sigma_th0"] > full["sigma_th0"]
    assert 0.0 <= res["params"]["lambda"] <= 1.0


def test_horizon_sweep_scores_further_ahead():
    fl, _ = _synthetic_flock(n_frames=3000)
    one = SP.fit_stage_p1([fl], SP.StageP1Config())
    two = SP.fit_stage_p1([fl], SP.StageP1Config(horizon_ticks=2))
    assert one["horizon_ticks"] == 1 and two["horizon_s"] == pytest.approx(0.2)
    assert two["n_pairs"] < one["n_pairs"]  # one more eligible frame is needed per row
    assert two["sigma_th0"] > one["sigma_th0"]  # noise accumulates over the longer horizon
    assert two["sigma_th0_per_tick_if_variance_linear_in_h"] == pytest.approx(two["sigma_th0"] / np.sqrt(2))
    assert two["sigma_th0_per_tick_if_linear_in_h"] == pytest.approx(two["sigma_th0"] / 2)
    with pytest.raises(ValueError):
        SP.StageP1Config(horizon_ticks=0)


def test_measured_gap_histogram_counts_airborne_gaps():
    n = 40
    dt = 0.2
    x = np.arange(n) * dt * 12.0
    measured = np.zeros((1, n), bool)
    measured[0, ::2] = True  # alternating fixes, the common pattern of the 2010 archive
    fl = _flight([x], [np.zeros(n)], dt, measured=measured)
    hist = P.measured_gap_histogram([fl])
    assert hist["n"] > 0 and hist["2"] == pytest.approx(1.0) and hist["1"] == 0.0 and hist["4+"] == 0.0
    dense = _flight([x], [np.zeros(n)], dt)
    assert P.measured_gap_histogram([dense])["1"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- real archives
@pytest.fixture(scope="module")
def spec():
    return TestSpec.load_frozen(SPEC_PATH)


@pytest.fixture(scope="module")
def input_refs(spec):
    refs = {Path(i.path).name: dict(i.__dict__) for i in spec.inputs}
    missing = [r["path"] for r in refs.values() if not os.path.exists(r["path"])]
    if missing:
        pytest.skip(f"T05 input archives not present on this machine: {missing}")
    isolation.verify_inputs(list(refs.values()))
    return refs


@pytest.fixture(scope="module")
def d2(input_refs):
    r = input_refs["trajectoriesD2.zip"]
    return P.load_archive(r["path"], expected_sha256=r["sha256"])


@pytest.fixture(scope="module")
def d1(input_refs):
    r = input_refs["pigeonflocks_trajectories.zip"]
    return P.load_archive(r["path"], expected_sha256=r["sha256"])


def test_spec_inputs_load_by_format(spec, input_refs):
    data = P.load_spec_inputs(spec.inputs, flights=["ffA1", "ff11"])
    assert set(data) == {"nagy2013", "nagy2010"}
    assert list(data["nagy2013"].flights) == ["ffA1"] and list(data["nagy2010"].flights) == ["ff11"]


def test_nagy2013_counts_match_the_spec(spec, d2):
    assert d2.name == "nagy2013" and d2.grid_cs == 10 and d2.n_members == 148
    assert len(d2.flights) == 15
    flocks = {fid: fl.flock for fid, fl in d2.flights.items()}
    assert sorted(set(flocks.values())) == ["A", "B", "C"]
    assert all(sum(1 for f in flocks.values() if f == k) == 5 for k in "ABC")
    assert d2.birds() == [f"{f}{k}" for f in "ABC" for k in range(10)]
    absent_total = 0
    for fid, fl in d2.flights.items():
        if fid == "ffA1":
            assert fl.birds == ["A0", "A1", "A2", "A3", "A5", "A6", "A7", "A9"]
        else:
            assert fl.n_birds == 10
        assert fl.flight_type == "free" and fl.flock == fid[2] and all(b[0] == fl.flock for b in fl.birds)
        # every file is a dense 0.1 s series without interior gaps; a few birds end one tick
        # before their flockmates (A4 in ffA2, seven birds in ffC5), so only the last frame
        # of a flight grid may be absent
        for b, bird in enumerate(fl.birds):
            idx = np.nonzero(fl.present[b])[0]
            assert idx[-1] - idx[0] + 1 == idx.size, f"{fid}/{bird}: interior gap"
        assert fl.present[:, :-1].all(), fid
        absent_total += int((~fl.present).sum())
        for bird, h in fl.headers.items():
            assert 0 <= fl.t0_cs - h["time_start_cs"] <= fl.grid_cs, (fid, bird)  # ffC4 starts one tick late
            assert fl.n_frames == (h["time_end_cs"] - fl.t0_cs) // fl.grid_cs + 1, (fid, bird)
    assert absent_total == 8
    tot = d2.cleaning_totals().as_dict()
    assert tot["rows_raw"] == tot["rows_kept"] == 3_282_222
    assert tot["snapped"] == tot["off_grid_dropped"] == tot["duplicates_dropped"] == tot["nan_dropped"] == 0
    assert "3 flocks x 10 birds, 15 flights, 10 Hz" in spec.inputs[0].description


def test_nagy2010_counts_and_documented_cleaning(spec, d1):
    assert d1.name == "nagy2010" and d1.grid_cs == 20 and d1.n_members == 137
    assert list(d1.flights) == [f"ff{k}" for k in range(1, 12)] + [f"hf{k}" for k in range(1, 5)]
    assert d1.birds() == list("ABCDEFGHIJKLM")
    assert {fid: fl.n_birds for fid, fl in d1.flights.items()} == D1_BIRDS_PER_FLIGHT
    assert {fl.flight_type for fid, fl in d1.flights.items() if fid.startswith("hf")} == {"homing"}
    assert {fl.flight_type for fid, fl in d1.flights.items() if fid.startswith("ff")} == {"free"}
    tot = d1.cleaning_totals().as_dict()
    assert tot == {
        "rows_raw": 2_426_294,
        "snapped": 83,
        "off_grid_dropped": 0,
        "duplicates_dropped": 13_969,
        "nan_dropped": 22_036,
        "rows_kept": 2_390_289,
    }
    # the archive's known artefacts, each handled by exactly the documented rule
    assert d1.flights["ff2"].cleaning["C"]["duplicates_dropped"] == 13_877  # series held twice
    assert d1.flights["hf1"].cleaning["H"]["duplicates_dropped"] == 92  # out-of-order opening block
    assert d1.flights["ff7"].cleaning["C"]["snapped"] == 75  # stamps 1 cs off the grid
    assert d1.flights["ff7"].cleaning["J"]["nan_dropped"] == 8_181  # the largest unobserved gap
    assert "1 flock, 15 flights, 5 Hz" in spec.inputs[1].description


def test_published_velocity_is_the_central_difference(d1, d2):
    worst = {}
    for ds in (d1, d2):
        for fid, fl in ds.flights.items():
            chk = P.velocity_self_check(fl)
            assert chk["n"] > 1000
            assert chk["median"] < 5e-4, (ds.name, fid, chk["median"])
            assert chk["p95"] < 2e-3, (ds.name, fid, chk["p95"])
            worst[ds.name] = max(worst.get(ds.name, 0.0), chk["median"])
    # the loader rule says about 0.0001 m/s (positions are printed to 0.1 mm)
    assert worst["nagy2010"] < 2e-4 and worst["nagy2013"] < 3e-4, worst


def test_airborne_segments_are_plausible(d1, d2):
    for ds in (d1, d2):
        for fid, fl in ds.flights.items():
            st = P.flight_stats(fl)
            assert 0.1 < st["airborne_fraction_of_present"] < 0.7, (ds.name, fid, st["airborne_fraction_of_present"])
            assert 8.0 < st["median_airborne_speed"] < 20.0, (ds.name, fid, st["median_airborne_speed"])
            assert 0.6 < st["reliable_fraction_of_present"] <= 1.0, (ds.name, fid)


def test_stage_p1_runs_on_a_real_flight(d2):
    res = SP.fit_stage_p1([d2.flights["ffA1"]], SP.StageP1Config())
    assert res["n_pairs"] > 100_000 and res["dt"] == 0.1
    for k in ("lambda", "k_soc", "k_spd"):
        assert np.isfinite(res["params"][k])
    assert 1e-4 < res["sigma_th0"] < 0.1 and 1e-3 < res["sigma_s0"] < 2.0
    assert set(res["per_flight"]) == {"ffA1"}
