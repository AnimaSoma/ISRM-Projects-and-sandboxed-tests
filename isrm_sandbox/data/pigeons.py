"""Independent loader for the Nagy et al. pigeon-flock GPS trajectory archives (spec T05).

Written from the archives' own readme and file headers and from the loader rules in
reports/REIMPLEMENTATION_SPECS.md (section T05). Nothing here imports or copies code
from the original Flock ISRM project; the sandbox import guard would refuse it anyway.

Archives
  nagy2010  Nagy, Akos, Biro, Vicsek (2010) Nature 464:890. pigeonflocks_trajectories.zip:
            one flock (birds A-M), 11 free flights ff1-ff11 and 4 homing flights hf1-hf4,
            0.2 s samples (20 cs grid). Members <flight>/<flight>_<bird>.txt (137 files).
  nagy2013  Nagy, Vasarhelyi, Pettit, Roberts-Mariani, Vicsek, Biro (2013) PNAS 110:13049.
            trajectoriesD2.zip ("D2"): flocks A, B, C x 5 free flights, 0.1 s samples
            (10 cs grid). Members flight/ff<F>n/ff<F>n_p<F>k (148 files; ffA1 has 8 birds).

File format (both): '#' header lines, then 11 whitespace-separated columns
  t(centisec) X(m) Y(m) Z(m) dX/dt dY/dt dZ/dt d2X/dt2 d2Y/dt2 d2Z/dt2 GPS_signal
with GPS_signal 1 for a measured fix and 0 for an author-interpolated point. Positions
are author-smoothed (centred Gaussian, sigma 0.4 s, plus cubic B-splines), so the
published velocity equals the one-sample central difference of the positions; the
loader checks this (`velocity_self_check`) and otherwise never uses the velocity columns.

Quirks handled here: in nagy2010 the last header line is not newline-terminated, so the
first data row is glued to it (the text is split once on "GPS signal"); nagy2013 rows end
with a trailing tab; some nagy2010 files use CRLF line ends; missing positions are
literal 'nan' tokens.

Cleaning rules (REIMPLEMENTATION_SPECS, T05), applied per file in this order:
  1. stamps within +/-2 cs of the grid are snapped to it; other off-grid rows are dropped;
  2. duplicated stamps keep the last occurrence in file order;
  3. rows without an x or y position are dropped (unobserved gaps);
  4. rows are sorted by stamp; frame = (t - flight t0) / grid, t0 = first stamp of any bird.
Derived masks: reliability = at least 2 measured fixes within +/-0.4 s (inclusive);
airborne = horizontal speed above 5 m/s from the central difference of the positions.
Only x and y are used; z and the published accelerations are discarded on load.
"""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from ..isolation import SandboxViolation, sha256_file, verify_inputs

N_COLUMNS = 11
COLUMNS = ("t_cs", "x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az", "gps_signal")
_SPLIT = re.compile(r"GPS[ _]signal")
_NATURAL = re.compile(r"([A-Za-z]+)(\d+)")


# --------------------------------------------------------------------------- formats
@dataclass(frozen=True)
class ArchiveFormat:
    name: str
    grid_cs: int
    member: re.Pattern
    description: str

    @property
    def dt(self) -> float:
        return self.grid_cs / 100.0


FORMATS: dict[str, ArchiveFormat] = {
    "nagy2010": ArchiveFormat(
        "nagy2010",
        20,
        re.compile(r"^pigeonflocks_trajectories/(?P<flight>[fh]f\d+)/(?P=flight)_(?P<bird>[A-Z])\.txt$"),
        "Nagy et al. 2010 (Nature): one flock, ff1-ff11 free and hf1-hf4 homing flights, 5 Hz",
    ),
    "nagy2013": ArchiveFormat(
        "nagy2013",
        10,
        re.compile(r"^flight/(?P<flight>ff[A-C]\d)/(?P=flight)_p(?P<bird>[A-C]\d)$"),
        "Nagy et al. 2013 (PNAS, ELTE D2): flocks A, B, C x 5 free flights, 10 Hz",
    ),
}


def detect_format(names: Iterable[str]) -> ArchiveFormat:
    names = list(names)
    for fmt in FORMATS.values():
        if any(fmt.member.match(n) for n in names):
            return fmt
    raise ValueError("archive matches no known pigeon trajectory format")


def _natural_key(flight_id: str) -> tuple:
    m = _NATURAL.match(flight_id)
    return (m.group(1), int(m.group(2))) if m else (flight_id, 0)


# --------------------------------------------------------------------------- parsing
_HEADER_FIELDS = {
    "date": re.compile(r"^#\s*(?:Date of measurement|date \(YYYY-MM-DD\))\s*[:=]\s*(.+?)\s*$", re.M),
    "origin": re.compile(r"\(0,0\)\s*:\s*(.+?)\s*$", re.M),
    "pigeon": re.compile(r"^#\s*(?:Pigeon|PIGEON)\s*[:=]\s*(\S+)", re.M),
    "flight_label": re.compile(r"^#\s*Flight type and number\s*:\s*(.+?)\s*$", re.M),
    "time_start_cs": re.compile(r"^#\s*timeSTART\s*=\s*(\d+)", re.M),
    "time_end_cs": re.compile(r"^#\s*timeEND\s*=\s*(\d+)", re.M),
    "gps_id": re.compile(r"^#\s*GPS\s*=\s*(\S+)", re.M),
}


def parse_header(header_text: str) -> dict:
    """Pull the documented metadata fields out of the '#' header block."""
    meta: dict = {"header_lines": header_text.count("\n") + 1}
    for key, rx in _HEADER_FIELDS.items():
        m = rx.search(header_text)
        if m:
            meta[key] = m.group(1).strip()
    for key in ("time_start_cs", "time_end_cs"):
        if key in meta:
            meta[key] = int(meta[key])
    return meta


def split_header(text: str) -> tuple[str, str]:
    """Split a file into header and data at the first 'GPS signal' / 'GPS_signal' token.

    This is the one safe rule for both archives: the nagy2010 column line is not
    newline-terminated, so the first data row follows the token directly.
    """
    m = _SPLIT.search(text)
    if m is None:
        raise ValueError("no 'GPS signal' column header found")
    return text[: m.end()], text[m.end():]


def _parse_table_slow(body: str) -> np.ndarray:
    rows = []
    for line in body.splitlines():
        tok = line.split()
        if not tok:
            continue
        if len(tok) != N_COLUMNS:
            raise ValueError(f"expected {N_COLUMNS} columns, got {len(tok)}: {line[:80]!r}")
        rows.append([float(v) for v in tok])
    return np.asarray(rows, dtype=float).reshape(-1, N_COLUMNS)


def parse_table(body: str) -> np.ndarray:
    """Parse the whitespace-separated data block into an (N, 11) float array.

    Uses the pandas C parser when available (a whitespace separator keeps the C engine
    and swallows the nagy2013 trailing tabs and CRLF ends); falls back to a strict
    pure-Python parser.
    """
    try:
        import pandas as pd

        frame = pd.read_csv(
            io.StringIO(body),
            sep=r"\s+",
            header=None,
            engine="c",
            dtype=float,
            usecols=list(range(N_COLUMNS)),
            na_values=["nan", "NaN", "NAN"],
        )
        arr = frame.to_numpy(dtype=float)
    except Exception:  # pragma: no cover - exercised only on odd files
        arr = _parse_table_slow(body)
    if arr.ndim != 2 or arr.shape[1] != N_COLUMNS:
        raise ValueError(f"table has shape {arr.shape}, expected (N, {N_COLUMNS})")
    if not np.all(np.isfinite(arr[:, 0])) or not np.all(np.isfinite(arr[:, 10])):
        raise ValueError("non-finite time stamp or GPS_signal: a row is short or corrupt")
    if not set(np.unique(arr[:, 10])) <= {0.0, 1.0}:
        raise ValueError(f"GPS_signal values other than 0/1: {np.unique(arr[:, 10])[:5]}")
    return arr


def parse_member_text(text: str) -> tuple[dict, np.ndarray]:
    header, body = split_header(text)
    return parse_header(header), parse_table(body)


# --------------------------------------------------------------------------- cleaning
@dataclass
class Cleaning:
    rows_raw: int = 0
    snapped: int = 0
    off_grid_dropped: int = 0
    duplicates_dropped: int = 0
    nan_dropped: int = 0
    rows_kept: int = 0

    def add(self, other: "Cleaning") -> None:
        for k, v in other.__dict__.items():
            setattr(self, k, getattr(self, k) + v)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def clean_table(arr: np.ndarray, grid_cs: int, snap_cs: int = 2) -> tuple[np.ndarray, Cleaning]:
    """Apply the T05 cleaning rules; returns the kept rows (stamps on the grid) and counts."""
    counts = Cleaning(rows_raw=int(arr.shape[0]))
    t = np.rint(arr[:, 0]).astype(np.int64)
    r = np.mod(t, grid_cs)
    down = (r > 0) & (r <= snap_cs)
    up = r >= grid_cs - snap_cs
    off = ~((r == 0) | down | up)
    t = np.where(down, t - r, np.where(up, t + (grid_cs - r), t))
    counts.snapped = int(down.sum() + up.sum())
    counts.off_grid_dropped = int(off.sum())
    idx = np.nonzero(~off)[0]
    # duplicated stamps: keep the LAST occurrence in file order
    rev = t[idx][::-1]
    _, first_in_rev = np.unique(rev, return_index=True)
    keep = idx[len(rev) - 1 - first_in_rev]
    counts.duplicates_dropped = int(len(idx) - len(keep))
    pos_ok = np.isfinite(arr[keep, 1]) & np.isfinite(arr[keep, 2])
    counts.nan_dropped = int((~pos_ok).sum())
    keep = keep[pos_ok]
    keep = keep[np.argsort(t[keep], kind="stable")]
    out = arr[keep].copy()
    out[:, 0] = t[keep]
    counts.rows_kept = int(len(keep))
    return out, counts


# --------------------------------------------------------------------------- tables
@dataclass
class TrackTable:
    """One parsed and cleaned file."""

    bird: str
    member: str
    header: dict
    t_cs: np.ndarray
    x: np.ndarray
    y: np.ndarray
    vx_pub: np.ndarray
    vy_pub: np.ndarray
    measured: np.ndarray
    cleaning: Cleaning


def track_from_text(text: str, bird: str, member: str, grid_cs: int) -> TrackTable:
    header, table = parse_member_text(text)
    clean, counts = clean_table(table, grid_cs)
    return TrackTable(
        bird=bird,
        member=member,
        header=header,
        t_cs=clean[:, 0].astype(np.int64),
        x=clean[:, 1],
        y=clean[:, 2],
        vx_pub=clean[:, 4],
        vy_pub=clean[:, 5],
        measured=clean[:, 10] == 1.0,
        cleaning=counts,
    )


@dataclass
class Flight:
    """All birds of one recording on a common frame grid. Arrays are [bird, frame]."""

    dataset: str
    flight: str
    flock: str
    flight_type: str
    date: str | None
    grid_cs: int
    t0_cs: int
    birds: list[str]
    x: np.ndarray
    y: np.ndarray
    vx_pub: np.ndarray
    vy_pub: np.ndarray
    measured: np.ndarray
    present: np.ndarray
    cleaning: dict[str, dict] = field(default_factory=dict)
    members: dict[str, str] = field(default_factory=dict)
    headers: dict[str, dict] = field(default_factory=dict)

    @property
    def dt(self) -> float:
        return self.grid_cs / 100.0

    @property
    def sample_hz(self) -> float:
        return 100.0 / self.grid_cs

    @property
    def n_birds(self) -> int:
        return len(self.birds)

    @property
    def n_frames(self) -> int:
        return int(self.x.shape[1])

    @property
    def span_s(self) -> float:
        return self.n_frames * self.dt

    def time_s(self) -> np.ndarray:
        return np.arange(self.n_frames) * self.dt


def _flock_and_type(fmt: ArchiveFormat, flight_id: str) -> tuple[str, str]:
    if fmt.name == "nagy2013":
        return flight_id[2], "free"
    return "D1", ("homing" if flight_id.startswith("hf") else "free")


def assemble_flight(fmt: ArchiveFormat, flight_id: str, tracks: list[TrackTable]) -> Flight:
    tracks = sorted(tracks, key=lambda tr: tr.bird)
    if not tracks:
        raise ValueError(f"{flight_id}: no tracks")
    t0 = min(int(tr.t_cs.min()) for tr in tracks)
    t1 = max(int(tr.t_cs.max()) for tr in tracks)
    n_frames = (t1 - t0) // fmt.grid_cs + 1
    shape = (len(tracks), n_frames)
    x = np.full(shape, np.nan)
    y = np.full(shape, np.nan)
    vx = np.full(shape, np.nan)
    vy = np.full(shape, np.nan)
    measured = np.zeros(shape, dtype=bool)
    present = np.zeros(shape, dtype=bool)
    for b, tr in enumerate(tracks):
        fr = (tr.t_cs - t0) // fmt.grid_cs
        x[b, fr] = tr.x
        y[b, fr] = tr.y
        vx[b, fr] = tr.vx_pub
        vy[b, fr] = tr.vy_pub
        measured[b, fr] = tr.measured
        present[b, fr] = True
    flock, ftype = _flock_and_type(fmt, flight_id)
    dates = {tr.header.get("date") for tr in tracks if tr.header.get("date")}
    return Flight(
        dataset=fmt.name,
        flight=flight_id,
        flock=flock,
        flight_type=ftype,
        date=sorted(dates)[0] if dates else None,
        grid_cs=fmt.grid_cs,
        t0_cs=t0,
        birds=[tr.bird for tr in tracks],
        x=x,
        y=y,
        vx_pub=vx,
        vy_pub=vy,
        measured=measured,
        present=present,
        cleaning={tr.bird: tr.cleaning.as_dict() for tr in tracks},
        members={tr.bird: tr.member for tr in tracks},
        headers={tr.bird: tr.header for tr in tracks},
    )


# --------------------------------------------------------------------------- datasets
@dataclass
class Dataset:
    name: str
    path: str
    sha256: str
    grid_cs: int
    flights: dict[str, Flight]
    n_members: int

    @property
    def dt(self) -> float:
        return self.grid_cs / 100.0

    def birds(self) -> list[str]:
        return sorted({b for fl in self.flights.values() for b in fl.birds})

    def cleaning_totals(self) -> Cleaning:
        tot = Cleaning()
        for fl in self.flights.values():
            for c in fl.cleaning.values():
                tot.add(Cleaning(**c))
        return tot

    def summary(self) -> dict:
        per_flight = {}
        for fid, fl in self.flights.items():
            fired = {
                b: c
                for b, c in fl.cleaning.items()
                if any(v for k, v in c.items() if k not in ("rows_raw", "rows_kept"))
            }
            per_flight[fid] = {
                "flock": fl.flock,
                "type": fl.flight_type,
                "date": fl.date,
                "birds": list(fl.birds),
                "n_birds": fl.n_birds,
                "n_frames": fl.n_frames,
                "span_s": round(fl.span_s, 1),
                "t0_cs": fl.t0_cs,
                "cleaning_fired": fired,
            }
        return {
            "dataset": self.name,
            "path": self.path,
            "sha256": self.sha256,
            "grid_cs": self.grid_cs,
            "sample_hz": 100.0 / self.grid_cs,
            "n_members": self.n_members,
            "n_flights": len(self.flights),
            "n_birds": len(self.birds()),
            "birds": self.birds(),
            "n_bird_flights": sum(fl.n_birds for fl in self.flights.values()),
            "flocks": sorted({fl.flock for fl in self.flights.values()}),
            "cleaning_totals": self.cleaning_totals().as_dict(),
            "flights": per_flight,
        }


def load_archive(
    path: str | Path,
    *,
    dataset: str | None = None,
    expected_sha256: str | None = None,
    flights: Iterable[str] | None = None,
) -> Dataset:
    """Parse every trajectory member of one archive into Flights.

    The archive is hashed first; a mismatch with `expected_sha256` raises
    SandboxViolation, so a spec's recorded input can never be silently replaced.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
        raise SandboxViolation(f"archive hash mismatch for {path}: expected {expected_sha256}, got {digest}")
    wanted = set(flights) if flights is not None else None
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        fmt = FORMATS[dataset] if dataset else detect_format(names)
        groups: dict[str, list[tuple[str, str]]] = {}
        n_members = 0
        for name in names:
            m = fmt.member.match(name)
            if not m:
                continue
            n_members += 1
            if wanted is not None and m["flight"] not in wanted:
                continue
            groups.setdefault(m["flight"], []).append((m["bird"], name))
        out: dict[str, Flight] = {}
        for flight_id in sorted(groups, key=_natural_key):
            tracks = []
            for bird, member in sorted(groups[flight_id]):
                text = zf.read(member).decode("utf-8", "replace")
                tracks.append(track_from_text(text, bird, member, fmt.grid_cs))
            out[flight_id] = assemble_flight(fmt, flight_id, tracks)
    return Dataset(fmt.name, str(path), digest, fmt.grid_cs, out, n_members)


def load_spec_inputs(inputs: Iterable, flights: Iterable[str] | None = None) -> dict[str, Dataset]:
    """Verify the spec's input references (path + SHA-256) and load each archive, keyed by format."""
    refs = [i if isinstance(i, dict) else i.__dict__ for i in inputs]
    verify_inputs(refs)
    out: dict[str, Dataset] = {}
    for ref in refs:
        ds = load_archive(ref["path"], expected_sha256=ref["sha256"], flights=flights)
        out[ds.name] = ds
    return out


# --------------------------------------------------------------------------- derived
def central_difference_velocity(flight: Flight) -> tuple[np.ndarray, np.ndarray]:
    """(P[n+1] - P[n-1]) / (2 dt); NaN at the ends and wherever a neighbour frame is absent."""
    vx = np.full_like(flight.x, np.nan)
    vy = np.full_like(flight.y, np.nan)
    h = 2.0 * flight.dt
    vx[:, 1:-1] = (flight.x[:, 2:] - flight.x[:, :-2]) / h
    vy[:, 1:-1] = (flight.y[:, 2:] - flight.y[:, :-2]) / h
    return vx, vy


def horizontal_speed(flight: Flight) -> np.ndarray:
    vx, vy = central_difference_velocity(flight)
    return np.hypot(vx, vy)


def reliability_mask(flight: Flight, window_s: float = 0.4, min_fixes: int = 2) -> np.ndarray:
    """At least `min_fixes` measured fixes within +/- window_s (inclusive) of the frame."""
    h = int(round(window_s / flight.dt))
    m = flight.measured.astype(np.int32)
    counts = np.zeros_like(m)
    for k in range(-h, h + 1):
        if k == 0:
            counts += m
        elif k > 0:
            counts[:, :-k] += m[:, k:]
        else:
            counts[:, -k:] += m[:, :k]
    return flight.present & (counts >= min_fixes)


def airborne_mask(flight: Flight, min_speed: float = 5.0) -> np.ndarray:
    speed = horizontal_speed(flight)
    with np.errstate(invalid="ignore"):
        return np.isfinite(speed) & (speed > min_speed)


def velocity_self_check(flight: Flight) -> dict:
    """Published velocity against the central difference of the published positions (m/s)."""
    vx, vy = central_difference_velocity(flight)
    mis = np.hypot(vx - flight.vx_pub, vy - flight.vy_pub)
    per_bird = {}
    pooled = []
    for b, bird in enumerate(flight.birds):
        v = mis[b][np.isfinite(mis[b])]
        if v.size:
            per_bird[bird] = {
                "n": int(v.size),
                "median": float(np.median(v)),
                "p95": float(np.percentile(v, 95)),
                "max": float(v.max()),
            }
            pooled.append(v)
    allv = np.concatenate(pooled) if pooled else np.array([])
    return {
        "n": int(allv.size),
        "median": float(np.median(allv)) if allv.size else float("nan"),
        "p95": float(np.percentile(allv, 95)) if allv.size else float("nan"),
        "per_bird": per_bird,
    }


def decimate(flight: Flight, factor: int = 2, phase: int = 0) -> Flight:
    """Keep every `factor`-th frame starting at `phase` (10 Hz -> 5 Hz with factor 2)."""
    if not (0 <= phase < factor):
        raise ValueError("phase must lie in [0, factor)")
    sl = slice(phase, None, factor)
    return Flight(
        dataset=f"{flight.dataset}_dec{factor}p{phase}",
        flight=flight.flight,
        flock=flight.flock,
        flight_type=flight.flight_type,
        date=flight.date,
        grid_cs=flight.grid_cs * factor,
        t0_cs=flight.t0_cs + phase * flight.grid_cs,
        birds=list(flight.birds),
        x=flight.x[:, sl].copy(),
        y=flight.y[:, sl].copy(),
        vx_pub=flight.vx_pub[:, sl].copy(),
        vy_pub=flight.vy_pub[:, sl].copy(),
        measured=flight.measured[:, sl].copy(),
        present=flight.present[:, sl].copy(),
        cleaning=dict(flight.cleaning),
        members=dict(flight.members),
        headers=dict(flight.headers),
    )


def measured_gap_histogram(flights: Iterable[Flight], min_speed: float = 5.0, cap: int = 4) -> dict:
    """Ticks between consecutive measured fixes, counted where the first fix is airborne and reliable.

    A tracker that ignores author-interpolated samples predicts across these gaps, so the
    histogram says how far ahead such a tracker really scores on each dataset.
    """
    counts: dict[int, int] = {}
    total = 0
    for fl in flights:
        usable = fl.present & reliability_mask(fl) & airborne_mask(fl, min_speed)
        for b in range(fl.n_birds):
            idx = np.nonzero(fl.measured[b])[0]
            if idx.size < 2:
                continue
            gaps = np.diff(idx)
            gaps = gaps[usable[b, idx[:-1]]]
            for g in gaps:
                k = int(min(g, cap))
                counts[k] = counts.get(k, 0) + 1
                total += 1
    out = {"n": total}
    for k in range(1, cap + 1):
        key = f"{k}" if k < cap else f"{cap}+"
        out[key] = counts.get(k, 0) / total if total else float("nan")
    return out


def flight_stats(flight: Flight, min_speed: float = 5.0) -> dict:
    """Data-only descriptive numbers used by the pre-check note and the protocol tests."""
    speed = horizontal_speed(flight)
    rel = reliability_mask(flight)
    air = airborne_mask(flight, min_speed)
    present = flight.present
    airborne_speeds = speed[air]
    return {
        "n_birds": flight.n_birds,
        "n_frames": flight.n_frames,
        "span_s": round(flight.span_s, 1),
        "present_fraction": float(present.mean()),
        "interpolated_fraction": float(1.0 - flight.measured[present].mean()) if present.any() else float("nan"),
        "reliable_fraction_of_present": float(rel[present].mean()) if present.any() else float("nan"),
        "airborne_fraction_of_present": float(air[present].mean()) if present.any() else float("nan"),
        "median_airborne_speed": float(np.median(airborne_speeds)) if airborne_speeds.size else float("nan"),
        "p99_airborne_speed": float(np.percentile(airborne_speeds, 99)) if airborne_speeds.size else float("nan"),
        "velocity_self_check": velocity_self_check(flight),
    }
