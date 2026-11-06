"""Two-dimensional ball-avoidance arena for T04, re-created independently from the rules
of the Total ISRM v3 benchmark (D:/Total ISRM/benchmarks/isrm-v3-2026-10-13: REPORT.txt,
protocol.json constants, SOURCE_MAPPING.txt, the engine's documented rules, and
MODEL_SPEC.txt of the application). No code is imported or copied; the hazard stream,
integration and accounting below are the sandbox's own.

World (exogenous, shared by every arm of a replicate)
----------------------------------------------------
* Circular arena of radius 280. N hazards (balls) with radii 7..13 start 90..245 units
  from the centre heading roughly inward, at speed s (0.72..1.28) where s is the cell's
  nominal speed. Each ball carries a turn timer (0.9 + 2.6 U s initially, 0.8 + 2.4 U s
  afterwards); when it fires the velocity is rotated by (U - 0.5) 2 pi x volatility.
  Balls reflect specularly off the wall. The whole 120 s hazard trajectory is drawn once
  per replicate at 60 Hz from the replicate's generator, so every arm sees the same balls
  and no controller can perturb the stream.
* Stationary grid (72 cells): hazards {4, 12, 26} x speed {40, 100, 180} x volatility
  {0, 0.15, 0.45, 0.9} x initial energy {20, 100}; a replicate draws its cell uniformly.

Agent (one per arm, independent)
-------------------------------
* Radius 9, starts at the centre at rest. A command is a target velocity; the agent
  accelerates toward it at 320 units/s^2, capped at speed 115, and is clamped at the wall.
* Energy: basal 0.19/s, motion 1.9 (speed/115)^2 per s, 9 per collision with a 0.65 s
  cooldown, plus every paid observation, planning pass and model operation its controller
  requests. Depleted (energy 0) means dead; survival is the time alive, capped at 120 s.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DT = 1.0 / 60.0
PHYS = {
    "radius": 280.0,
    "agent_radius": 9.0,
    "max_speed": 115.0,
    "accel": 320.0,
    "hit_cost": 9.0,
    "cooldown": 0.65,
    "basal": 0.19,
    "motion": 1.9,
    "horizon": 0.9,
    "limit": 120.0,
}
GRID = {"count": (4, 12, 26), "speed": (40.0, 100.0, 180.0), "volatility": (0.0, 0.15, 0.45, 0.9), "energy": (20.0, 100.0)}
TICKS = int(round(PHYS["limit"] / DT))


@dataclass(frozen=True)
class Cell:
    count: int
    speed: float
    volatility: float
    energy: float


def all_cells() -> list[Cell]:
    return [Cell(c, s, v, e) for c in GRID["count"] for s in GRID["speed"] for v in GRID["volatility"] for e in GRID["energy"]]


def draw_cell(rng: np.random.Generator) -> Cell:
    cells = all_cells()
    return cells[int(rng.integers(len(cells)))]


def reflect(px, py, vx, vy, limit):
    """Specular reflection of bodies that crossed the circular wall (vectorised)."""
    d = np.hypot(px, py)
    out = d > limit
    if np.any(out):
        nx, ny = px[out] / d[out], py[out] / d[out]
        vn = vx[out] * nx + vy[out] * ny
        bounce = vn > 0
        vx_o, vy_o = vx[out].copy(), vy[out].copy()
        vx_o[bounce] -= 2 * vn[bounce] * nx[bounce]
        vy_o[bounce] -= 2 * vn[bounce] * ny[bounce]
        vx[out], vy[out] = vx_o, vy_o
        px[out], py[out] = nx * (limit[out] - 1e-6), ny * (limit[out] - 1e-6)
    return px, py, vx, vy


def make_world(rng: np.random.Generator, cell: Cell | None = None) -> dict:
    """Draw the cell (unless given) and the full exogenous hazard trajectory."""
    if cell is None:
        cell = draw_cell(rng)
    N, T = cell.count, TICKS
    angle = rng.random(N) * 2 * np.pi
    dist = 90.0 + rng.random(N) * 155.0
    r = 7.0 + rng.random(N) * 6.0
    heading = angle + np.pi + (rng.random(N) - 0.5) * 1.4
    sp = cell.speed * (0.72 + rng.random(N) * 0.56)
    turn_in = 0.9 + rng.random(N) * 2.6
    px, py = np.cos(angle) * dist, np.sin(angle) * dist
    vx, vy = np.cos(heading) * sp, np.sin(heading) * sp
    limit = PHYS["radius"] - r
    H = np.empty((T + 1, N, 4))
    H[0] = np.column_stack([px, py, vx, vy])
    for t in range(1, T + 1):
        turn_in -= DT
        fire = np.flatnonzero(turn_in <= 0)
        for i in fire:
            turn_in[i] = 0.8 + rng.random() * 2.4
            a = (rng.random() - 0.5) * 2 * np.pi * cell.volatility
            c, s = np.cos(a), np.sin(a)
            vx[i], vy[i] = vx[i] * c - vy[i] * s, vx[i] * s + vy[i] * c
        px = px + vx * DT
        py = py + vy * DT
        px, py, vx, vy = reflect(px, py, vx, vy, limit)
        H[t, :, 0], H[t, :, 1], H[t, :, 2], H[t, :, 3] = px, py, vx, vy
    return {"cell": cell, "H": H, "r": r, "T": T}


# --------------------------------------------------------------------------- agent physics
def integrate_agent(x, y, vx, vy, tx, ty, dt=DT):
    """One tick of acceleration-limited pursuit of a target velocity, clamped at the wall."""
    dx, dy = tx - vx, ty - vy
    n = np.hypot(dx, dy)
    f = min(1.0, PHYS["accel"] * dt / n) if n > 0 else 0.0
    vx, vy = vx + dx * f, vy + dy * f
    x, y = x + vx * dt, y + vy * dt
    d = np.hypot(x, y)
    limit = PHYS["radius"] - PHYS["agent_radius"]
    if d > limit:
        nx, ny = x / d, y / d
        x, y = nx * limit, ny * limit
        radial = max(0.0, vx * nx + vy * ny)
        vx, vy = vx - radial * nx, vy - radial * ny
    return x, y, vx, vy


def run_agent(world: dict, controller) -> dict:
    """Drive one controller through the world; returns survival and diagnostics.

    The controller exposes reset(cell) and decide(tick, hazards, radii, agent) -> (tx, ty),
    and may debit agent["energy"] for its paid operations.
    """
    H, r, T, cell = world["H"], world["r"], world["T"], world["cell"]
    agent = {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "energy": float(cell.energy), "hits": 0, "cooldown": 0.0, "alive": True, "lifetime": 0.0,
             "movement_cost": 0.0, "basal_cost": 0.0, "collision_cost": 0.0}
    controller.reset(cell)
    tx, ty = controller.decide(0, H[0], r, agent)
    for t in range(1, T + 1):
        agent["cooldown"] = max(0.0, agent["cooldown"] - DT)
        x, y, vx, vy = integrate_agent(agent["x"], agent["y"], agent["vx"], agent["vy"], tx, ty)
        agent.update(x=x, y=y, vx=vx, vy=vy)
        motion = PHYS["motion"] * (np.hypot(vx, vy) / PHYS["max_speed"]) ** 2 * DT
        basal = PHYS["basal"] * DT
        agent["energy"] -= motion + basal
        agent["movement_cost"] += motion
        agent["basal_cost"] += basal
        if agent["cooldown"] <= 0 and agent["energy"] > 0:
            gap = np.hypot(H[t, :, 0] - x, H[t, :, 1] - y) - r - PHYS["agent_radius"]
            if np.any(gap < 0):
                agent["hits"] += 1
                agent["energy"] -= PHYS["hit_cost"]
                agent["collision_cost"] += PHYS["hit_cost"]
                agent["cooldown"] = PHYS["cooldown"]
        agent["lifetime"] = t * DT
        if agent["energy"] <= 0:
            agent["energy"] = 0.0
            agent["alive"] = False
            break
        if t % 6 == 0 and t < T:
            tx, ty = controller.decide(t, H[t], r, agent)
    out = {
        "survival": agent["lifetime"] if not agent["alive"] else PHYS["limit"],
        "alive": agent["alive"],
        "terminal_energy": agent["energy"],
        "hits": agent["hits"],
        "movement_cost": agent["movement_cost"],
        "collision_cost": agent["collision_cost"],
    }
    out.update(controller.diagnostics())
    return out
