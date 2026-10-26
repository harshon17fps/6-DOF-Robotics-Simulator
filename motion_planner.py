"""
motion_planner.py — Collision-free 3D path & trajectory planning.

Pipeline
--------
1. ``RRTPlanner.plan``       Rapidly-exploring Random Tree with goal biasing
                             finds *a* collision-free path through 3D space.
2. ``RRTPlanner.shortcut``   Greedy line-of-sight shortcutting removes the
                             random zig-zags typical of raw RRT output.
3. ``smooth_path``           A C2-continuous cubic spline is fitted through
                             the waypoints and re-validated; if it clips an
                             obstacle the control polygon is refined until the
                             curve is provably clear.
4. ``minimum_jerk_trajectory``  Time-parameterises the curve with a quintic
                             (minimum-jerk) profile so motion starts and stops
                             with zero velocity and zero acceleration.

Collision model
---------------
Obstacles expose a vectorised signed-distance function (SDF). Because any SDF
is 1-Lipschitz, sampling a segment every ``h`` metres and requiring
SDF >= clearance at each sample guarantees the *continuous* segment stays at
least ``clearance - h/2`` away from every obstacle. The environment therefore
checks segments with ``h = clearance / 2`` by default, so no "tunnelling"
through thin obstacles is possible.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Sequence, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import CubicSpline

FloatArray: TypeAlias = NDArray[np.float64]
StateValidator: TypeAlias = Callable[[FloatArray], bool]

__all__ = [
    "Obstacle",
    "SphereObstacle",
    "BoxObstacle",
    "Environment",
    "RRTConfig",
    "PlanningResult",
    "RRTPlanner",
    "Trajectory",
    "smooth_path",
    "minimum_jerk_trajectory",
    "path_length",
    "densify",
    "duration_for_peak_speed",
]


# --------------------------------------------------------------------------- #
# Obstacles
# --------------------------------------------------------------------------- #
class Obstacle(ABC):
    """A static 3D obstacle described by its signed-distance function."""

    @abstractmethod
    def signed_distance(self, points: FloatArray) -> FloatArray:
        """Signed distance from each point to the surface.

        Args:
            points: Array of shape (N, 3).

        Returns:
            Array of shape (N,): positive outside, zero on the surface,
            negative inside.
        """


@dataclass(frozen=True)
class SphereObstacle(Obstacle):
    """Sphere with a centre (x, y, z) and a radius [m]."""

    center: tuple[float, float, float]
    radius: float

    def signed_distance(self, points: FloatArray) -> FloatArray:
        return np.linalg.norm(points - np.asarray(self.center), axis=1) - self.radius


@dataclass(frozen=True)
class BoxObstacle(Obstacle):
    """Axis-aligned box given by its centre and half-extents [m]."""

    center: tuple[float, float, float]
    half_extents: tuple[float, float, float]

    def signed_distance(self, points: FloatArray) -> FloatArray:
        # Exact box SDF: q = |p - c| - h; outside part + inside part.
        q = np.abs(points - np.asarray(self.center)) - np.asarray(self.half_extents)
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside


# --------------------------------------------------------------------------- #
# Environment (workspace bounds + obstacles + safety margin)
# --------------------------------------------------------------------------- #
@dataclass
class Environment:
    """Axis-aligned workspace with obstacles and a safety clearance.

    Attributes:
        lower: Lower corner (x, y, z) of the sampling/validity box [m].
        upper: Upper corner (x, y, z) of the sampling/validity box [m].
        obstacles: Static obstacles.
        clearance: Minimum distance any valid point must keep from every
            obstacle surface [m] (models the gripper's physical size).
        resolution: Sampling step for segment checks [m]; defaults to
            clearance / 2 (see module docstring for the guarantee).
    """

    lower: FloatArray
    upper: FloatArray
    obstacles: Sequence[Obstacle] = field(default_factory=list)
    clearance: float = 0.05
    resolution: float | None = None

    def __post_init__(self) -> None:
        self.lower = np.asarray(self.lower, dtype=float)
        self.upper = np.asarray(self.upper, dtype=float)
        if self.lower.shape != (3,) or self.upper.shape != (3,):
            raise ValueError("Bounds must be 3-vectors.")
        if np.any(self.lower >= self.upper):
            raise ValueError("Every lower bound must be below its upper bound.")
        if self.clearance < 0:
            raise ValueError("clearance must be non-negative.")
        if self.resolution is None:
            self.resolution = max(self.clearance / 2.0, 1e-3)

    def distance(self, points: ArrayLike) -> FloatArray:
        """Distance from each point to the nearest obstacle (inf if none)."""
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        if not self.obstacles:
            return np.full(len(pts), np.inf)
        return np.min([obs.signed_distance(pts) for obs in self.obstacles], axis=0)

    def in_bounds(self, points: ArrayLike) -> NDArray[np.bool_]:
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        return np.all((pts >= self.lower) & (pts <= self.upper), axis=1)

    def is_point_free(self, point: ArrayLike) -> bool:
        """True if the point is inside the workspace and clear of obstacles."""
        p = np.atleast_2d(point)
        return bool(self.in_bounds(p)[0] and self.distance(p)[0] >= self.clearance)

    def is_segment_free(self, p0: ArrayLike, p1: ArrayLike) -> bool:
        """True if the straight segment p0 -> p1 is collision-free."""
        return self.is_polyline_free(np.vstack([p0, p1]))

    def is_polyline_free(self, points: ArrayLike) -> bool:
        """True if every segment of a polyline is collision-free."""
        dense = densify(np.asarray(points, dtype=float), self.resolution)
        return bool(
            np.all(self.in_bounds(dense)) and np.all(self.distance(dense) >= self.clearance)
        )

    def polyline_clearance(self, points: ArrayLike, resolution: float | None = None) -> float:
        """Minimum obstacle distance along a polyline (e.g. an arm skeleton)."""
        dense = densify(np.asarray(points, dtype=float), resolution or self.resolution)
        return float(np.min(self.distance(dense)))


# --------------------------------------------------------------------------- #
# RRT planner
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RRTConfig:
    """Tunable RRT parameters.

    Attributes:
        step_size: Maximum edge length when extending the tree [m].
        goal_bias: Probability of sampling the goal instead of a random point.
        max_iterations: Sampling budget before declaring failure.
        goal_tolerance: Distance at which a direct connection to the goal is
            attempted [m]; defaults to ``step_size``.
        shortcut: Apply greedy shortcutting to the raw tree path.
    """

    step_size: float = 0.04
    goal_bias: float = 0.10
    max_iterations: int = 8000
    goal_tolerance: float | None = None
    shortcut: bool = True


@dataclass(frozen=True)
class PlanningResult:
    """Everything the planner produced, including the tree for visualisation."""

    success: bool
    path: FloatArray          # final waypoints (shortcut if enabled), (K, 3)
    raw_path: FloatArray      # unprocessed tree path, (K0, 3)
    tree_nodes: FloatArray    # (M, 3)
    tree_parents: NDArray[np.int64]  # (M,), -1 for the root
    iterations: int
    planning_time: float
    message: str

    @property
    def path_length(self) -> float:
        return path_length(self.path)


class RRTPlanner:
    """Goal-biased Rapidly-exploring Random Tree in 3D Cartesian space.

    The tree is stored in pre-allocated NumPy arrays and nearest-neighbour
    queries are a single vectorised distance computation, which keeps the
    planner fast without external spatial-index dependencies.

    Args:
        environment: Workspace bounds, obstacles and clearance.
        config: Algorithm parameters.
        state_validator: Optional extra predicate every waypoint must satisfy
            (e.g. "the arm can reach this point"), which keeps the planner
            decoupled from any specific robot model.
        seed: Seed for reproducible sampling.
    """

    def __init__(
        self,
        environment: Environment,
        config: RRTConfig | None = None,
        state_validator: StateValidator | None = None,
        seed: int | None = None,
    ) -> None:
        self.env = environment
        self.config = config or RRTConfig()
        self.state_validator = state_validator
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ API
    def plan(self, start: ArrayLike, goal: ArrayLike) -> PlanningResult:
        """Search for a collision-free path from ``start`` to ``goal``."""
        t0 = time.perf_counter()
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        for name, p in (("start", start), ("goal", goal)):
            if not self._is_state_valid(p):
                return self._failure(start, f"The {name} position {p} is in collision "
                                     "or outside the workspace.", t0)

        cfg = self.config
        goal_tol = cfg.goal_tolerance if cfg.goal_tolerance is not None else cfg.step_size

        # Pre-allocated tree storage (+2: root and final goal node).
        nodes = np.empty((cfg.max_iterations + 2, 3))
        parents = np.full(cfg.max_iterations + 2, -1, dtype=np.int64)
        nodes[0] = start
        n_nodes = 1

        # Trivial case: straight line of sight.
        if self._is_edge_valid(start, goal):
            nodes[1], parents[1] = goal, 0
            return self._success(nodes[:2], parents[:2], 1, 0, t0)

        for it in range(1, cfg.max_iterations + 1):
            sample = (goal if self.rng.random() < cfg.goal_bias
                      else self.rng.uniform(self.env.lower, self.env.upper))

            # Nearest neighbour: one vectorised squared-distance pass.
            diff = nodes[:n_nodes] - sample
            nearest = int(np.argmin(np.einsum("ij,ij->i", diff, diff)))

            # Steer: move at most one step from the nearest node towards sample.
            direction = sample - nodes[nearest]
            dist = float(np.linalg.norm(direction))
            if dist < 1e-9:
                continue
            new = nodes[nearest] + direction * min(1.0, cfg.step_size / dist)
            if not self._is_edge_valid(nodes[nearest], new):
                continue

            nodes[n_nodes], parents[n_nodes] = new, nearest
            n_nodes += 1

            # Goal connection attempt.
            if np.linalg.norm(goal - new) <= goal_tol and self._is_edge_valid(new, goal):
                nodes[n_nodes], parents[n_nodes] = goal, n_nodes - 1
                n_nodes += 1
                return self._success(nodes[:n_nodes], parents[:n_nodes],
                                     n_nodes - 1, it, t0)

        tree_nodes, tree_parents = nodes[:n_nodes].copy(), parents[:n_nodes].copy()
        return PlanningResult(
            success=False, path=np.empty((0, 3)), raw_path=np.empty((0, 3)),
            tree_nodes=tree_nodes, tree_parents=tree_parents,
            iterations=cfg.max_iterations, planning_time=time.perf_counter() - t0,
            message=f"No path found within {cfg.max_iterations} iterations.",
        )

    def shortcut(self, path: FloatArray) -> FloatArray:
        """Greedy line-of-sight shortcutting.

        From each waypoint, jump to the furthest later waypoint that is
        directly visible. O(K^2) segment checks in the worst case, and it
        never increases path length (triangle inequality).
        """
        if len(path) <= 2:
            return path.copy()
        result = [path[0]]
        i = 0
        last = len(path) - 1
        while i < last:
            j = last
            while j > i + 1 and not self._is_edge_valid(path[i], path[j]):
                j -= 1
            result.append(path[j])
            i = j
        return np.asarray(result)

    # -------------------------------------------------------------- helpers
    def _is_state_valid(self, p: FloatArray) -> bool:
        if not self.env.is_point_free(p):
            return False
        return self.state_validator is None or self.state_validator(p)

    def _is_edge_valid(self, p0: FloatArray, p1: FloatArray) -> bool:
        if not self.env.is_segment_free(p0, p1):
            return False
        if self.state_validator is None:
            return True
        # The validator is evaluated every step_size along the edge.
        n = max(1, int(np.ceil(np.linalg.norm(p1 - p0) / self.config.step_size)))
        return all(self.state_validator(p0 + (p1 - p0) * t)
                   for t in np.linspace(0.0, 1.0, n + 1)[1:])

    def _success(self, nodes: FloatArray, parents: NDArray[np.int64],
                 goal_index: int, iterations: int, t0: float) -> PlanningResult:
        raw = _backtrack(nodes, parents, goal_index)
        path = self.shortcut(raw) if self.config.shortcut else raw.copy()
        return PlanningResult(
            success=True, path=path, raw_path=raw,
            tree_nodes=nodes.copy(), tree_parents=parents.copy(),
            iterations=iterations, planning_time=time.perf_counter() - t0,
            message=f"Path found after {iterations} iterations "
                    f"({len(nodes)} tree nodes).",
        )

    @staticmethod
    def _failure(start: FloatArray, message: str, t0: float) -> PlanningResult:
        return PlanningResult(
            success=False, path=np.empty((0, 3)), raw_path=np.empty((0, 3)),
            tree_nodes=start.reshape(1, 3), tree_parents=np.array([-1]),
            iterations=0, planning_time=time.perf_counter() - t0, message=message,
        )


def _backtrack(nodes: FloatArray, parents: NDArray[np.int64], index: int) -> FloatArray:
    """Follow parent pointers from ``index`` back to the root."""
    chain = []
    while index != -1:
        chain.append(index)
        index = int(parents[index])
    return nodes[chain[::-1]].copy()


# --------------------------------------------------------------------------- #
# Path post-processing & time parameterisation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Trajectory:
    """Time-parameterised Cartesian trajectory.

    Attributes:
        times: Sample times, shape (N,) [s].
        positions: Positions, shape (N, 3) [m].
        speeds: Path speed |v| at each sample, shape (N,) [m/s].
        length: Total path length [m].
    """

    times: FloatArray
    positions: FloatArray
    speeds: FloatArray
    length: float

    @property
    def duration(self) -> float:
        return float(self.times[-1])


def path_length(path: ArrayLike) -> float:
    """Total Euclidean length of a polyline."""
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def densify(path: FloatArray, resolution: float) -> FloatArray:
    """Resample a polyline so that consecutive points are <= resolution apart.

    All original vertices are preserved; the operation is fully vectorised.
    """
    path = np.atleast_2d(path)
    if len(path) < 2:
        return path.copy()
    seg = np.diff(path, axis=0)
    counts = np.maximum(1, np.ceil(np.linalg.norm(seg, axis=1) / resolution)).astype(int)
    # Parameter t in [0, 1) for each sub-sample of each segment.
    seg_index = np.repeat(np.arange(len(seg)), counts)
    t = (np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)) / \
        np.repeat(counts, counts)
    dense = path[seg_index] + seg[seg_index] * t[:, None]
    return np.vstack([dense, path[-1]])


def smooth_path(
    waypoints: ArrayLike,
    environment: Environment,
    *,
    state_validator: StateValidator | None = None,
    validator_spacing: float = 0.02,
    samples_per_meter: int = 400,
    max_refinements: int = 6,
) -> FloatArray:
    """Fit a collision-checked C2 cubic spline through polyline waypoints.

    The spline is parameterised by cumulative chord length. If the curve
    violates the clearance anywhere (splines can overshoot at sharp corners),
    the control polygon is subdivided, which pulls the spline closer to the
    known-safe polyline, and the fit is repeated. After ``max_refinements``
    failed attempts the densified polyline itself is returned, so the result
    is always collision-free when the input polyline is.

    Args:
        waypoints: Collision-free polyline, shape (K, 3).
        environment: Obstacles and clearance used to validate the curve.
        state_validator: Optional extra predicate (e.g. IK reachability),
            evaluated every ``validator_spacing`` metres along the curve.
        validator_spacing: Arc-length spacing of validator calls [m].
        samples_per_meter: Output sampling density.
        max_refinements: Subdivision rounds before falling back.

    Returns:
        Dense array of shape (N, 3) sampled roughly uniformly along the curve.
    """
    control = np.asarray(waypoints, dtype=float)
    fallback = densify(control, 1.0 / samples_per_meter)
    if len(control) < 3:
        return fallback
    stride = max(1, int(validator_spacing * samples_per_meter))

    for _ in range(max_refinements + 1):
        u = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(control, axis=0), axis=1))])
        spline = CubicSpline(u, control, axis=0, bc_type="natural")
        n_samples = max(50, int(u[-1] * samples_per_meter))
        curve = spline(np.linspace(0.0, u[-1], n_samples))
        if environment.is_polyline_free(curve) and (
            state_validator is None
            or all(state_validator(p) for p in np.vstack([curve[::stride], curve[-1]]))
        ):
            return curve
        control = densify(control, path_length(control) / (2 * (len(control) - 1)))
    return fallback


def minimum_jerk_trajectory(path: ArrayLike, duration: float, n_samples: int) -> Trajectory:
    """Time-parameterise a geometric path with a minimum-jerk profile.

    Arc length follows the quintic s(tau) = L (10 tau^3 - 15 tau^4 + 6 tau^5),
    tau = t / T, which is the unique polynomial with zero velocity and zero
    acceleration at both ends (minimises integrated squared jerk).

    Args:
        path: Dense geometric path, shape (M, 3).
        duration: Total motion time T [s].
        n_samples: Number of output samples (animation frames).
    """
    path = np.asarray(path, dtype=float)
    if duration <= 0 or n_samples < 2:
        raise ValueError("duration must be > 0 and n_samples >= 2.")
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    length = float(arc[-1])

    tau = np.linspace(0.0, 1.0, n_samples)
    s = length * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)
    speeds = (length / duration) * (30 * tau**2 - 60 * tau**3 + 30 * tau**4)
    positions = np.column_stack([np.interp(s, arc, path[:, k]) for k in range(3)])
    return Trajectory(times=tau * duration, positions=positions, speeds=speeds, length=length)


def duration_for_peak_speed(length: float, peak_speed: float) -> float:
    """Duration such that the minimum-jerk profile peaks at ``peak_speed``.

    The quintic profile's peak speed is 15/8 * L / T (reached at tau = 0.5).
    """
    return max(1e-3, 1.875 * length / peak_speed)
