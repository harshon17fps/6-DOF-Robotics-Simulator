"""
main.py — Orchestrator: plan, solve and animate a 6-DOF pick-and-place move.

Pipeline
--------
1. Build the UR5 kinematic model (base yawed to face +x, 10 cm gripper).
2. Define the workcell: table bounds, a box pillar and a floating sphere.
3. Plan a TCP path with a configuration-aware RRT: every candidate waypoint
   must be reachable by IK *and* keep the whole arm body (not just the
   gripper) clear of obstacles and the table.
4. Shortcut, spline-smooth and minimum-jerk time-parameterise the path.
5. Track the trajectory with warm-started damped least-squares IK.
6. Verify whole-body clearance on every frame; re-plan with a new seed if
   any check fails.
7. Launch animated 3D playback, or export a GIF/MP4/PNG.

Usage
-----
    python main.py                          # interactive 3D window
    python main.py --export docs/demo.gif   # render an animation off-screen
    python main.py --headless               # run the pipeline + report only
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# numpy/scipy/pyvista live only in the project virtualenv (VTK has no wheels for
# newer system Pythons), so re-launch inside it when started with a bare python3.
_PROJECT_DIR = Path(__file__).resolve().parent
_VENV_DIR = _PROJECT_DIR / ".venv"
_VENV_PYTHON = _VENV_DIR / "bin" / "python"
if Path(sys.prefix).resolve() != _VENV_DIR.resolve():
    if not _VENV_PYTHON.exists():
        sys.exit(f"Project environment not found. Run this once first:\n  {_PROJECT_DIR / 'setup.sh'}")
    os.execv(_VENV_PYTHON, [str(_VENV_PYTHON), *sys.argv])

import numpy as np

from kinematics import IKResult, RoboticArm
from motion_planner import (
    BoxObstacle,
    Environment,
    PlanningResult,
    RRTConfig,
    RRTPlanner,
    SphereObstacle,
    duration_for_peak_speed,
    minimum_jerk_trajectory,
    path_length,
    smooth_path,
)

# --------------------------------------------------------------------------- #
# Scenario definition (world frame, metres)
# --------------------------------------------------------------------------- #
TOOL_LENGTH = 0.10        # gripper: flange -> TCP
LINK_RADIUS = 0.045       # collision capsule radius of every link
JOINT_RADIUS = 0.07       # collision sphere enclosing each joint housing
BODY_MARGIN = 0.01        # extra safety gap demanded of the arm body while planning
TCP_CLEARANCE = 0.06      # TCP must stay this far from obstacle surfaces

# Comfortable "ready" posture: elbow up, gripper pointing straight down.
Q_READY = np.array([0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])

START = np.array([0.42, -0.38, 0.12])   # pick location
GOAL = np.array([0.42, 0.38, 0.12])     # place location


def build_arm() -> RoboticArm:
    """UR5 mounted on the table, yawed 180 deg so its working side faces +x."""
    yaw_pi = np.diag([-1.0, -1.0, 1.0, 1.0])
    return RoboticArm.ur5(tool_length=TOOL_LENGTH, base_transform=yaw_pi)


def build_environment() -> Environment:
    """Table-top workcell: a pillar between pick & place plus a floating ball."""
    return Environment(
        lower=[0.15, -0.65, 0.06],
        upper=[0.75, 0.65, 0.70],
        obstacles=[
            BoxObstacle(center=(0.45, 0.0, 0.16), half_extents=(0.07, 0.10, 0.16)),
            SphereObstacle(center=(0.30, 0.02, 0.50), radius=0.08),
        ],
        clearance=TCP_CLEARANCE,
    )


# --------------------------------------------------------------------------- #
# Whole-body feasibility (bridges the robot model and the environment)
# --------------------------------------------------------------------------- #
class WholeBodyChecker:
    """Decides whether a TCP position is feasible for the *entire* arm.

    A point is feasible when position IK (seeded from the ready posture, so
    the redundancy resolution is deterministic) converges, and the arm's
    collision model (a capsule around every link plus a sphere around every
    joint housing) keeps ``margin`` metres from every obstacle and the table.
    The planner only sees the boolean ``__call__`` interface, which keeps
    ``motion_planner`` free of any robot-specific code.
    """

    def __init__(self, arm: RoboticArm, env: Environment, q_rest: np.ndarray,
                 link_radius: float, joint_radius: float, margin: float,
                 floor_z: float = 0.0) -> None:
        self.arm = arm
        self.env = env
        self.q_rest = q_rest
        self.link_radius = link_radius
        self.joint_radius = joint_radius
        self.margin = margin
        self.floor_z = floor_z
        self.calls = 0

    def solve(self, p: np.ndarray, q_seed: np.ndarray | None = None) -> IKResult:
        """Position IK with null-space attraction to the ready posture."""
        return self.arm.inverse_kinematics(
            p, q_init=self.q_rest if q_seed is None else q_seed, q_rest=self.q_rest,
            position_tolerance=1e-5, max_iterations=100, max_restarts=0,
        )

    def body_gap(self, q: np.ndarray) -> float:
        """Smallest distance between the arm's surface and an obstacle/table [m]."""
        line = self.arm.link_polyline(q)
        link_gap = self.env.polyline_clearance(line, resolution=0.01) - self.link_radius
        # Housings of joints 2..n sit on the origins of frames 1..n-1.
        joints = self.arm.frame_poses(q)[1:-1, :3, 3]
        joint_gap = float(np.min(self.env.distance(joints))) - self.joint_radius
        # Everything from the shoulder onwards must stay above the table.
        floor_gap = min(float(np.min(line[2:, 2])) - self.link_radius,
                        float(np.min(joints[:, 2])) - self.joint_radius) - self.floor_z
        return min(link_gap, joint_gap, floor_gap)

    def __call__(self, p: np.ndarray) -> bool:
        self.calls += 1
        result = self.solve(p)
        return result.success and self.body_gap(result.q) >= self.margin


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PipelineOutput:
    plan: PlanningResult
    smoothed: np.ndarray
    times: np.ndarray
    joints: np.ndarray
    tcp_path: np.ndarray
    sigma_min: np.ndarray
    ik_errors: np.ndarray
    ik_iterations: np.ndarray
    body_gap: float
    validator_calls: int
    attempts: int
    wall_time: float


def track_trajectory(checker: WholeBodyChecker, positions: np.ndarray, q0: np.ndarray):
    """Warm-started IK along a Cartesian trajectory.

    Seeding every solve with the previous frame's solution keeps the joint
    trajectory continuous (no configuration flips) and converges in a
    handful of iterations per frame.
    """
    n = len(positions)
    joints = np.empty((n, checker.arm.dof))
    sigma = np.empty(n)
    errors = np.empty(n)
    iterations = np.empty(n, dtype=int)
    ok = True
    q = q0
    for k, p in enumerate(positions):
        r = checker.solve(p, q_seed=q)
        ok &= r.success
        q = r.q
        joints[k], sigma[k], errors[k], iterations[k] = q, r.min_singular_value, \
            r.position_error, r.iterations
    return joints, sigma, errors, iterations, ok


def run_pipeline(seed: int, n_frames: int, peak_speed: float,
                 max_attempts: int = 6, verbose: bool = True) -> tuple[RoboticArm, Environment, PipelineOutput]:
    """Plan -> smooth -> time-parameterise -> IK -> verify, with re-planning."""
    t_start = time.perf_counter()
    arm = build_arm()
    env = build_environment()
    checker = WholeBodyChecker(arm, env, Q_READY, LINK_RADIUS, JOINT_RADIUS, BODY_MARGIN)

    for name, p in (("start", START), ("goal", GOAL)):
        if not checker(p):
            raise SystemExit(f"Scenario error: the {name} pose {p} is not feasible.")
    q_start = checker.solve(START).q

    for attempt in range(1, max_attempts + 1):
        checker.calls = 0
        planner = RRTPlanner(env, RRTConfig(step_size=0.05, goal_bias=0.10,
                                            max_iterations=8000),
                             state_validator=checker, seed=seed + attempt - 1)
        plan = planner.plan(START, GOAL)
        if not plan.success:
            _log(verbose, f"  attempt {attempt}: {plan.message} -> re-planning")
            continue

        smoothed = smooth_path(plan.path, env, state_validator=checker)
        duration = duration_for_peak_speed(path_length(smoothed), peak_speed)
        traj = minimum_jerk_trajectory(smoothed, duration, n_frames)
        joints, sigma, errors, iters, ik_ok = track_trajectory(checker, traj.positions, q_start)
        gap = min(checker.body_gap(q) for q in joints)

        if ik_ok and gap >= 0.0:
            tcp = np.array([arm.tcp_position(q) for q in joints])
            return arm, env, PipelineOutput(
                plan=plan, smoothed=smoothed, times=traj.times, joints=joints,
                tcp_path=tcp, sigma_min=sigma, ik_errors=errors, ik_iterations=iters,
                body_gap=gap, validator_calls=checker.calls, attempts=attempt,
                wall_time=time.perf_counter() - t_start,
            )
        reason = "IK lost track" if not ik_ok else f"arm clearance {gap * 100:.1f} cm < 0"
        _log(verbose, f"  attempt {attempt}: {reason} -> re-planning with a new seed")

    raise SystemExit(f"No feasible motion found in {max_attempts} attempts.")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)


def print_report(arm: RoboticArm, env: Environment, out: PipelineOutput) -> None:
    plan = out.plan
    dt = np.diff(out.times)
    joint_speed = np.abs(np.diff(out.joints, axis=0)) / dt[:, None]
    rule = "=" * 74
    rows = [
        ("Robot", f"{arm.name}, {arm.dof} revolute joints, {TOOL_LENGTH * 100:.0f} cm gripper"),
        ("Workcell", f"{len(env.obstacles)} obstacles, TCP clearance {env.clearance * 100:.0f} cm"),
        ("Body model", f"link capsules r={LINK_RADIUS * 100:.1f} cm, "
                       f"joint spheres r={JOINT_RADIUS * 100:.1f} cm"),
        ("RRT search", f"{plan.iterations} iterations, {len(plan.tree_nodes)} nodes, "
                       f"{plan.planning_time * 1e3:.0f} ms, {out.validator_calls} whole-body IK checks"),
        ("Path", f"raw {len(plan.raw_path)} wp / {path_length(plan.raw_path):.2f} m  ->  "
                 f"shortcut {len(plan.path)} wp / {plan.path_length:.2f} m  ->  "
                 f"spline {path_length(out.smoothed):.2f} m"),
        ("Trajectory", f"minimum-jerk, {out.times[-1]:.2f} s, {len(out.times)} frames"),
        ("IK tracking", f"max error {out.ik_errors.max() * 1e3:.3f} mm, "
                        f"{out.ik_iterations.mean():.1f} iterations/frame"),
        ("Joint motion", f"peak joint speed {np.degrees(joint_speed.max()):.1f} deg/s"),
        ("Singularity", f"min sigma(Jv) = {out.sigma_min.min():.3f}  (damping below 0.050)"),
        ("Safety", f"min arm-to-obstacle gap {out.body_gap * 100:.1f} cm"),
        ("Total", f"{out.wall_time:.2f} s on {out.attempts} planning attempt(s)"),
    ]
    print(rule)
    print(" 6-DOF Robotic Arm Kinematics Simulator & Trajectory Planner")
    print(rule)
    for key, value in rows:
        print(f" {key:<13} {value}")
    print(rule)


def hud_summary(out: PipelineOutput) -> dict[str, str]:
    plan = out.plan
    return {
        "RRT": f"{len(plan.tree_nodes)} nodes, {plan.planning_time * 1e3:.0f} ms",
        "Path": f"{path_length(plan.raw_path):.2f} m raw -> {path_length(out.smoothed):.2f} m smooth",
        "IK": f"max err {out.ik_errors.max() * 1e3:.2f} mm, "
              f"{out.ik_iterations.mean():.1f} it/frame",
        "Clearance": f"min arm gap {out.body_gap * 100:.1f} cm",
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, default=7, help="RRT random seed")
    parser.add_argument("--frames", type=int, default=240, help="trajectory samples")
    parser.add_argument("--speed", type=float, default=0.35, help="peak TCP speed [m/s]")
    parser.add_argument("--fps", type=int, default=30, help="playback frame rate")
    parser.add_argument("--export", metavar="PATH",
                        help="render the animation off-screen to .gif or .mp4")
    parser.add_argument("--gif-width", type=int, default=960, help="GIF width in pixels")
    parser.add_argument("--screenshot", metavar="PATH", help="save a mid-motion .png")
    parser.add_argument("--headless", action="store_true", help="no rendering, report only")
    parser.add_argument("--hide-tree", action="store_true", help="hide the RRT tree")
    parser.add_argument("--no-loop", action="store_true", help="play the motion once")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print("Planning... (configuration-aware RRT + damped least-squares IK)")
    arm, env, out = run_pipeline(args.seed, args.frames, args.speed)
    print_report(arm, env, out)

    if args.headless:
        return 0

    # Imported lazily so --headless runs never touch the GPU / VTK stack.
    from visualization import SimulationRecording, SimulationViewer

    recording = SimulationRecording(
        times=out.times, joint_trajectory=out.joints, tcp_path=out.tcp_path,
        sigma_min=out.sigma_min, environment=env, planning=out.plan,
        smoothed_path=out.smoothed, start=START, goal=GOAL, summary=hud_summary(out),
    )
    off_screen = bool(args.export or args.screenshot)
    viewer = SimulationViewer(arm, recording, show_tree=not args.hide_tree,
                              off_screen=off_screen,
                              window_size=(1600, 900) if not off_screen else (1920, 1080),
                              link_radius=LINK_RADIUS, joint_radius=JOINT_RADIUS)
    if off_screen:
        if args.screenshot:
            print(f"Saved screenshot -> {viewer.screenshot(args.screenshot)}")
        if args.export:
            stride = 2 if args.export.lower().endswith(".gif") else 1
            path = viewer.export(args.export, fps=args.fps, stride=stride,
                                 width=args.gif_width)
            print(f"Saved animation  -> {path}")
        viewer.close()
        return 0

    print("Launching 3D viewer  [space] pause  [n] restart  [g] toggle tree  [q] quit")
    viewer.show(fps=args.fps, loop=not args.no_loop)
    return 0


if __name__ == "__main__":
    sys.exit(main())
