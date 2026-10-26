"""
kinematics.py — Forward & inverse kinematics engine for a serial 6-DOF arm.

Built from first principles on NumPy/SciPy (no robotics toolboxes).

Conventions
-----------
* Standard (distal) Denavit-Hartenberg parameters. Link i is described by
  (a_i, alpha_i, d_i, theta_i) and the transform from frame i-1 to frame i is

      A_i = Rot_z(theta_i) · Trans_z(d_i) · Trans_x(a_i) · Rot_x(alpha_i)

* Frame 0 is the robot base (z-axis up), frame n is the tool flange, and a
  constant tool transform maps the flange to the tool centre point (TCP).
* All joints are revolute. SI units throughout: metres and radians.

Main entry points
-----------------
* ``RoboticArm.forward_kinematics(q)``   -> 4x4 TCP pose
* ``RoboticArm.jacobian(q)``             -> 6xn geometric Jacobian (analytic)
* ``RoboticArm.inverse_kinematics(p)``   -> singularity-robust numerical IK
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = ["DHLink", "IKResult", "RoboticArm"]


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DHLink:
    """One row of a standard Denavit-Hartenberg table (revolute joint).

    Attributes:
        a: Link length, distance along x_i from z_{i-1} to z_i [m].
        alpha: Link twist, angle about x_i from z_{i-1} to z_i [rad].
        d: Link offset, distance along z_{i-1} from x_{i-1} to x_i [m].
        theta_offset: Constant added to the joint variable so that
            theta_i = q_i + theta_offset [rad].
        q_min: Lower joint limit for q_i [rad].
        q_max: Upper joint limit for q_i [rad].
    """

    a: float
    alpha: float
    d: float
    theta_offset: float = 0.0
    q_min: float = -np.pi
    q_max: float = np.pi


@dataclass(frozen=True)
class IKResult:
    """Outcome of an inverse-kinematics solve.

    Attributes:
        q: Best joint vector found (returned even when ``success`` is False,
            so callers can degrade gracefully instead of crashing).
        success: True if every tolerance was met.
        iterations: Total solver iterations across all restarts.
        position_error: ||p_target - p(q)|| at the returned solution [m].
        orientation_error: Residual rotation angle [rad]; 0 if unconstrained.
        min_singular_value: Smallest singular value of the task Jacobian at
            the solution: a direct measure of distance to a singularity.
        restarts_used: Number of random restarts that were needed.
    """

    q: FloatArray
    success: bool
    iterations: int
    position_error: float
    orientation_error: float
    min_singular_value: float
    restarts_used: int


# --------------------------------------------------------------------------- #
# Robotic arm model
# --------------------------------------------------------------------------- #
class RoboticArm:
    """Serial revolute manipulator described by a standard DH table.

    The DH table is cached as NumPy columns so that every forward-kinematics
    call builds all n link transforms in one vectorised operation.
    """

    def __init__(
        self,
        links: Sequence[DHLink],
        *,
        base_transform: ArrayLike | None = None,
        tool_transform: ArrayLike | None = None,
        name: str = "6-DOF Arm",
    ) -> None:
        if len(links) == 0:
            raise ValueError("A robotic arm needs at least one link.")
        self.links: tuple[DHLink, ...] = tuple(links)
        self.name = name
        self.n: int = len(self.links)

        # Vectorised DH columns (shape (n,)), cached once.
        self._a = np.array([lk.a for lk in self.links], dtype=float)
        self._d = np.array([lk.d for lk in self.links], dtype=float)
        alpha = np.array([lk.alpha for lk in self.links], dtype=float)
        self._cos_alpha = np.cos(alpha)
        self._sin_alpha = np.sin(alpha)
        self._theta_offset = np.array([lk.theta_offset for lk in self.links])
        self.q_min = np.array([lk.q_min for lk in self.links], dtype=float)
        self.q_max = np.array([lk.q_max for lk in self.links], dtype=float)
        if np.any(self.q_min >= self.q_max):
            raise ValueError("Every joint needs q_min < q_max.")

        self.base_transform = self._as_transform(base_transform)
        self.tool_transform = self._as_transform(tool_transform)

    # ----------------------------------------------------------------- factory
    @classmethod
    def ur5(
        cls, tool_length: float = 0.0, base_transform: ArrayLike | None = None
    ) -> RoboticArm:
        """Universal Robots UR5 geometry from its published DH parameters.

        Joint limits are tightened from the hardware's +/-2*pi to +/-pi, which
        keeps the numerical IK away from redundant multi-turn solutions.

        Args:
            tool_length: Distance from the flange to the TCP along the flange
                z-axis [m] (e.g. the length of a gripper).
            base_transform: Optional 4x4 mounting pose of the base in the world.
        """
        half_pi = np.pi / 2.0
        links = [
            DHLink(a=0.0, alpha=half_pi, d=0.089159),
            DHLink(a=-0.425, alpha=0.0, d=0.0),
            DHLink(a=-0.39225, alpha=0.0, d=0.0),
            DHLink(a=0.0, alpha=half_pi, d=0.10915),
            DHLink(a=0.0, alpha=-half_pi, d=0.09465),
            DHLink(a=0.0, alpha=0.0, d=0.0823),
        ]
        tool = np.eye(4)
        tool[2, 3] = tool_length
        return cls(
            links, base_transform=base_transform, tool_transform=tool, name="UR5 (DH model)"
        )

    # ------------------------------------------------------------- properties
    @property
    def dof(self) -> int:
        """Number of joints."""
        return self.n

    @property
    def reach(self) -> float:
        """Conservative upper bound of the reach measured from frame 1 [m]."""
        tool_offset = float(np.linalg.norm(self.tool_transform[:3, 3]))
        return float(np.sum(np.abs(self._a[1:]) + np.abs(self._d[1:]))) + tool_offset

    # --------------------------------------------------------- forward model
    def link_transforms(self, q: ArrayLike) -> FloatArray:
        """All n DH link transforms A_1..A_n in one vectorised pass.

        Args:
            q: Joint vector, shape (n,).

        Returns:
            Array of shape (n, 4, 4) where element i is ^{i}A_{i+1}.
        """
        theta = self._check_q(q) + self._theta_offset
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = self._cos_alpha, self._sin_alpha

        A = np.zeros((self.n, 4, 4))
        A[:, 0, 0] = ct
        A[:, 0, 1] = -st * ca
        A[:, 0, 2] = st * sa
        A[:, 0, 3] = self._a * ct
        A[:, 1, 0] = st
        A[:, 1, 1] = ct * ca
        A[:, 1, 2] = -ct * sa
        A[:, 1, 3] = self._a * st
        A[:, 2, 1] = sa
        A[:, 2, 2] = ca
        A[:, 2, 3] = self._d
        A[:, 3, 3] = 1.0
        return A

    def frame_poses(self, q: ArrayLike) -> FloatArray:
        """World poses of DH frames 0..n.

        Returns:
            Array of shape (n + 1, 4, 4): element 0 is the base frame and
            element i is ^{W}T_i = T_base · A_1 · ... · A_i.
        """
        A = self.link_transforms(q)
        frames = np.empty((self.n + 1, 4, 4))
        frames[0] = self.base_transform
        for i in range(self.n):
            frames[i + 1] = frames[i] @ A[i]
        return frames

    def forward_kinematics(self, q: ArrayLike) -> FloatArray:
        """4x4 homogeneous pose of the tool centre point in the world frame."""
        return self.frame_poses(q)[-1] @ self.tool_transform

    def tcp_position(self, q: ArrayLike) -> FloatArray:
        """Cartesian position (x, y, z) of the TCP [m]."""
        return self.forward_kinematics(q)[:3, 3]

    def joint_positions(self, q: ArrayLike) -> FloatArray:
        """Skeleton of the arm: origins of frames 0..n followed by the TCP.

        Returns:
            Array of shape (n + 2, 3). Consecutive rows are the polyline used
            for drawing and for whole-body collision checks.
        """
        frames = self.frame_poses(q)
        tcp = frames[-1] @ self.tool_transform
        return np.vstack([frames[:, :3, 3], tcp[:3, 3]])

    def link_polyline(self, q: ArrayLike) -> FloatArray:
        """Exact centre-line of the physical links, for whole-body collision checks.

        Each DH link travels d_i along z_{i-1} and then a_i along x_i, so its
        centre-line passes through the corner e_i = o_{i-1} + d_i z_{i-1}.

        Returns:
            Array of shape (2n + 2, 3): o_0, e_1, o_1, e_2, ..., e_n, o_n, TCP.
        """
        frames = self.frame_poses(q)
        origins = frames[:, :3, 3]
        pts = np.empty((2 * self.n + 2, 3))
        pts[0:-2:2] = origins[:-1]
        pts[1:-2:2] = origins[:-1] + self._d[:, None] * frames[:-1, :3, 2]
        pts[-2] = origins[-1]
        pts[-1] = (frames[-1] @ self.tool_transform)[:3, 3]
        return pts

    # --------------------------------------------------------------- Jacobian
    def jacobian(self, q: ArrayLike) -> FloatArray:
        """Analytic geometric Jacobian J(q) at the TCP, world frame.

        For revolute joint i with axis z_{i-1} through origin o_{i-1}:

            J_v,i = z_{i-1} x (p_tcp - o_{i-1})      (linear velocity)
            J_w,i = z_{i-1}                          (angular velocity)

        Returns:
            Array of shape (6, n): [linear; angular] rows.
        """
        frames = self.frame_poses(q)
        return self._jacobian_from_frames(frames, frames[-1] @ self.tool_transform)

    def numerical_jacobian(self, q: ArrayLike, eps: float = 1e-6) -> FloatArray:
        """Central finite-difference Jacobian, used to verify the analytic one.

        The angular part uses the rotation vector of R(q+h)·R(q-h)^T / 2h,
        which is the finite-difference analogue of the angular velocity.
        """
        q = self._check_q(q)
        J = np.empty((6, self.n))
        for i in range(self.n):
            dq = np.zeros(self.n)
            dq[i] = eps
            T_plus = self.forward_kinematics(q + dq)
            T_minus = self.forward_kinematics(q - dq)
            J[:3, i] = (T_plus[:3, 3] - T_minus[:3, 3]) / (2.0 * eps)
            dR = T_plus[:3, :3] @ T_minus[:3, :3].T
            J[3:, i] = Rotation.from_matrix(dR).as_rotvec() / (2.0 * eps)
        return J

    def manipulability(self, q: ArrayLike, position_only: bool = False) -> float:
        """Yoshikawa manipulability w = sqrt(det(J J^T)) = prod(sigma_i).

        w -> 0 as the arm approaches a kinematic singularity.
        """
        J = self.jacobian(q)
        if position_only:
            J = J[:3]
        return float(np.prod(np.linalg.svd(J, compute_uv=False)))

    def singular_values(self, q: ArrayLike, position_only: bool = False) -> FloatArray:
        """Singular values of J (descending); the smallest one is the
        distance-to-singularity measure used by the IK damping schedule."""
        J = self.jacobian(q)
        return np.linalg.svd(J[:3] if position_only else J, compute_uv=False)

    # ---------------------------------------------------------------- inverse
    def inverse_kinematics(
        self,
        target_position: ArrayLike,
        target_orientation: ArrayLike | Rotation | None = None,
        q_init: ArrayLike | None = None,
        *,
        q_rest: ArrayLike | None = None,
        position_tolerance: float = 1e-4,
        orientation_tolerance: float = 1e-3,
        max_iterations: int = 150,
        max_restarts: int = 20,
        damping: float = 0.08,
        singularity_threshold: float = 0.05,
        max_step: float = 0.3,
        null_space_gain: float = 0.2,
        rng: np.random.Generator | None = None,
    ) -> IKResult:
        """Solve for joint angles that place the TCP at a target.

        Algorithm: damped least-squares (Levenberg-Marquardt) Newton iteration
        on the Jacobian pseudo-inverse,

            dq = J^T (J J^T + lambda^2 I)^-1 e  +  (I - J^+ J) k (q_rest - q),

        evaluated through a single SVD per iteration. Singularity handling:

        * Adaptive damping: lambda = 0 while sigma_min >= singularity_threshold
          (exact pseudo-inverse, fast convergence), then ramps smoothly up to
          ``damping`` as sigma_min -> 0, so joint velocities stay bounded in
          singular configurations.
        * Trust region: the largest joint step is capped at ``max_step``.
        * Backtracking line search: a step is only accepted if it reduces the
          task error, which makes the iteration globally robust.
        * Random restarts escape local minima and unreachable seeds.

        With position-only targets the 6-DOF arm is kinematically redundant
        (6 joints, 3 task constraints), so the remaining 3 DOF are used by the
        null-space term to pull the arm towards a comfortable ``q_rest`` posture
        without disturbing the TCP.

        Args:
            target_position: Desired TCP position (x, y, z) [m].
            target_orientation: Optional desired TCP orientation, a 3x3
                rotation matrix or ``scipy.spatial.transform.Rotation``.
            q_init: Initial guess (warm start). Defaults to ``q_rest`` or zeros.
            q_rest: Preferred posture for the null-space secondary task.
            position_tolerance: Convergence threshold on position error [m].
            orientation_tolerance: Convergence threshold on rotation error [rad].
            max_iterations: Iterations per attempt.
            max_restarts: Random restarts allowed after the first attempt.
            damping: Maximum damping factor lambda_max.
            singularity_threshold: sigma_min below which damping activates.
            max_step: Largest allowed change of any joint per iteration [rad].
            null_space_gain: Gain k of the posture-attraction secondary task.
            rng: Random generator used for restarts (reproducibility).

        Returns:
            IKResult with the best solution found.
        """
        p_target = np.asarray(target_position, dtype=float).reshape(-1)
        if p_target.shape != (3,) or not np.all(np.isfinite(p_target)):
            raise ValueError("target_position must be a finite 3-vector.")
        R_target = self._as_rotation_matrix(target_orientation)
        q_rest_arr = None if q_rest is None else self._check_q(q_rest)
        if q_init is not None:
            q0 = self._check_q(q_init)
        elif q_rest_arr is not None:
            q0 = q_rest_arr.copy()
        else:
            q0 = np.zeros(self.n)
        rng = rng if rng is not None else np.random.default_rng()

        settings = _SolverSettings(
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            max_iterations=max_iterations,
            damping=damping,
            singularity_threshold=singularity_threshold,
            max_step=max_step,
            null_space_gain=null_space_gain,
        )

        best: _Attempt | None = None
        total_iterations = 0
        for attempt in range(max_restarts + 1):
            seed = q0 if attempt == 0 else self.random_configuration(rng)
            result = self._solve_from(seed, p_target, R_target, q_rest_arr, settings)
            total_iterations += result.iterations
            if best is None or result.cost < best.cost:
                best = result
            if result.converged:
                break

        assert best is not None  # loop always runs at least once
        return IKResult(
            q=best.q,
            success=best.converged,
            iterations=total_iterations,
            position_error=best.position_error,
            orientation_error=best.orientation_error,
            min_singular_value=best.min_singular_value,
            restarts_used=attempt,
        )

    def random_configuration(self, rng: np.random.Generator) -> FloatArray:
        """Uniformly random joint vector inside the joint limits."""
        return rng.uniform(self.q_min, self.q_max)

    # --------------------------------------------------------------- helpers
    def _solve_from(
        self,
        q: FloatArray,
        p_target: FloatArray,
        R_target: FloatArray | None,
        q_rest: FloatArray | None,
        s: _SolverSettings,
    ) -> _Attempt:
        """One damped least-squares solve from a single seed."""
        q = np.clip(q, self.q_min, self.q_max)
        err, frames, tcp = self._task_error(q, p_target, R_target)
        m = err.size
        identity = np.eye(self.n)
        use_null_space = q_rest is not None and s.null_space_gain > 0 and m < self.n

        sigma_min = np.inf
        iterations = 0
        while iterations < s.max_iterations:
            if self._converged(err, s):
                break
            iterations += 1

            J = self._jacobian_from_frames(frames, tcp)[:m]
            U, S, Vt = np.linalg.svd(J, full_matrices=False)
            sigma_min = S[-1]

            # Adaptive damping (Nakamura-Hanafusa / Chiaverini schedule).
            if sigma_min >= s.singularity_threshold:
                lam2 = 0.0
            else:
                ratio = sigma_min / s.singularity_threshold
                lam2 = (1.0 - ratio * ratio) * s.damping * s.damping

            # Damped pseudo-inverse via SVD: J^+ = V diag(s/(s^2+lam^2)) U^T.
            dq = Vt.T @ ((S / (S * S + lam2)) * (U.T @ err))

            if use_null_space:
                # Exact projector onto null(J): N = I - V_r V_r^T.
                null_projector = identity - Vt.T @ Vt
                dq += null_projector @ (s.null_space_gain * (q_rest - q))

            peak = np.max(np.abs(dq))
            if peak > s.max_step:
                dq *= s.max_step / peak

            # Backtracking line search on the task-space error norm.
            err_norm = np.linalg.norm(err)
            step = 1.0
            for _ in range(6):
                q_new = np.clip(q + step * dq, self.q_min, self.q_max)
                err_new, frames_new, tcp_new = self._task_error(q_new, p_target, R_target)
                if np.linalg.norm(err_new) < err_norm:
                    break
                step *= 0.5
            else:
                break  # Stalled (local minimum or joint limit): let a restart try.
            q, err, frames, tcp = q_new, err_new, frames_new, tcp_new

        J = self._jacobian_from_frames(frames, tcp)[:m]
        sigma_min = float(np.linalg.svd(J, compute_uv=False)[-1])
        pos_err = float(np.linalg.norm(err[:3]))
        ori_err = float(np.linalg.norm(err[3:])) if m == 6 else 0.0
        return _Attempt(
            q=q,
            converged=self._converged(err, s),
            iterations=iterations,
            position_error=pos_err,
            orientation_error=ori_err,
            min_singular_value=sigma_min,
        )

    def _task_error(
        self, q: FloatArray, p_target: FloatArray, R_target: FloatArray | None
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Task-space error e = [p_t - p; rotvec(R_t R^T)] plus the FK results
        it was computed from (re-used for the Jacobian to avoid a second FK)."""
        frames = self.frame_poses(q)
        tcp = frames[-1] @ self.tool_transform
        e_pos = p_target - tcp[:3, 3]
        if R_target is None:
            return e_pos, frames, tcp
        e_rot = Rotation.from_matrix(R_target @ tcp[:3, :3].T).as_rotvec()
        return np.concatenate([e_pos, e_rot]), frames, tcp

    def _jacobian_from_frames(self, frames: FloatArray, tcp: FloatArray) -> FloatArray:
        z = frames[:-1, :3, 2]  # joint axes z_0..z_{n-1}, shape (n, 3)
        o = frames[:-1, :3, 3]  # axis origins o_0..o_{n-1}, shape (n, 3)
        J = np.empty((6, self.n))
        J[:3] = np.cross(z, tcp[:3, 3] - o).T
        J[3:] = z.T
        return J

    @staticmethod
    def _converged(err: FloatArray, s: _SolverSettings) -> bool:
        if np.linalg.norm(err[:3]) > s.position_tolerance:
            return False
        return err.size == 3 or np.linalg.norm(err[3:]) <= s.orientation_tolerance

    def _check_q(self, q: ArrayLike) -> FloatArray:
        """Validate shape and finiteness (limits are enforced by the solvers)."""
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape != (self.n,):
            raise ValueError(f"Expected {self.n} joint values, got shape {q.shape}.")
        if not np.all(np.isfinite(q)):
            raise ValueError("Joint vector contains NaN or inf.")
        return q

    @staticmethod
    def _as_transform(T: ArrayLike | None) -> FloatArray:
        if T is None:
            return np.eye(4)
        T = np.array(T, dtype=float)
        if T.shape != (4, 4):
            raise ValueError("Transforms must be 4x4 homogeneous matrices.")
        return T

    @staticmethod
    def _as_rotation_matrix(R: ArrayLike | Rotation | None) -> FloatArray | None:
        if R is None:
            return None
        if isinstance(R, Rotation):
            return R.as_matrix()
        R = np.asarray(R, dtype=float)
        if R.shape != (3, 3):
            raise ValueError("target_orientation must be a 3x3 rotation matrix.")
        return R

    def __repr__(self) -> str:
        return f"RoboticArm(name={self.name!r}, dof={self.n}, reach~{self.reach:.3f} m)"


# --------------------------------------------------------------------------- #
# Internal solver bookkeeping
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _SolverSettings:
    position_tolerance: float
    orientation_tolerance: float
    max_iterations: int
    damping: float
    singularity_threshold: float
    max_step: float
    null_space_gain: float


@dataclass(frozen=True)
class _Attempt:
    q: FloatArray
    converged: bool
    iterations: int
    position_error: float
    orientation_error: float
    min_singular_value: float

    @property
    def cost(self) -> float:
        """Scalar used to rank attempts (converged ones always win)."""
        return (0.0 if self.converged else 1.0) + self.position_error + self.orientation_error
