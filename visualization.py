"""
visualization.py — Real-time 3D playback of the arm with PyVista (VTK).

Design notes
------------
* Every rigid body (link, joint housing, gripper) is meshed ONCE in its own
  local DH frame. Animating a frame then only means assigning each actor's
  4x4 ``user_matrix`` from forward kinematics — no per-frame mesh rebuilding,
  so playback stays smooth even with high-resolution geometry.
* The same ``SimulationViewer`` drives both the interactive window and
  off-screen export (GIF via Pillow, MP4 via imageio-ffmpeg if installed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, TypeAlias

import numpy as np
import pyvista as pv
from numpy.typing import NDArray

from kinematics import DHLink, RoboticArm
from motion_planner import BoxObstacle, Environment, PlanningResult, SphereObstacle

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = ["Theme", "SimulationRecording", "ArmRenderer", "SimulationViewer", "obstacle_summary"]


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Theme:
    """Colour palette for the scene (dark 'engineering lab' look)."""

    background_top: str = "#1b2838"
    background_bottom: str = "#07090d"
    floor: str = "#121820"
    grid: str = "#33475b"
    link: str = "#d5dbe3"
    joint: str = "#2f7bff"
    gripper: str = "#30343b"
    pedestal: str = "#3a4048"
    obstacle: str = "#ff4757"
    tree: str = "#6c7a89"
    raw_path: str = "#ffa502"
    smooth_path: str = "#00d2ff"
    trail: str = "#2ed573"
    start: str = "#2ed573"
    goal: str = "#ffd32a"
    text: str = "#e8edf3"
    joint_colors: tuple[str, ...] = (
        "#ff6b6b", "#ffa502", "#ffd32a", "#2ed573", "#1e90ff", "#a55eea",
    )


# --------------------------------------------------------------------------- #
# Data handed over by the orchestrator
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SimulationRecording:
    """Everything needed to replay a planned motion.

    Attributes:
        times: Frame timestamps, shape (N,) [s].
        joint_trajectory: Joint vectors per frame, shape (N, n) [rad].
        tcp_path: TCP positions per frame (from FK), shape (N, 3) [m].
        sigma_min: Smallest Jacobian singular value per frame, shape (N,).
        environment: Workspace and obstacles.
        planning: Raw RRT output (tree, raw path, shortcut path).
        smoothed_path: Dense spline path that was time-parameterised.
        start: Start TCP position.
        goal: Goal TCP position.
        summary: Ordered key/value lines shown in the HUD.
    """

    times: FloatArray
    joint_trajectory: FloatArray
    tcp_path: FloatArray
    sigma_min: FloatArray
    environment: Environment
    planning: PlanningResult
    smoothed_path: FloatArray
    start: FloatArray
    goal: FloatArray
    summary: dict[str, str] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return len(self.times)


# --------------------------------------------------------------------------- #
# Arm geometry + animation
# --------------------------------------------------------------------------- #
class ArmRenderer:
    """Builds the arm's meshes in local frames and poses them from FK.

    Each actor is bound to a frame index into the pose stack
    [frame_0, ..., frame_n, tcp]; ``update(q)`` only rewrites user matrices.

    The drawn geometry never exceeds the collision model used by the planner:
    links fit inside capsules of ``link_radius`` and every joint housing is
    inscribed in a sphere of ``joint_radius``.
    """

    def __init__(self, arm: RoboticArm, link_radius: float = 0.045,
                 joint_radius: float = 0.07, theme: Theme | None = None) -> None:
        self.arm = arm
        self.link_radius = link_radius
        self.joint_radius = joint_radius
        self.theme = theme or Theme()
        self._bound_actors: list[tuple[int, pv.Actor]] = []

    def add_to(self, plotter: pv.Plotter) -> None:
        """Create all arm actors on ``plotter``."""
        n = self.arm.n
        material = dict(smooth_shading=True, specular=0.6, specular_power=30, ambient=0.15)

        for i, link in enumerate(self.arm.links, start=1):
            mesh = self._link_mesh(link, self._radius_for(i))
            if mesh is not None:
                actor = plotter.add_mesh(mesh, color=self.theme.link, **material)
                self._bound_actors.append((i, actor))

        # Joint i rotates about z_{i-1} through o_{i-1}: its housing is a
        # cylinder rigidly attached to frame i-1 (joint 1 sits in the pedestal).
        # Radius 0.8 R and half-height 0.6 R put its rim exactly on the
        # collision sphere of radius R (0.8^2 + 0.6^2 = 1).
        for i in range(2, n + 1):
            R = self.joint_radius * self._radius_for(i) / self.link_radius
            housing = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=0.8 * R,
                                  height=1.2 * R, resolution=48)
            actor = plotter.add_mesh(housing, color=self.theme.joint, **material)
            self._bound_actors.append((i - 1, actor))

        gripper = self._gripper_mesh()
        actor = plotter.add_mesh(gripper, color=self.theme.gripper, **material)
        self._bound_actors.append((n, actor))

        # Small RGB triad on the TCP frame (x=red, y=green, z=blue).
        for axis, color in zip(np.eye(3), ("#ff4757", "#2ed573", "#1e90ff")):
            arrow = pv.Arrow(start=(0, 0, 0), direction=axis, scale=0.07,
                             tip_length=0.3, tip_radius=0.12, shaft_radius=0.04)
            actor = plotter.add_mesh(arrow, color=color, smooth_shading=True)
            self._bound_actors.append((n + 1, actor))

    def update(self, q: FloatArray) -> None:
        """Pose every actor for joint vector ``q``."""
        frames = self.arm.frame_poses(q)
        poses = np.concatenate([frames, (frames[-1] @ self.arm.tool_transform)[None]])
        for index, actor in self._bound_actors:
            actor.user_matrix = poses[index]

    # ----------------------------------------------------------- geometry
    def _radius_for(self, link_index: int) -> float:
        """Wrist links (last three) are drawn slimmer, like real cobots."""
        return self.link_radius * (0.8 if link_index > self.arm.n - 3 else 1.0)

    @staticmethod
    def _link_mesh(link: DHLink, radius: float) -> pv.PolyData | None:
        """Rigid mesh of link i expressed in frame i.

        Inverting A_i shows that, in frame i, the previous origin o_{i-1} sits
        at (-a, -d sin(alpha), -d cos(alpha)) independent of theta_i, so the
        link (o_{i-1} -> along z_{i-1} by d -> along x_i by a) is rigid here.
        """
        sa, ca = np.sin(link.alpha), np.cos(link.alpha)
        p_prev = np.array([-link.a, -link.d * sa, -link.d * ca])
        p_mid = np.array([-link.a, 0.0, 0.0])
        p_end = np.zeros(3)

        parts: list[pv.PolyData] = []
        for p0, p1 in ((p_prev, p_mid), (p_mid, p_end)):
            length = float(np.linalg.norm(p1 - p0))
            if length > 1e-6:
                parts.append(pv.Cylinder(center=(p0 + p1) / 2, direction=p1 - p0,
                                         radius=radius, height=length, resolution=40))
        if len(parts) == 2:  # round the DH "elbow" between the d and a segments
            parts.append(pv.Sphere(radius=radius, center=p_mid))
        return pv.merge(parts) if parts else None

    def _gripper_mesh(self) -> pv.PolyData:
        """Two-finger parallel gripper in the flange frame, spanning to the TCP.

        Its cross-section stays within the link capsule radius, so the
        flange -> TCP capsule of the collision model covers it.
        """
        length = max(float(self.arm.tool_transform[2, 3]), 0.06)
        r = 0.8 * self.link_radius
        flange = pv.Cylinder(center=(0, 0, 0.1 * length), direction=(0, 0, 1),
                             radius=r, height=0.2 * length, resolution=40)
        palm = pv.Box(bounds=(-1.2 * r, 1.2 * r, -0.45 * r, 0.45 * r,
                              0.2 * length, 0.4 * length))
        fingers = [
            pv.Box(bounds=(s * 1.1 * r - 0.15 * r, s * 1.1 * r + 0.15 * r,
                           -0.35 * r, 0.35 * r, 0.4 * length, length))
            for s in (-1.0, 1.0)
        ]
        return pv.merge([flange.triangulate(), palm.triangulate(),
                         *(f.triangulate() for f in fingers)])


# --------------------------------------------------------------------------- #
# Scene + playback
# --------------------------------------------------------------------------- #
class SimulationViewer:
    """Full scene: environment, planner diagnostics, arm, HUD and joint chart.

    Args:
        arm: The kinematic model being animated.
        recording: Planned motion to replay.
        show_tree: Draw the explored RRT tree.
        off_screen: Render without a window (for exports / CI).
        window_size: Render resolution in pixels.
        link_radius: Link capsule radius of the collision model [m].
        joint_radius: Joint-housing collision sphere radius [m].
        theme: Colour palette.
    """

    def __init__(
        self,
        arm: RoboticArm,
        recording: SimulationRecording,
        *,
        show_tree: bool = True,
        off_screen: bool = False,
        window_size: tuple[int, int] = (1600, 900),
        link_radius: float = 0.045,
        joint_radius: float = 0.07,
        theme: Theme | None = None,
    ) -> None:
        self.arm = arm
        self.rec = recording
        self.theme = theme or Theme()
        self.plotter = pv.Plotter(off_screen=off_screen, window_size=list(window_size),
                                  title="6-DOF Robotic Arm Simulator")
        self.arm_renderer = ArmRenderer(arm, link_radius, joint_radius, theme=self.theme)
        self._frame = 0
        self._paused = False
        self._loop = True
        self._cycle = recording.n_frames
        self._trail: pv.PolyData | None = None
        self._tree_actor: pv.Actor | None = None
        self._hud: pv.CornerAnnotation | None = None
        self._cursor = None
        self._build_scene(show_tree)
        self.render_frame(0)

    # ------------------------------------------------------------- public
    def show(self, fps: int = 30, loop: bool = True) -> None:
        """Open the interactive window and play the motion in real time.

        Keys: [space] pause/resume, [n] restart, [g] toggle RRT tree, [q] quit.
        """
        pl = self.plotter
        pl.add_key_event("space", self._toggle_pause)
        pl.add_key_event("n", lambda: self._seek(0))
        pl.add_key_event("g", self._toggle_tree)

        self._loop = loop
        self._cycle = self.rec.n_frames + int(0.8 * fps)  # hold the final pose briefly
        # The interactor renders after every timer callback.
        pl.add_timer_event(max_steps=10**9, duration=int(1000 / fps), callback=self._on_tick)
        pl.show()

    def _on_tick(self, _step: int) -> None:
        """Advance playback by one frame (called by the VTK timer)."""
        if self._paused:
            return
        nxt = self._frame + 1
        if nxt >= self._cycle:
            if not self._loop:
                return
            nxt = 0
        self._frame = nxt
        self.render_frame(min(nxt, self.rec.n_frames - 1))

    def export(self, path: str | Path, fps: int = 30, stride: int = 1,
               width: int | None = None) -> Path:
        """Render the motion off-screen to ``.gif`` (Pillow) or ``.mp4``.

        Args:
            path: Output file; the suffix selects the format.
            fps: Playback rate of the exported file.
            stride: Render every ``stride``-th frame (smaller files).
            width: Optional output width in pixels for GIFs (keeps aspect).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        indices = list(range(0, self.rec.n_frames, stride))
        indices += [self.rec.n_frames - 1] * int(0.6 * fps)  # hold the final pose
        suffix = path.suffix.lower()

        if suffix == ".mp4":
            try:
                self.plotter.open_movie(str(path), framerate=fps, quality=8)
            except (ImportError, ModuleNotFoundError) as exc:
                raise RuntimeError("MP4 export needs: pip install 'imageio[ffmpeg]'") from exc
            for k in indices:
                self._render_offscreen(k)
                self.plotter.write_frame()
            self.plotter.mwriter.close()
            return path

        if suffix != ".gif":
            raise ValueError("Export format must be .gif or .mp4")
        from PIL import Image  # Pillow ships with PyVista's dependency tree

        frames: list[Image.Image] = []
        for k in indices:
            self._render_offscreen(k)
            img = Image.fromarray(self.plotter.screenshot(return_img=True)).convert("RGB")
            if width is not None and img.width != width:
                img = img.resize((width, round(img.height * width / img.width)),
                                 Image.Resampling.LANCZOS)
            frames.append(img)
        _save_delta_gif(frames, path, frame_ms=int(1000 * stride / fps))
        return path

    def screenshot(self, path: str | Path, frame: int | None = None) -> Path:
        """Save a still image of a given frame (default: mid-motion)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._render_offscreen(self.rec.n_frames // 2 if frame is None else frame)
        self.plotter.screenshot(str(path))
        return path

    def close(self) -> None:
        self.plotter.close()

    def _render_offscreen(self, k: int) -> None:
        """Update frame k and force a render pass. Off-screen there is no
        interactor timer to trigger rendering, so moved actors would otherwise
        not appear in the captured image."""
        self.render_frame(k)
        self.plotter.render()

    def render_frame(self, k: int) -> None:
        """Pose the arm, grow the trail and refresh HUD/chart for frame k."""
        rec = self.rec
        q = rec.joint_trajectory[k]
        self.arm_renderer.update(q)

        if self._trail is not None:
            idx = np.minimum(np.arange(rec.n_frames), k)
            self._trail.points = rec.tcp_path[idx]

        if self._hud is not None:
            p = rec.tcp_path[k]
            q_deg = np.degrees(q)
            lines = [
                f"t = {rec.times[k]:5.2f} / {rec.times[-1]:.2f} s",
                f"TCP  x {p[0]:+.3f}  y {p[1]:+.3f}  z {p[2]:+.3f} m",
                "q [deg] " + " ".join(f"{v:+6.1f}" for v in q_deg),
                f"sigma_min(Jv) = {rec.sigma_min[k]:.3f}",
            ]
            self._hud.set_text("upper_right", "\n".join(lines))

        if self._cursor is not None:
            t = rec.times[k]
            self._cursor.update([t, t], list(self._chart_range))

    # -------------------------------------------------------------- scene
    def _build_scene(self, show_tree: bool) -> None:
        pl, th, rec = self.plotter, self.theme, self.rec
        pl.set_background(th.background_bottom, top=th.background_top)
        pl.enable_depth_peeling()  # order-independent transparency for obstacles

        self._add_floor()
        pedestal = pv.Cylinder(center=(0, 0, 0.02), direction=(0, 0, 1), radius=0.085,
                               height=0.04, resolution=60)
        pl.add_mesh(pedestal, color=th.pedestal, smooth_shading=True, specular=0.4)

        for obs in rec.environment.obstacles:
            pl.add_mesh(self._obstacle_mesh(obs), color=th.obstacle, opacity=0.5,
                        smooth_shading=isinstance(obs, SphereObstacle),
                        specular=0.8, specular_power=40)

        if len(rec.planning.tree_nodes) > 1:
            self._tree_actor = pl.add_mesh(self._tree_mesh(rec.planning), color=th.tree,
                                           opacity=0.45, line_width=1)
            self._tree_actor.visibility = show_tree
        if len(rec.planning.raw_path) > 1:
            pl.add_mesh(pv.lines_from_points(rec.planning.raw_path), color=th.raw_path,
                        line_width=2, opacity=0.85)
        pl.add_mesh(pv.lines_from_points(rec.smoothed_path), color=th.smooth_path,
                    line_width=2, opacity=0.7)

        for point, color in ((rec.start, th.start), (rec.goal, th.goal)):
            pl.add_mesh(pv.Sphere(radius=0.018, center=point), color=color,
                        smooth_shading=True, specular=0.5)

        # Executed path: points are rewritten every frame, so it is drawn as a
        # GPU-shaded tube (render_lines_as_tubes) rather than a tube mesh.
        self._trail = pv.lines_from_points(rec.tcp_path.copy())
        pl.add_mesh(self._trail, color=th.trail, line_width=7, render_lines_as_tubes=True)

        self.arm_renderer.add_to(pl)
        self._add_hud()
        self._add_joint_chart()

        pl.camera_position = [(1.80, -1.55, 1.12), (0.30, 0.02, 0.33), (0.0, 0.0, 1.0)]
        pl.camera.view_angle = 30

    def _add_floor(self) -> None:
        th = self.theme
        floor = pv.Plane(center=(0.2, 0.0, -0.001), direction=(0, 0, 1),
                         i_size=4.0, j_size=4.0, i_resolution=40, j_resolution=40)
        self.plotter.add_mesh(floor, color=th.floor, ambient=0.3)
        self.plotter.add_mesh(floor, color=th.grid, style="wireframe", line_width=1,
                              opacity=0.9)

    def _add_hud(self) -> None:
        th, rec = self.theme, self.rec
        blocks = (
            ("6-DOF ROBOTIC ARM SIMULATOR", 0.975, 15, th.text),
            ("RRT path planning  +  damped least-squares inverse kinematics", 0.935, 10,
             th.smooth_path),
            ("\n".join(f"{k:<10}{v}" for k, v in rec.summary.items()), 0.900, 9, th.text),
        )
        for text, y, size, color in blocks:
            actor = self.plotter.add_text(text, position=(0.012, y), viewport=True,
                                          font_size=size, color=color, font="courier")
            actor.GetTextProperty().SetVerticalJustificationToTop()
        self._hud = self.plotter.add_text("", position="upper_right", font_size=9,
                                          color=th.text, font="courier")
        legend = ("red: obstacles   grey: RRT tree   orange: raw RRT path\n"
                  "cyan: smoothed spline   green: executed TCP trail")
        self.plotter.add_text(legend, position="lower_left", font_size=8, color=th.tree,
                              font="courier")

    def _add_joint_chart(self) -> None:
        """Embedded 2D chart of q(t) with a moving time cursor."""
        rec, th = self.rec, self.theme
        q_deg = np.degrees(rec.joint_trajectory)
        lo, hi = float(q_deg.min()), float(q_deg.max())
        pad = 0.08 * (hi - lo + 1e-9)
        self._chart_range = (lo - pad, hi + pad)

        chart = pv.Chart2D(size=(0.30, 0.30), loc=(0.69, 0.03),
                           x_label="time [s]", y_label="joint angle [deg]")
        for i in range(rec.joint_trajectory.shape[1]):
            chart.line(rec.times, q_deg[:, i], color=th.joint_colors[i % len(th.joint_colors)],
                       width=2.0, label=f"q{i + 1}")
        self._cursor = chart.line([0.0, 0.0], list(self._chart_range), color="white",
                                  width=1.5, style="--")
        chart.x_axis.range = [float(rec.times[0]), float(rec.times[-1])]
        chart.y_axis.range = list(self._chart_range)
        chart.background_color = (0.03, 0.05, 0.08, 0.75)
        chart.border_color = (0.3, 0.4, 0.5, 1.0)
        for axis in (chart.x_axis, chart.y_axis):
            axis.pen.color = (0.7, 0.75, 0.8, 1.0)
            axis.grid_pen.color = (0.25, 0.3, 0.35, 1.0)
            axis.GetTitleProperties().SetColor(0.85, 0.88, 0.92)
            axis.GetLabelProperties().SetColor(0.75, 0.8, 0.85)
            axis.label_size = 11
            axis.tick_label_size = 10
        chart.legend_visible = True
        chart.GetLegend().GetLabelProperties().SetColor(0.85, 0.88, 0.92)
        chart.GetLegend().GetBrush().SetColorF(0.03, 0.05, 0.08, 0.6)
        self.plotter.add_chart(chart)

    @staticmethod
    def _obstacle_mesh(obs) -> pv.PolyData:
        if isinstance(obs, SphereObstacle):
            return pv.Sphere(radius=obs.radius, center=obs.center,
                             theta_resolution=60, phi_resolution=60)
        if isinstance(obs, BoxObstacle):
            c, h = np.asarray(obs.center), np.asarray(obs.half_extents)
            lo, hi = c - h, c + h
            return pv.Box(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
        raise TypeError(f"No mesh for obstacle type {type(obs).__name__}")

    @staticmethod
    def _tree_mesh(planning: PlanningResult) -> pv.PolyData:
        """All RRT edges as one PolyData with line cells (node -> parent)."""
        nodes, parents = planning.tree_nodes, planning.tree_parents
        child = np.flatnonzero(parents >= 0)
        cells = np.column_stack([np.full(len(child), 2), child, parents[child]]).ravel()
        return pv.PolyData(nodes, lines=cells)

    # ----------------------------------------------------------- controls
    def _toggle_pause(self) -> None:
        self._paused = not self._paused

    def _toggle_tree(self) -> None:
        if self._tree_actor is not None:
            self._tree_actor.visibility = not self._tree_actor.visibility
            self.plotter.render()

    def _seek(self, k: int) -> None:
        self._frame = k
        self.render_frame(k)
        self.plotter.render()


def _save_delta_gif(frames: list, path: Path, frame_ms: int) -> None:
    """Write a compact animated GIF.

    All frames share one adaptive 255-colour palette, and every pixel that is
    unchanged since the previous frame is written as the transparent index
    (with "do not dispose"), so the LZW encoder only pays for moving pixels.
    This typically shrinks a rendered animation by 5-10x.
    """
    from PIL import Image

    transparent = 255
    # Build the palette from a mosaic of evenly spaced frames.
    sample = frames[:: max(1, len(frames) // 8)]
    mosaic = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for i, frame in enumerate(sample):
        mosaic.paste(frame, (0, i * frame.height))
    palette_img = mosaic.quantize(colors=transparent, method=Image.Quantize.MEDIANCUT)

    indexed = [np.asarray(f.quantize(palette=palette_img, dither=Image.Dither.NONE))
               for f in frames]
    palette = palette_img.getpalette()[: 3 * transparent] + [0, 0, 0]

    out = []
    for i, current in enumerate(indexed):
        data = current.copy()
        if i > 0:
            data[current == indexed[i - 1]] = transparent
        img = Image.fromarray(data, mode="P")
        img.putpalette(palette)
        out.append(img)
    out[0].save(path, save_all=True, append_images=out[1:], loop=0, duration=frame_ms,
                transparency=transparent, disposal=1, optimize=False)


def obstacle_summary(obstacles: Sequence) -> str:
    """Compact human-readable obstacle count, e.g. '1 box, 1 sphere'."""
    boxes = sum(isinstance(o, BoxObstacle) for o in obstacles)
    spheres = sum(isinstance(o, SphereObstacle) for o in obstacles)
    parts = [f"{boxes} box" + ("es" if boxes != 1 else "")] if boxes else []
    if spheres:
        parts.append(f"{spheres} sphere" + ("s" if spheres != 1 else ""))
    return ", ".join(parts) or "none"
