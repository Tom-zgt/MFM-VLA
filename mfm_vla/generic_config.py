"""Frozen constants of the generic GPT + π0.5 (+ Blender) agent.

Every constant is a physical or budget quantity that is set on the dev split
only and then frozen; each episode record stores the full config.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class MoveConfig:
    """Free-space end-effector motion; stops on stall, never pushes."""

    gain: float = 1.6
    tolerance_m: float = 0.01
    max_steps: int = 80
    stall_window: int = 8
    stall_min_residual_m: float = 0.03
    stall_progress_m: float = 0.002
    brake: bool = False
    """G5f: with a limited change of motion per step, slow down early enough to stop at the
    goal (speed below sqrt(2 * change * distance)); otherwise the hand overshoots by several
    centimetres and the overshoot reads as a stall."""
    metres_per_step_per_unit: float = 0.0126
    """Hand speed for one unit of position action on LIBERO's OSC controller."""


@dataclass(frozen=True)
class GroundingConfig:
    """Coarse-to-fine image grounding; crops follow the coarse boxes, not a fixed window."""

    render_px: int = 1024
    coarse_px: int = 768
    fine_px: int = 768
    max_candidates: int = 3
    crop_margin: float = 0.6
    min_crop_frac: float = 0.25


@dataclass(frozen=True)
class ApproachConfig:
    """Where π0.5 takes over, open gripper above the object.

    ``reference="support"`` (G1, G2a): a fixed height above the support surface.
    ``reference="top"`` (G2b): a clearance above the measured top of the object.
    ``reference="policy"`` (G2c): the VLA's own typical pre-grasp offset from the
    object top, profiled on the VLA's training tasks (``configs/policies``).
    """

    reference: str = "support"
    clearance_above_support_m: float = 0.13
    clearance_above_top_m: float = 0.10
    policy_profile: str = "configs/policies/pi05_libero.json"


@dataclass(frozen=True)
class PlaceConfig:
    """Where π0.5 takes over again after a grasp: its own typical pre-release pose
    over the place target, profiled on its training tasks (G3)."""

    policy_profile: str = "configs/policies/pi05_libero_place.json"
    grasp_settle_m: float | None = None
    grasp_settle_steps: int = 2
    carry_max_action_change: float | None = None
    """Per-step, per-axis limit on the change of the commanded motion while carrying
    (action units; 0.15 is about 2 mm/step², π0.5's own p90 while carrying)."""
    carry_max_action_norm: float | None = None
    """Cap on the commanded motion's norm while carrying (action units; on LIBERO's
    OSC one unit moves the hand about 1.26 cm/step, so 0.85 is π0.5's own median
    carrying speed of 1.07 cm/step)."""
    reroute_on_stall: bool = False
    """A stalled carry (joint limit, rim contact) is retried once through the start
    pose, where the arm is far from its limits. Approaches are not rerouted: a
    stalled approach is handed to π0.5 where it stopped."""
    release_wait_steps: int | None = None
    """After the hand-back over the target, open the gripper in free space if π0.5 has
    not opened it within this many steps (None: π0.5 alone decides)."""
    max_redo_after_drop: int = 0
    """If the object is no longer held after the carry, hand off to it again."""
    reroute_min_residual_m: float = 0.05
    """A carry counts as blocked, and is rerouted, only this far from its waypoint; nearer,
    the hand is just slowing down on arrival."""
    vla_carry_first_steps: int | None = None
    """After a confirmed grasp, π0.5 carries first; the agent carries only if, after this
    many steps, the hand's horizontal distance to the measured place target has not
    shrunk below ``vla_carry_progress`` of where it started (None: the agent always carries)."""
    vla_carry_progress: float = 0.7
    vla_carry_arrive_m: float = 0.04
    """π0.5's own carry counts as arrived within this horizontal distance of the target."""
    vla_carry_min_progress_m: float | None = None
    """G5c: instead of ``vla_carry_progress``, the agent carries unless π0.5 has come at
    least this much closer to the target within ``vla_carry_first_steps``; π0.5's own
    successful carries cover 2.2-3.4 cm in their first 5 steps, the rest under 0.5 cm."""
    vla_carry_max_steps: int | None = None
    """G5g: the agent takes the carry over once π0.5 has carried this long without arriving;
    π0.5 can creep toward the target too slowly to trip the stall check."""
    vla_carry_timeout_progress_m: float | None = None
    """G5i: past ``vla_carry_max_steps``, take over only while π0.5 has come less than this
    much closer over the last ``vla_carry_stall_window`` steps (a slow but steady carry
    is left alone)."""
    vla_carry_stall_window: int = 10
    vla_carry_stall_m: float = 0.01
    """G5c: the agent also carries if π0.5 comes less than this much closer over a window."""
    vla_carry_stall_beyond_m: float | None = None
    """G5d: the window check applies only farther than this from the target; nearer,
    π0.5 is lowering the object to place it and makes no horizontal progress."""
    deadline_release_steps: int | None = None
    """G5f: with this many steps left, an object still held within
    ``deadline_release_near_m`` (horizontal) of the place target is let go in place; π0.5
    can hover over the target, opening only for single steps, until time runs out."""
    deadline_release_near_m: float = 0.06
    vla_carry_open_far_m: float | None = None
    """G5c: π0.5 opening the gripper farther than this from the target would drop the
    object; the agent closes it again and carries."""


@dataclass(frozen=True)
class MeasureConfig:
    """Second-view measurement acceptance."""

    max_ray_gap_m: float = 0.01
    relook_on: str = "midpoint"
    from_home: bool = False
    """Return the hand to its start pose before each measurement, so one target's
    re-looks do not leave the wrist camera aimed away from the next target."""
    order: str = "place_first"
    """G3: which target is measured first. The grasp target rarely needs a re-look, so
    measuring it first usually saves the return to the start pose."""
    use_best: bool = False
    """Keep the view whose two rays met most closely, not the last one."""
    survey_depth_m: tuple[float, float] | None = None
    """G5: before measuring, one move to a wrist pose that sees every grounded target's
    fixed-view ray over this span of depths below the start hand height (None: measure
    each target from the start pose, re-looking as needed)."""
    waypoint_tolerance_m: float = 0.03
    """Arrival tolerance of the approach's intermediate waypoint (G5); the final
    pre-grasp pose keeps ``MoveConfig.tolerance_m``."""
    accept_gap_m: float | None = None
    """Accept a measurement whose best ray gap is at most this, after the re-looks
    allowed by ``max_ray_gap_m`` (None: ``max_ray_gap_m``). The triangulation error is
    about half the gap, small against a container's opening."""


@dataclass(frozen=True)
class VerifyConfig:
    """M5 checks on real images. Grasp check: after a grasp-and-lift, GPT confirms the
    held object; if wrong, open, rise, and hand back to π0.5 from a closer pre-grasp
    pose (``retry_profile``), at most ``max_regrasp`` times."""

    grasp_check: bool = False
    grasp_reference: bool = False
    """Show the verifier the grounded target (start crop + description), not just its name."""
    check_px: int = 512
    max_regrasp: int = 2
    release_rise_m: float = 0.05
    retry_profile: str = "configs/policies/pi05_libero_lag10.json"
    hold_min_opening_m: float | None = None
    """Finger separation above which the gripper counts as holding (None: the
    adapter's 8 mm). A bowl rim is held at 3-7 mm; the GPT grasp check then
    separates a thin grasp from an empty close."""


@dataclass(frozen=True)
class ProgramConfig:
    """G3b task program: GPT splits the instruction into single-skill steps.

    A carry step ends on a proprioceptive release (held, then opened, then
    ``retreat_steps`` more); any other step ends when the verifier says done,
    asked every ``check_every`` steps, or after ``max_step_steps``.
    """

    plan_px: int = 768
    max_steps_in_program: int = 4
    hold_steps_before_release: int = 5
    retreat_steps: int = 20
    check_every: int = 50
    check_px: int = 512
    max_step_steps: int = 250


@dataclass(frozen=True)
class RouteConfig:
    """G5: episodes that are one of the VLA's training tasks in the same arrangement
    are left to the VLA; the rest are assisted."""

    enabled: bool = False
    catalog: str | None = "configs/policies/pi05_libero_catalog/catalog.json"
    """G5h: None routes from the instruction and the start image only (no training-task
    list, no reference images)."""
    image_px: int = 512
    vla_first_steps: int | None = None
    """π0.5-first (off): after the step-0 measurements the hand goes back to the start pose and π0.5
    tries the grasp alone for this many steps; the agent hands off only when no grasp is
    confirmed by then, or earlier when π0.5 closes the gripper away from the target."""
    vla_first_close_xy_m: float | None = None
    """Horizontal distance from the measured grasp target beyond which a gripper close
    during π0.5's try counts as aimed at another object."""


@dataclass(frozen=True)
class ArticulationConfig:
    """G5e: drawers outside the VLA's training distribution are opened by the harness.

    The handle is measured in two views (fixed, then the wrist from a survey pose),
    measured again from close in front of it, then opened with the VLA's own motion
    profiled on its training tasks (``policy_profile``): turn the hand, come in above
    the handle, press down behind it until blocked, pull along the measured outward
    direction until the drawer stops. A GPT check on the images ends the attempt.
    """

    enabled: bool = False
    policy_profile: str = "configs/policies/pi05_libero_drawer.json"
    render_px: int = 1024
    max_ray_gap_m: float = 0.02
    accept_gap_m: float = 0.03
    front_clearance_m: float = 0.05
    """The close look is taken from this far out along the pull direction, above the hook pose."""
    close_max_shift_m: float = 0.06
    """The close look replaces the first measurement (position and pull direction) unless
    its handle is this far from the first one, i.e. another handle."""
    pull_travel_m: float = 0.25
    """Pull at most this far; the drawer's end stop ends the pull earlier (stall)."""
    pull_max_action_norm: float = 0.6
    max_attempts: int = 2
    min_steps_for_attempt: int = 70


@dataclass(frozen=True)
class ModelConfig:
    perception: str = field(
        default_factory=lambda: __import__("os").environ.get(
            "MFM_MODEL", __import__("os").environ.get("OPENAI_MODEL", "gpt-4o")
        )
    )
    planning: str = field(
        default_factory=lambda: __import__("os").environ.get(
            "MFM_MODEL", __import__("os").environ.get("OPENAI_MODEL", "gpt-4o")
        )
    )
    verification: str = field(
        default_factory=lambda: __import__("os").environ.get(
            "MFM_MODEL", __import__("os").environ.get("OPENAI_MODEL", "gpt-4o")
        )
    )


@dataclass(frozen=True)
class AgentConfig:
    stage: str = "G1"
    move: MoveConfig = field(default_factory=MoveConfig)
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    approach: ApproachConfig = field(default_factory=ApproachConfig)
    measure: MeasureConfig = field(default_factory=MeasureConfig)
    place: PlaceConfig = field(default_factory=PlaceConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)
    route: RouteConfig = field(default_factory=RouteConfig)
    articulation: ArticulationConfig = field(default_factory=ArticulationConfig)
    models: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> dict:
        return asdict(self)
