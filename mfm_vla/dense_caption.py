"""Dense per-chunk captions for π0.5 (`action_horizon = 10`).

Instead of conditioning every 10-step action chunk on the static episode-level task
instruction (which lets a small-data VLA memorize `(scene, global instruction) ->
trajectory` shortcuts), each frame `t` is conditioned on a local dense caption
describing:
  1. `scene`: visible objects/fixtures and their coarse workspace locations;
  2. `subgoal`: what the next 10 steps should accomplish locally;
  3. `geometry`: direction and estimated distance (cm) from the gripper to the
     active target (and destination when carrying);
  4. `next 10 steps`: coarse 10-step displacement direction/magnitude and gripper
     action over `[t, t + 10)`.

Workspace / camera directional convention (matches LIBERO language and agentview):
  +x = front (towards camera/viewer), -x = back (towards robot base)
  +y = right,                         -y = left
  +z = up / above,                    -z = down / below
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

import numpy as np

HORIZON = 10
_INDEX_RE = re.compile(r"_\d+$")

# Clean human-readable names for LIBERO object/fixture/site identifiers.
_PRETTY_NAMES = {
    "akita_black_bowl": "black bowl",
    "white_bowl": "white bowl",
    "red_coffee_mug": "red mug",
    "porcelain_mug": "white mug",
    "white_yellow_mug": "yellow and white mug",
    "moka_pot": "moka pot",
    "chefmate_8_frypan": "frypan",
    "alphabet_soup": "alphabet soup can",
    "cream_cheese": "cream cheese box",
    "Cream_cheese": "cream cheese box",
    "tomato_sauce": "tomato sauce can",
    "ketchup": "ketchup bottle",
    "butter": "butter stick",
    "milk": "milk carton",
    "orange_juice": "orange juice carton",
    "chocolate_pudding": "chocolate pudding box",
    "new_salad_dressing": "salad dressing bottle",
    "salad_dressing": "salad dressing bottle",
    "bbq_sauce": "bbq sauce bottle",
    " wine_bottle": "wine bottle",
    "wine_bottle": "wine bottle",
    "black_book": "black book",
    "yellow_book": "yellow book",
    "book": "book",
    "plate": "plate",
    "basket": "basket",
    "wooden_tray": "wooden tray",
    "desk_caddy": "desk caddy",
    "wine_rack": "wine rack",
    "flat_stove": "stove",
    "microwave": "microwave",
    "wooden_cabinet": "wooden cabinet",
    "white_cabinet": "white cabinet",
    "wooden_two_layer_shelf": "wooden shelf",
}


def clean_name(raw: str | None) -> str:
    """Convert a simulator object/site name into a clean natural-language phrase."""
    if not raw:
        return "target"
    s = str(raw)
    for prefix in ("kitchen_table_", "living_room_table_", "study_table_", "main_table_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    # Handle articulated / container regions such as wooden_cabinet_1_top_region
    m = re.match(
        r"^([a-zA-Z0-9_]+?)_\d+_(top|middle|bottom|front|back|left|right|contain|heating|cook)_(?:region|side)$",
        s,
    )
    if m:
        base, part = m.group(1), m.group(2)
        base_clean = _PRETTY_NAMES.get(base, base.replace("_", " "))
        if "cabinet" in base_clean:
            if part in ("top", "middle", "bottom"):
                return f"{part} drawer of {base_clean}"
            return f"{part} of {base_clean}"
        if "shelf" in base_clean:
            if part == "top":
                return f"top layer of {base_clean}"
            if part == "bottom":
                return f"bottom layer of {base_clean}"
        if "caddy" in base_clean:
            return f"{part} compartment of {base_clean}"
        if part == "cook":
            return f"{base_clean} burner"
        if part in ("contain", "heating"):
            return base_clean
        return f"{part} of {base_clean}"
    m_rel = re.match(r"^([a-zA-Z0-9_]+?)_(left|right|front|back)_region$", s)
    if m_rel:
        base = _INDEX_RE.sub("", m_rel.group(1))
        base_clean = _PRETTY_NAMES.get(base, base.replace("_", " "))
        return f"table {m_rel.group(2)} of {base_clean}"
    base = _INDEX_RE.sub("", s)
    return _PRETTY_NAMES.get(base, base.replace("_", " "))


def table_sector(pos_xyz: np.ndarray, center_xy: np.ndarray) -> str:
    """Coarse workspace sector of an object relative to the table centroid."""
    dx = float(pos_xyz[0] - center_xy[0])
    dy = float(pos_xyz[1] - center_xy[1])
    fb = "front" if dx > 0.06 else ("back" if dx < -0.06 else "mid")
    lr = "right" if dy > 0.06 else ("left" if dy < -0.06 else "center")
    if fb == "mid" and lr == "center":
        return "center"
    if fb == "mid":
        return lr
    if lr == "center":
        return fb
    return f"{fb}-{lr}"


def format_delta_3d(
    delta_xyz: np.ndarray,
    *,
    quantize_cm: int = 1,
    qualitative_only: bool = False,
    min_axis_cm: float = 1.5,
) -> str:
    """Format `target_pos - eef_pos` in natural camera-aligned directions and cm."""
    dx_cm = float(delta_xyz[0]) * 100.0
    dy_cm = float(delta_xyz[1]) * 100.0
    dz_cm = float(delta_xyz[2]) * 100.0
    dist_cm = float(np.linalg.norm(delta_xyz)) * 100.0

    def _q(val: float) -> int:
        return int(max(1, round(abs(val) / quantize_cm) * quantize_cm))

    parts = []
    # Horizontal front/back (+x = front, -x = back) and left/right (+y = right, -y = left)
    fb = ("front" if dx_cm > 0 else "back") if abs(dx_cm) >= min_axis_cm else ""
    lr = ("right" if dy_cm > 0 else "left") if abs(dy_cm) >= min_axis_cm else ""
    horiz_cm = float(np.hypot(dx_cm, dy_cm))
    if fb and lr:
        dir_xy = f"{fb}-{lr}"
    else:
        dir_xy = fb or lr

    if dir_xy:
        if qualitative_only:
            parts.append(dir_xy)
        else:
            parts.append(f"{_q(horiz_cm)}cm {dir_xy}")
    else:
        parts.append("aligned horizontally")

    if abs(dz_cm) >= min_axis_cm:
        ud = "above" if dz_cm > 0 else "below"
        if qualitative_only:
            parts.append(f"{ud} gripper")
        else:
            parts.append(f"{_q(dz_cm)}cm {ud} gripper")
    else:
        parts.append("at gripper height")

    if qualitative_only:
        bucket = "very close (<5cm)" if dist_cm < 5.0 else ("near (~10cm)" if dist_cm < 15.0 else "far (>15cm)")
        return f"{', '.join(parts)} ({bucket})"
    return f"{', '.join(parts)} (dist {_q(dist_cm)}cm)"


def format_displacement_and_grip(
    delta_xyz: np.ndarray,
    grip_str: str,
    *,
    quantize_cm: int = 1,
) -> str:
    dx_cm, dy_cm, dz_cm = np.asarray(delta_xyz, dtype=np.float64) * 100.0
    horiz_cm = float(np.hypot(dx_cm, dy_cm))

    move_parts = []
    fb = ("front" if dx_cm > 0 else "back") if abs(dx_cm) >= 1.2 else ""
    lr = ("right" if dy_cm > 0 else "left") if abs(dy_cm) >= 1.2 else ""
    dir_xy = f"{fb}-{lr}" if (fb and lr) else (fb or lr)
    if dir_xy and horiz_cm >= 1.2:
        move_parts.append(f"{max(1, round(horiz_cm / quantize_cm) * quantize_cm)}cm {dir_xy}")
    if abs(dz_cm) >= 1.2:
        ud = "up" if dz_cm > 0 else "down"
        move_parts.append(f"{max(1, round(abs(dz_cm) / quantize_cm) * quantize_cm)}cm {ud}")

    motion_str = f"move {' and '.join(move_parts)}" if move_parts else "hold position / fine-align"
    return f"{motion_str}; {grip_str}"


def format_chunk_motion(
    eef_start: np.ndarray,
    eef_end: np.ndarray,
    grip_chunk: np.ndarray,
    prev_grip: float,
    *,
    quantize_cm: int = 1,
) -> str:
    """Summarize the 10-step end-effector motion `[t, t+10)` and gripper command."""
    delta = np.asarray(eef_end, dtype=np.float64) - np.asarray(eef_start, dtype=np.float64)
    g_signs = np.sign(grip_chunk)
    g0 = float(np.sign(prev_grip) if prev_grip != 0 else g_signs[0])
    if (g_signs > 0).any() and g0 <= 0:
        grip_str = "close gripper to grasp"
    elif (g_signs < 0).any() and g0 > 0:
        grip_str = "open gripper to release"
    elif g_signs[-1] > 0:
        grip_str = "keep gripper closed"
    else:
        grip_str = "keep gripper open"
    return format_displacement_and_grip(delta, grip_str, quantize_cm=quantize_cm)


def causal_chunk_motion(
    phase_tag: str,
    delta_target: np.ndarray | None,
    *,
    quantize_cm: int = 1,
) -> str:
    """Test-time compatible 10-step motion cue computed purely from `(phase_tag, delta_target)`."""
    if delta_target is None:
        return "hold position / fine-align; keep gripper open"
    d = np.asarray(delta_target, dtype=np.float64)
    if phase_tag == "reach":
        # Aim ~10cm above target
        aim = d + np.array([0.0, 0.0, 0.10])
        step = aim * min(1.0, 0.06 / max(float(np.linalg.norm(aim)), 1e-6))
        return format_displacement_and_grip(step, "keep gripper open", quantize_cm=quantize_cm)
    if phase_tag == "descend_grasp":
        step = d * min(1.0, 0.05 / max(float(np.linalg.norm(d)), 1e-6))
        return format_displacement_and_grip(step, "close gripper to grasp", quantize_cm=quantize_cm)
    if phase_tag == "lift":
        return format_displacement_and_grip(np.array([0.0, 0.0, 0.06]), "keep gripper closed", quantize_cm=quantize_cm)
    if phase_tag == "transport":
        aim = d + np.array([0.0, 0.0, 0.10])
        step = aim * min(1.0, 0.06 / max(float(np.linalg.norm(aim)), 1e-6))
        return format_displacement_and_grip(step, "keep gripper closed", quantize_cm=quantize_cm)
    if phase_tag == "lower_release":
        step = d * min(1.0, 0.05 / max(float(np.linalg.norm(d)), 1e-6))
        return format_displacement_and_grip(step, "open gripper to release", quantize_cm=quantize_cm)
    if phase_tag == "retract":
        return format_displacement_and_grip(np.array([0.0, 0.0, 0.05]), "keep gripper open", quantize_cm=quantize_cm)
    if phase_tag == "artic_approach":
        step = d * min(1.0, 0.06 / max(float(np.linalg.norm(d)), 1e-6))
        return format_displacement_and_grip(step, "keep gripper open", quantize_cm=quantize_cm)
    # artic_engage
    step = d * min(1.0, 0.05 / max(float(np.linalg.norm(d)), 1e-6))
    return format_displacement_and_grip(step, "engage fixture", quantize_cm=quantize_cm)


@dataclass
class SubgoalSpan:
    """One active goal span `[start, end)` within an episode."""
    start: int
    end: int
    predicate: str          # "in" | "on" | "open" | "close" | "turnon" | "turnoff"
    pick_name: str          # raw simulator name of first argument
    place_name: str | None  # raw simulator name of second argument (if in/on)
    close_step: int | None  # step where gripper closes on pick_name
    lift_step: int | None   # step where pick_name is lifted >= 3cm
    open_step: int | None   # step where gripper opens to release


def _instance_id(raw: str) -> str:
    """Extract the base object/fixture instance ID (e.g. `basket_1` from `basket_1_contain_region`)."""
    s = str(raw)
    for prefix in ("kitchen_table_", "living_room_table_", "study_table_", "main_table_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    m = re.match(r"^([a-zA-Z0-9_]+?_\d+)", s)
    return m.group(1) if m else s


def _distinguish_identical_names(
    names: list[str],
    positions_t0: dict[str, np.ndarray],
) -> dict[str, str]:
    """Give disambiguating spatial qualifiers only when distinct object instances share a clean name."""
    # Group by (clean_name, instance_id) so a container and its `*_contain_region` site
    # are recognized as the same physical object instance.
    base_instances: dict[str, dict[str, list[str]]] = {}
    for n in names:
        b = clean_name(n)
        inst = _instance_id(n)
        base_instances.setdefault(b, {}).setdefault(inst, []).append(n)

    out: dict[str, str] = {}
    for b, inst_map in base_instances.items():
        if len(inst_map) == 1:
            for k in next(iter(inst_map.values())):
                out[k] = b
            continue
        inst_keys = list(inst_map.keys())
        inst_pos = {
            inst: positions_t0[inst_map[inst][0]]
            for inst in inst_keys
            if inst_map[inst][0] in positions_t0
        }
        xs = [float(inst_pos[inst][0]) for inst in inst_keys if inst in inst_pos]
        ys = [float(inst_pos[inst][1]) for inst in inst_keys if inst in inst_pos]
        if len(xs) == len(inst_keys) and (max(xs) - min(xs) >= max(ys) - min(ys)):
            order = sorted(inst_keys, key=lambda inst: float(inst_pos[inst][0]), reverse=True)
            labels = ["front", "middle", "back"] if len(order) == 3 else ["front", "back"]
            for idx, inst in enumerate(order):
                tag = labels[min(idx, len(labels) - 1)]
                for k in inst_map[inst]:
                    out[k] = f"{tag} {b}"
        elif len(ys) == len(inst_keys):
            order = sorted(inst_keys, key=lambda inst: float(inst_pos[inst][1]))
            labels = ["left", "middle", "right"] if len(order) == 3 else ["left", "right"]
            for idx, inst in enumerate(order):
                tag = labels[min(idx, len(labels) - 1)]
                for k in inst_map[inst]:
                    out[k] = f"{tag} {b}"
        else:
            for inst in inst_keys:
                for k in inst_map[inst]:
                    out[k] = b
    return out


def compute_spans_from_subgoal_array(
    *,
    actions: np.ndarray,
    eef_pos: np.ndarray,
    goals_ordered: list[tuple[str, ...]],
    subgoal_arr: np.ndarray | None = None,
) -> list[SubgoalSpan]:
    """Build `SubgoalSpan`s from ordered goals and per-frame subgoal indices."""
    T = len(actions)
    G = len(goals_ordered)
    if G == 0:
        return []
    grip = np.sign(actions[:, 6])

    # Determine raw boundaries from subgoal_arr or equal/grip split
    bounds = [0]
    if subgoal_arr is not None and len(subgoal_arr) == T and G > 1:
        for k in range(1, G):
            idx = np.flatnonzero(subgoal_arr >= k)
            bounds.append(int(idx[0]) if len(idx) > 0 else T)
    elif G > 1:
        for k in range(1, G):
            bounds.append(int(round(k * T / G)))
    bounds.append(T)

    spans: list[SubgoalSpan] = []
    start = 0
    for idx_k in range(G):
        end = T if idx_k == G - 1 else max(start + 1, min(T, bounds[idx_k + 1]))
        goal = [str(x) for x in goals_ordered[idx_k]]
        pred = goal[0].lower()
        pick_name = goal[1] if len(goal) > 1 else ""
        place_name = goal[2] if (pred in ("in", "on") and len(goal) > 2) else None

        search_end = min(T, end + 15)
        seg_grip = grip[start:search_end]
        flips = np.flatnonzero(seg_grip[1:] != seg_grip[:-1]) + 1 + start
        closes = [int(i) for i in flips if grip[i] > 0]
        opens = [int(i) for i in flips if grip[i] < 0]

        close_step = closes[0] if closes else None
        open_step = next((o for o in opens if close_step is None or o > close_step), None)
        if pred in ("in", "on") and open_step is not None and idx_k < G - 1:
            end = max(end, min(T, open_step + 8))

        lift_step = None
        if close_step is not None:
            grasp_settle = max(close_step, min(close_step + 8, T - 1))
            z_min = float(np.min(eef_pos[close_step : grasp_settle + 1, 2]))
            for t_scan in range(grasp_settle, min(close_step + 50, max(end, grasp_settle + 1))):
                if t_scan < T and float(eef_pos[t_scan, 2]) - z_min >= 0.03:
                    lift_step = t_scan
                    break
            if lift_step is None:
                lift_step = min(close_step + 15, T - 1)

        spans.append(
            SubgoalSpan(
                start=start,
                end=end,
                predicate=pred,
                pick_name=pick_name,
                place_name=place_name,
                close_step=close_step,
                lift_step=lift_step,
                open_step=open_step,
            )
        )
        start = end
    return spans


def build_compact_episode_meta(
    *,
    prompt: str,
    actions: np.ndarray,
    eef_pos: np.ndarray,
    spans: list[SubgoalSpan],
    static_positions: dict[str, np.ndarray],
    recovery_len: int = 0,
) -> dict:
    """Build a memory-compact episode metadata dict (~1.2 KB per episode).

    `static_positions` maps object and target names to their 3D world coordinates `(3,)`.
    """
    T = len(actions)
    pos_clean = {k: np.asarray(v, dtype=np.float32).reshape(3) for k, v in static_positions.items()}
    name_map = _distinguish_identical_names(list(pos_clean.keys()), pos_clean)
    return {
        "T": int(T),
        "prompt": str(prompt),
        "eef_pos": np.asarray(eef_pos, dtype=np.float16),
        "grip": np.sign(actions[:, 6]).astype(np.int8),
        "spans": spans,
        "pos": pos_clean,
        "name_map": name_map,
        "recovery_len": int(recovery_len),
    }


def build_episode_dense_metadata(
    *,
    actions: np.ndarray,
    eef_pos: np.ndarray,
    goals: list[tuple[str, ...]],
    goal_held: np.ndarray,
    obj_positions: dict[str, np.ndarray],
    target_positions: dict[str, np.ndarray],
    recovery_len: int = 0,
    prompt: str = "",
) -> dict:
    """Build frame-aligned dense-caption metadata for an episode of length `T`."""
    T = len(actions)
    G = len(goals)
    done_at = []
    for g in range(G):
        if G == 0 or goal_held.shape[1] <= g:
            done_at.append(T)
        elif goal_held[:, g].all():
            done_at.append(0)
        else:
            done_at.append(T - int(np.argmin(goal_held[::-1, g])))
    order = sorted(range(G), key=lambda g: done_at[g]) if G > 0 else []
    goals_ordered = [goals[g] for g in order]
    subgoal_arr = np.zeros(T, dtype=np.int16)
    s_idx = 0
    for k, g in enumerate(order):
        e_idx = T if k == len(order) - 1 else max(s_idx, done_at[g])
        subgoal_arr[s_idx:e_idx] = k
        s_idx = e_idx

    spans = compute_spans_from_subgoal_array(
        actions=actions,
        eef_pos=eef_pos,
        goals_ordered=goals_ordered,
        subgoal_arr=subgoal_arr,
    )
    all_pos_t0 = {k: np.asarray(v[0], dtype=np.float32) for k, v in {**target_positions, **obj_positions}.items()}
    compact = build_compact_episode_meta(
        prompt=prompt,
        actions=actions,
        eef_pos=eef_pos,
        spans=spans,
        static_positions=all_pos_t0,
        recovery_len=recovery_len,
    )
    compact["obj_positions"] = {k: np.asarray(v, dtype=np.float32) for k, v in obj_positions.items()}
    compact["target_positions"] = {k: np.asarray(v, dtype=np.float32) for k, v in target_positions.items()}
    return compact


def render_dense_caption(
    meta: dict,
    t: int,
    *,
    horizon: int = HORIZON,
    pos_noise_std_m: float = 0.0,
    qualitative_only: bool = False,
    use_causal_motion: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, str]:
    """Render both structured fields and the combined dense caption for chunk `[t, t + horizon)`.

    Supports both full per-frame `meta` and compact `meta` (`build_compact_episode_meta`).
    """
    T = int(meta["T"])
    t = int(np.clip(t, 0, T - 1))
    t_end = min(T - 1, t + horizon)
    eef_t = np.asarray(meta["eef_pos"][t], dtype=np.float64)
    eef_end = np.asarray(meta["eef_pos"][t_end], dtype=np.float64)
    if "actions" in meta:
        grip_chunk = meta["actions"][t : min(T, t + horizon), 6]
        prev_grip = float(meta["actions"][t - 1, 6]) if t > 0 else float(meta["actions"][0, 6])
    else:
        grip_chunk = meta["grip"][t : min(T, t + horizon)]
        prev_grip = float(meta["grip"][t - 1]) if t > 0 else float(meta["grip"][0])
    name_map: dict[str, str] = meta["name_map"]

    def _noisy(pos: np.ndarray) -> np.ndarray:
        p = np.asarray(pos, dtype=np.float64).copy()
        if pos_noise_std_m > 0.0 and rng is not None:
            p += rng.normal(0.0, pos_noise_std_m, size=3)
        return p

    if "obj_positions" in meta and "target_positions" in meta:
        obj_pos_t = {k: _noisy(v[t]) for k, v in meta["obj_positions"].items()}
        tgt_pos_t = {k: _noisy(v[t]) for k, v in meta["target_positions"].items()}
        all_t = {**obj_pos_t, **tgt_pos_t}
    else:
        all_t = {k: _noisy(v) for k, v in meta["pos"].items()}

    if all_t:
        center_xy = np.mean([p[:2] for p in all_t.values()], axis=0)
    else:
        center_xy = np.zeros(2)

    # Order scene items: goal-referenced objects first, then other scene objects
    spans: list[SubgoalSpan] = meta["spans"]
    goal_keys = []
    for s in spans:
        if s.pick_name and s.pick_name in all_t:
            goal_keys.append(s.pick_name)
        if s.place_name and s.place_name in all_t:
            goal_keys.append(s.place_name)
    ordered_keys = goal_keys + [k for k in all_t if k not in goal_keys]

    scene_items = []
    seen_labels = set()
    for k in ordered_keys:
        p = all_t[k]
        label = name_map.get(k, clean_name(k))
        if label in seen_labels:
            continue
        seen_labels.add(label)
        scene_items.append(f"{label} ({table_sector(p, center_xy)})")
        if len(scene_items) >= 5:
            break
    scene_str = ", ".join(scene_items) if scene_items else "tabletop workspace"

    span = next((s for s in spans if s.start <= t < s.end), spans[-1] if spans else None)
    phase_tag = "reach"
    active_delta = None

    if span is None:
        subgoal_str = "hold position"
        geom_str = "no active target"
    else:
        pick_label = name_map.get(span.pick_name, clean_name(span.pick_name))
        place_label = name_map.get(span.place_name, clean_name(span.place_name)) if span.place_name else None
        p_pick = all_t.get(span.pick_name)
        p_place = all_t.get(span.place_name) if span.place_name else None

        in_recovery = t < int(meta.get("recovery_len", 0))
        if span.predicate in ("in", "on"):
            prep = "into" if span.predicate == "in" else "onto"
            c_step = span.close_step
            l_step = span.lift_step
            o_step = span.open_step

            if c_step is None or t < c_step:
                active_target_label = pick_label
                active_target_pos = p_pick
                sec_label, sec_pos = place_label, p_place
                if in_recovery:
                    phase_tag = "reach"
                    subgoal_str = f"recover posture and move toward {pick_label}"
                elif c_step is not None and t + horizon >= c_step:
                    phase_tag = "descend_grasp"
                    subgoal_str = f"descend to {pick_label} and close gripper to grasp"
                elif p_pick is not None and np.hypot(p_pick[0] - eef_t[0], p_pick[1] - eef_t[1]) < 0.045:
                    phase_tag = "descend_grasp"
                    subgoal_str = f"lower gripper straight down to {pick_label}"
                else:
                    phase_tag = "reach"
                    subgoal_str = f"reach above {pick_label}"
            elif l_step is not None and t < l_step:
                active_target_label = place_label or pick_label
                active_target_pos = p_place if p_place is not None else p_pick
                sec_label, sec_pos = None, None
                phase_tag = "lift"
                subgoal_str = f"lift grasped {pick_label} up from surface"
            elif o_step is None or t < o_step:
                active_target_label = place_label or pick_label
                active_target_pos = p_place if p_place is not None else p_pick
                sec_label, sec_pos = None, None
                if in_recovery:
                    phase_tag = "transport"
                    subgoal_str = f"carry {pick_label} and recover path toward {place_label}"
                elif o_step is not None and t + horizon >= o_step:
                    phase_tag = "lower_release"
                    subgoal_str = f"lower {pick_label} {prep} {place_label} and open gripper"
                elif p_place is not None and np.hypot(p_place[0] - eef_t[0], p_place[1] - eef_t[1]) < 0.05:
                    phase_tag = "lower_release"
                    subgoal_str = f"align and lower {pick_label} {prep} {place_label}"
                else:
                    phase_tag = "transport"
                    subgoal_str = f"transport held {pick_label} toward {place_label}"
            else:
                active_target_label = place_label or pick_label
                active_target_pos = p_place if p_place is not None else p_pick
                sec_label, sec_pos = None, None
                phase_tag = "retract"
                subgoal_str = f"release {pick_label} {prep} {place_label} and retract gripper"

            geom_parts = []
            if active_target_pos is not None:
                active_delta = active_target_pos - eef_t
                role_tag = "destination" if (c_step is not None and t >= c_step) else "target"
                geom_parts.append(
                    f"{role_tag} {active_target_label} is "
                    f"{format_delta_3d(active_delta, qualitative_only=qualitative_only)}"
                )
            if sec_label and sec_pos is not None:
                geom_parts.append(
                    f"destination {sec_label} is "
                    f"{format_delta_3d(sec_pos - eef_t, qualitative_only=qualitative_only)}"
                )
            geom_str = "; ".join(geom_parts) if geom_parts else "target in view"

        else:
            verb = {"open": "pull open", "close": "push closed", "turnon": "turn on", "turnoff": "turn off"}.get(
                span.predicate, span.predicate
            )
            dist = float(np.linalg.norm(p_pick - eef_t)) if p_pick is not None else 0.0
            if in_recovery:
                phase_tag = "artic_approach"
                subgoal_str = f"recover posture and approach {pick_label}"
            elif dist > 0.09:
                phase_tag = "artic_approach"
                subgoal_str = f"approach {pick_label} to {verb}"
            else:
                phase_tag = "artic_engage"
                subgoal_str = f"engage and {verb} {pick_label}"
            if p_pick is not None:
                active_delta = p_pick - eef_t
                geom_str = (
                    f"target {pick_label} is "
                    f"{format_delta_3d(active_delta, qualitative_only=qualitative_only)}"
                )
            else:
                geom_str = f"target {pick_label}"

    if use_causal_motion:
        motion_str = causal_chunk_motion(phase_tag, active_delta)
    else:
        motion_str = format_chunk_motion(eef_t, eef_end, grip_chunk, prev_grip)

    full_caption = (
        f"scene: {scene_str}. "
        f"subgoal: {subgoal_str}. "
        f"geometry: {geom_str}. "
        f"next 10 steps: {motion_str}."
    )
    return {
        "scene": scene_str,
        "subgoal": subgoal_str,
        "geometry": geom_str,
        "motion": motion_str,
        "caption": full_caption,
    }
