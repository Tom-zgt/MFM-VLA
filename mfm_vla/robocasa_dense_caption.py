"""Dense-Caption generator and stage segmenter for RoboCasa365 (`PandaOmron`).

Unifies training-time dense captioning (from human demonstrations and G5j hand-off
recovery rollouts) with test-time dynamic prompt conditioning inside `eval_robocasa.py`.

Coordinate convention (Robot Base Frame `R_base`, matching `state[0:3]` and `action[0:3]`):
  +x = forward (towards workspace in front of robot), -x = backward (towards robot body)
  +y = left,                                          -y = right
  +z = up / above,                                    -z = down / below
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def quat_xyzw_to_rotmat(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert `(x, y, z, w)` quaternion to 3x3 rotation matrix."""
    q = np.asarray(quat_xyzw, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        return np.eye(3, dtype=np.float64)
    return Rotation.from_quat(q / norm).as_matrix()


def compute_world_eef_trajectory(states_16d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute world-frame EEF positions `(T, 3)` and base rotation matrices `(T, 3, 3)`
    from OpenPI-ordered 16-D RoboCasa states:
      `[0:3]`   = `eef_pos_rel` (in base frame)
      `[3:7]`   = `eef_quat_rel`
      `[7:10]`  = `base_pos` (in world frame)
      `[10:14]` = `base_quat` (xyzw in world frame)
      `[14:16]` = `gripper_qpos`
    """
    T = len(states_16d)
    eef_world = np.zeros((T, 3), dtype=np.float64)
    R_bases = np.zeros((T, 3, 3), dtype=np.float64)
    for t in range(T):
        eef_rel = states_16d[t, 0:3]
        base_pos = states_16d[t, 7:10]
        base_quat = states_16d[t, 10:14]
        R_b = quat_xyzw_to_rotmat(base_quat)
        R_bases[t] = R_b
        eef_world[t] = base_pos + R_b @ eef_rel
    return eef_world, R_bases


def format_base_delta_3d(
    delta_base_xyz: np.ndarray,
    *,
    quantize_cm: int = 1,
    min_axis_cm: float = 1.5,
) -> str:
    """Format `R_base.T @ (target_world - eef_world)` in robot-base coordinates:
      `+x` = forward, `-x` = backward
      `+y` = left,    `-y` = right
      `+z` = above,   `-z` = below
    """
    dx_cm = float(delta_base_xyz[0]) * 100.0
    dy_cm = float(delta_base_xyz[1]) * 100.0
    dz_cm = float(delta_base_xyz[2]) * 100.0
    dist_cm = float(np.linalg.norm(delta_base_xyz)) * 100.0

    def _q(val: float) -> int:
        return int(max(1, round(abs(val) / quantize_cm) * quantize_cm))

    parts: list[str] = []
    fb = ("forward" if dx_cm > 0 else "backward") if abs(dx_cm) >= min_axis_cm else ""
    lr = ("left" if dy_cm > 0 else "right") if abs(dy_cm) >= min_axis_cm else ""
    horiz_cm = float(np.hypot(dx_cm, dy_cm))
    if fb and lr:
        parts.append(f"{_q(horiz_cm)}cm {fb}-{lr}")
    elif fb or lr:
        parts.append(f"{_q(horiz_cm)}cm {fb or lr}")
    else:
        parts.append("aligned horizontally")

    if abs(dz_cm) >= min_axis_cm:
        ud = "above" if dz_cm > 0 else "below"
        parts.append(f"{_q(dz_cm)}cm {ud}")
    else:
        parts.append("level")

    return f"{', '.join(parts)} ({_q(dist_cm)}cm)"


def format_next10_base_motion(
    delta_base_10: np.ndarray,
    grip_cmd_now: float,
    grip_cmd_10: np.ndarray,
) -> str:
    """Summarize the next 10 steps of base-frame EEF motion and gripper action."""
    dx_cm, dy_cm, dz_cm = np.asarray(delta_base_10, dtype=np.float64) * 100.0
    horiz_cm = float(np.hypot(dx_cm, dy_cm))
    move_parts: list[str] = []
    fb = ("forward" if dx_cm > 0 else "backward") if abs(dx_cm) >= 1.2 else ""
    lr = ("left" if dy_cm > 0 else "right") if abs(dy_cm) >= 1.2 else ""
    dir_xy = f"{fb}-{lr}" if (fb and lr) else (fb or lr)
    if dir_xy and horiz_cm >= 1.2:
        move_parts.append(f"{int(max(1, round(horiz_cm)))}cm {dir_xy}")
    if abs(dz_cm) >= 1.2:
        ud = "up" if dz_cm > 0 else "down"
        move_parts.append(f"{int(max(1, round(abs(dz_cm))))}cm {ud}")
    motion_str = f"move {' '.join(move_parts)}" if move_parts else "align"

    end_close = float(grip_cmd_10[-1]) > 0.0 if len(grip_cmd_10) > 0 else (grip_cmd_now > 0.0)
    now_close = grip_cmd_now > 0.0
    if not now_close and end_close:
        grip_str = "grasp"
    elif now_close and not end_close:
        grip_str = "release"
    elif end_close:
        grip_str = "hold"
    else:
        grip_str = "open"
    return f"{motion_str}, {grip_str}"


def detect_grasp_release_intervals(
    states_16d: np.ndarray,
    actions_12d: np.ndarray,
    eef_world: np.ndarray,
    min_hold_frames: int = 12,
    min_disp_m: float = 0.05,
) -> List[Tuple[int, int]]:
    """Detect `(grasp_frame, release_frame)` intervals from `gripper_close` (`actions[:, 6]`)."""
    T = len(actions_12d)
    grip_close = actions_12d[:, 6] > 0.0
    raw_runs: List[Tuple[int, int]] = []
    i = 0
    while i < T:
        if grip_close[i]:
            j = i + 1
            while j < T and grip_close[j]:
                j += 1
            if j - i >= min_hold_frames:
                raw_runs.append((i, min(j, T - 1)))
            i = j
        else:
            i += 1

    # Merge adjacent runs separated by brief (< 8 frame) glitches
    merged: List[Tuple[int, int]] = []
    for s, e in raw_runs:
        if merged and s - merged[-1][1] < 8:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    # Filter by minimum EEF displacement or lift during hold
    valid: List[Tuple[int, int]] = []
    for s, e in merged:
        seg = eef_world[s : e + 1]
        max_disp = float(np.linalg.norm(seg - seg[0], axis=1).max()) if len(seg) > 1 else 0.0
        if max_disp >= min_disp_m:
            valid.append((s, e))
    return valid


def task_stage_templates(task_name: str, instruction: str) -> List[Dict]:
    """Return canonical ordered stage specs matching G5j's planner for `task_name`."""
    inst = instruction.strip()
    inst_low = inst.lower().rstrip(".")

    if task_name == "CloseFridge":
        return [{
            "kind": "fixture_act",
            "approach": "front_horizontal",
            "grasp_target": "fridge door",
            "place_target": None,
            "stage_instruction": "close the fridge door",
        }]
    if task_name == "CloseToasterOvenDoor":
        return [{
            "kind": "fixture_act",
            "approach": "front_horizontal",
            "grasp_target": "toaster oven door",
            "place_target": None,
            "stage_instruction": "close the toaster oven door",
        }]
    if task_name == "TurnOnMicrowave":
        return [{
            "kind": "fixture_act",
            "approach": "front_horizontal",
            "grasp_target": "microwave start button",
            "place_target": None,
            "stage_instruction": "press the start button on the microwave",
        }]
    if task_name == "TurnOffStove":
        return [{
            "kind": "fixture_act",
            "approach": "front_horizontal",
            "grasp_target": "stove knob",
            "place_target": None,
            "stage_instruction": inst_low or "turn off the stove burner knob",
        }]
    if task_name == "PickPlaceCounterToCabinet":
        m = re.search(r"pick the (.+?) from the counter and place it in the cabinet", inst_low)
        obj = m.group(1) if m else "object on the counter"
        return [{
            "kind": "pick_place",
            "approach": "top_down",
            "grasp_target": obj,
            "place_target": "cabinet",
            "stage_instruction": f"pick the {obj} from the counter and place it in the cabinet",
        }]
    if task_name == "PickPlaceCounterToStove":
        m = re.search(r"pick the (.+?) from the (.+?) and place it in the (.+)", inst_low)
        obj = m.group(1) if m else "food item"
        src = m.group(2) if m else "counter"
        dst = m.group(3) if m else "pan on the stove"
        return [{
            "kind": "pick_place",
            "approach": "top_down",
            "grasp_target": obj,
            "place_target": dst,
            "stage_instruction": f"pick the {obj} from the {src} and place it in the {dst}",
        }]
    if task_name == "StoreLeftoversInBowl":
        m = re.search(r"pick the chicken drumstick and (.+?) from their plates", inst_low)
        veg = m.group(1) if m else "vegetable"
        return [
            {
                "kind": "pick_place",
                "approach": "top_down",
                "grasp_target": "chicken drumstick",
                "place_target": "bowl",
                "stage_instruction": "pick the chicken drumstick from the plate and place it in the bowl",
            },
            {
                "kind": "pick_place",
                "approach": "top_down",
                "grasp_target": veg,
                "place_target": "bowl",
                "stage_instruction": f"pick the {veg} from the plate and place it in the bowl",
            },
            {
                "kind": "pick_place",
                "approach": "top_down",
                "grasp_target": "bowl",
                "place_target": "fridge",
                "stage_instruction": "pick up the bowl by its rim and place it in the fridge",
            },
        ]
    if task_name == "PrepareCoffee":
        return [
            {
                "kind": "pick_place",
                "approach": "top_down",
                "grasp_target": "mug",
                "place_target": "coffee machine dispenser",
                "stage_instruction": "pick the mug from the cabinet and place it under the coffee machine dispenser",
            },
            {
                "kind": "fixture_act",
                "approach": "front_horizontal",
                "grasp_target": "coffee machine start button",
                "place_target": None,
                "stage_instruction": "press the start button on the coffee machine",
            },
        ]
    if task_name == "GetToastedBread":
        return [
            {
                "kind": "fixture_act",
                "approach": "front_horizontal",
                "grasp_target": "toaster lever",
                "place_target": None,
                "stage_instruction": "push down the lever of the toaster to start it",
            },
            {
                "kind": "pick_place",
                "approach": "top_down",
                "grasp_target": "toasted bread",
                "place_target": "plate on the counter",
                "stage_instruction": "pick up the toasted bread from the toaster and place it on the plate",
            },
        ]
    if task_name == "LoadDishwasher":
        return [
            {
                "kind": "pick_place",
                "approach": "top_down",
                "grasp_target": "cup",
                "place_target": "dishwasher rack",
                "stage_instruction": "pick up the cup from the counter and place it in the dishwasher",
            },
            {
                "kind": "pick_place",
                "approach": "top_down",
                "grasp_target": "bowl",
                "place_target": "dishwasher rack",
                "stage_instruction": "pick up the bowl from the counter and place it in the dishwasher",
            },
            {
                "kind": "fixture_act",
                "approach": "front_horizontal",
                "grasp_target": "dishwasher door",
                "place_target": None,
                "stage_instruction": "close the dishwasher door",
            },
        ]
    if task_name == "SteamInMicrowave":
        m = re.search(r"pick the (.+?) from the sink and place it in the bowl", inst_low)
        veg = m.group(1) if m else "vegetable"
        return [
            {
                "kind": "pick_place",
                "approach": "top_down",
                "grasp_target": veg,
                "place_target": "bowl",
                "stage_instruction": f"pick the {veg} from the sink and place it in the bowl",
            },
            {
                "kind": "pick_place",
                "approach": "top_down",
                "grasp_target": "bowl",
                "place_target": "microwave",
                "stage_instruction": "pick up the bowl by its rim and place it in the microwave",
            },
            {
                "kind": "fixture_act",
                "approach": "front_horizontal",
                "grasp_target": "microwave door and start button",
                "place_target": None,
                "stage_instruction": "close the microwave door and press the start button",
            },
        ]
    if task_name == "StackBowlsCabinet":
        return [
            {
                "kind": "pick_place",
                "approach": "top_down",
                "grasp_target": "smaller bowl",
                "place_target": "larger bowl",
                "stage_instruction": "pick up the smaller bowl on the counter and stack it on the larger bowl",
            },
            {
                "kind": "pick_place",
                "approach": "top_down",
                "grasp_target": "stacked bowls",
                "place_target": "open cabinet",
                "stage_instruction": "pick up the bowls and place them in the open cabinet",
            },
        ]

    # --- Remaining 12 Atomic-Seen tasks ---
    if task_name == "CloseBlenderLid":
        return [{
            "kind": "pick_place",
            "approach": "top_down",
            "grasp_target": "blender lid",
            "place_target": "blender",
            "stage_instruction": "pick up the blender lid and place it securely on top of the blender",
        }]
    if task_name == "CoffeeSetupMug":
        return [{
            "kind": "pick_place",
            "approach": "top_down",
            "grasp_target": "mug",
            "place_target": "coffee machine dispenser",
            "stage_instruction": inst_low or "pick the mug and place it under the coffee machine dispenser",
        }]
    if task_name == "NavigateKitchen":
        return [{
            "kind": "fixture_act",
            "approach": "front_horizontal",
            "grasp_target": "target fixture",
            "place_target": None,
            "stage_instruction": inst_low or "navigate to the target kitchen fixture",
        }]
    if task_name == "OpenCabinet":
        return [{
            "kind": "fixture_act",
            "approach": "front_horizontal",
            "grasp_target": "cabinet door handle",
            "place_target": None,
            "stage_instruction": inst_low or "open the cabinet door",
        }]
    if task_name == "OpenDrawer":
        return [{
            "kind": "fixture_act",
            "approach": "front_horizontal",
            "grasp_target": "drawer handle",
            "place_target": None,
            "stage_instruction": inst_low or "open the drawer",
        }]
    if task_name == "OpenStandMixerHead":
        return [{
            "kind": "fixture_act",
            "approach": "front_horizontal",
            "grasp_target": "stand mixer head",
            "place_target": None,
            "stage_instruction": "open the stand mixer head",
        }]
    if task_name == "PickPlaceDrawerToCounter":
        m = re.search(r"pick the (.+?) from the drawer", inst_low)
        obj = m.group(1) if m else "object in the drawer"
        return [{
            "kind": "pick_place",
            "approach": "top_down",
            "grasp_target": obj,
            "place_target": "counter",
            "stage_instruction": inst_low or f"pick the {obj} from the drawer and place it on the counter",
        }]
    if task_name == "PickPlaceSinkToCounter":
        m = re.search(r"pick the (.+?) from the sink", inst_low)
        obj = m.group(1) if m else "object in the sink"
        return [{
            "kind": "pick_place",
            "approach": "top_down",
            "grasp_target": obj,
            "place_target": "container on the counter",
            "stage_instruction": inst_low or f"pick the {obj} from the sink and place it on the counter",
        }]
    if task_name == "PickPlaceToasterToCounter":
        return [{
            "kind": "pick_place",
            "approach": "top_down",
            "grasp_target": "toasted bread",
            "place_target": "plate on the counter",
            "stage_instruction": inst_low or "pick the toasted item from the toaster and place it on the plate",
        }]
    if task_name == "SlideDishwasherRack":
        return [{
            "kind": "fixture_act",
            "approach": "front_horizontal",
            "grasp_target": "top dishwasher rack",
            "place_target": None,
            "stage_instruction": inst_low or "slide the top dishwasher rack",
        }]
    if task_name == "TurnOnElectricKettle":
        return [{
            "kind": "fixture_act",
            "approach": "top_down",
            "grasp_target": "electric kettle lever",
            "place_target": None,
            "stage_instruction": "press down the lever to turn on the electric kettle",
        }]
    if task_name == "TurnOnSinkFaucet":
        return [{
            "kind": "fixture_act",
            "approach": "front_horizontal",
            "grasp_target": "sink faucet handle",
            "place_target": None,
            "stage_instruction": inst_low or "turn on the sink faucet",
        }]

    # --- Remaining 10 Composite-Seen tasks ---
    if task_name == "DeliverStraw":
        return [
            {
                "kind": "fixture_act",
                "approach": "front_horizontal",
                "grasp_target": "drawer handle",
                "place_target": None,
                "stage_instruction": "open the drawer containing the straw",
            },
            {
                "kind": "pick_place",
                "approach": "top_down",
                "grasp_target": "straw",
                "place_target": "glass cup on the dining counter",
                "stage_instruction": "pick the straw from the drawer and place it inside the glass cup",
            },
        ]
    if task_name == "KettleBoiling":
        return [
            {
                "kind": "pick_place",
                "approach": "top_down",
                "grasp_target": "kettle",
                "place_target": "stove burner",
                "stage_instruction": "pick the kettle from the counter and place it on a stove burner",
            },
            {
                "kind": "fixture_act",
                "approach": "front_horizontal",
                "grasp_target": "stove burner knob",
                "place_target": None,
                "stage_instruction": "turn the stove burner knob on",
            },
        ]
    if task_name == "PackIdenticalLunches":
        return [
            {
                "kind": "pick_place",
                "approach": "top_down",
                "grasp_target": "first food item",
                "place_target": "first tupperware",
                "stage_instruction": "pick the first food item and place it in the first tupperware",
            },
            {
                "kind": "pick_place",
                "approach": "top_down",
                "grasp_target": "second food item",
                "place_target": "tupperware",
                "stage_instruction": "place the remaining food items into the two tupperware containers",
            },
        ]
    if task_name == "PreSoakPan":
        return [
            {
                "kind": "pick_place",
                "approach": "top_down",
                "grasp_target": "pan",
                "place_target": "sink basin",
                "stage_instruction": "pick the pan and place it into the sink",
            },
            {
                "kind": "pick_place",
                "approach": "top_down",
                "grasp_target": "sponge",
                "place_target": "pan in the sink",
                "stage_instruction": "pick the sponge and place it into the pan in the sink",
            },
            {
                "kind": "fixture_act",
                "approach": "front_horizontal",
                "grasp_target": "sink faucet handle",
                "place_target": None,
                "stage_instruction": "turn on the sink faucet water",
            },
        ]
    if task_name == "RinseSinkBasin":
        return [
            {
                "kind": "fixture_act",
                "approach": "front_horizontal",
                "grasp_target": "sink faucet handle",
                "place_target": None,
                "stage_instruction": "turn on the sink faucet",
            },
            {
                "kind": "fixture_act",
                "approach": "front_horizontal",
                "grasp_target": "sink spout",
                "place_target": None,
                "stage_instruction": "maneuver the sink spout to rinse both sides of the sink basin",
            },
        ]
    if task_name == "ScrubCuttingBoard":
        return [
            {
                "kind": "pick_place",
                "approach": "top_down",
                "grasp_target": "sponge",
                "place_target": "cutting board",
                "stage_instruction": "pick up the sponge from the counter and scrub the cutting board",
            },
        ]
    if task_name == "SearingMeat":
        return [
            {
                "kind": "pick_place",
                "approach": "top_down",
                "grasp_target": "pan",
                "place_target": "stove burner",
                "stage_instruction": "pick the pan from the cabinet and place it on the stove burner",
            },
            {
                "kind": "pick_place",
                "approach": "top_down",
                "grasp_target": "meat",
                "place_target": "pan on the stove",
                "stage_instruction": "pick the meat and place it in the pan on the stove",
            },
            {
                "kind": "fixture_act",
                "approach": "front_horizontal",
                "grasp_target": "stove burner knob",
                "place_target": None,
                "stage_instruction": "turn the stove burner knob on",
            },
        ]
    if task_name == "SetUpCuttingStation":
        return [
            {
                "kind": "pick_place",
                "approach": "top_down",
                "grasp_target": "knife",
                "place_target": "cutting board",
                "stage_instruction": "pick up the knife from the drawer and place it on the cutting board",
            },
            {
                "kind": "pick_place",
                "approach": "top_down",
                "grasp_target": "meat",
                "place_target": "cutting board",
                "stage_instruction": "pick the meat from the plate and place it on the cutting board",
            },
        ]
    if task_name == "StirVegetables":
        return [
            {
                "kind": "pick_place",
                "approach": "top_down",
                "grasp_target": "first vegetable",
                "place_target": "pot on the stove",
                "stage_instruction": "pick the first vegetable and place it in the pot",
            },
            {
                "kind": "pick_place",
                "approach": "top_down",
                "grasp_target": "second vegetable",
                "place_target": "pot on the stove",
                "stage_instruction": "pick the second vegetable and place it in the pot",
            },
            {
                "kind": "pick_place",
                "approach": "top_down",
                "grasp_target": "spatula",
                "place_target": "pot on the stove",
                "stage_instruction": "retrieve the spatula and stir the vegetables in the pot",
            },
        ]
    if task_name == "WashLettuce":
        return [
            {
                "kind": "fixture_act",
                "approach": "front_horizontal",
                "grasp_target": "sink faucet handle",
                "place_target": None,
                "stage_instruction": "turn on the sink faucet water",
            },
            {
                "kind": "pick_place",
                "approach": "top_down",
                "grasp_target": "lettuce",
                "place_target": "running water in the sink",
                "stage_instruction": "pick up the lettuce in the sink and wash it under the running water",
            },
        ]

    # Generic fallback
    return [{
        "kind": "pick_place",
        "approach": "top_down",
        "grasp_target": "target object",
        "place_target": "destination",
        "stage_instruction": inst_low,
    }]


def build_episode_dense_meta(
    task_name: str,
    instruction: str,
    states_16d: np.ndarray,
    actions_12d: np.ndarray,
) -> Dict:
    """Segment a demonstration or hand-off rollout into stages and compute per-stage
    3D target points and priority contact windows."""
    T = len(states_16d)
    eef_world, R_bases = compute_world_eef_trajectory(states_16d)
    templates = task_stage_templates(task_name, instruction)
    intervals = detect_grasp_release_intervals(states_16d, actions_12d, eef_world)

    stages: List[Dict] = []
    n_tmpl = len(templates)
    pick_Idx = 0
    cursor = 0

    for k, tmpl in enumerate(templates):
        is_last = k == n_tmpl - 1
        if tmpl["kind"] == "pick_place" and pick_Idx < len(intervals):
            g_t, r_t = intervals[pick_Idx]
            pick_Idx += 1
            start_t = cursor
            end_t = T if (is_last and pick_Idx >= len(intervals)) else min(T, r_t + 12)
            g_t = int(np.clip(g_t, start_t, max(start_t, end_t - 1)))
            r_t = int(np.clip(r_t, g_t, max(g_t, end_t - 1)))
            stages.append({
                **tmpl,
                "stage_idx": k,
                "num_stages": n_tmpl,
                "start_t": int(start_t),
                "grasp_t": int(g_t),
                "release_t": int(r_t),
                "end_t": int(end_t),
                "grasp_pos_world": eef_world[g_t].astype(np.float32),
                "place_pos_world": eef_world[r_t].astype(np.float32),
            })
            cursor = end_t
        else:
            start_t = cursor
            if not is_last and pick_Idx < len(intervals):
                end_t = max(start_t + 10, intervals[pick_Idx][0] - 25)
            else:
                end_t = T
            end_t = int(np.clip(end_t, start_t + 1, T))
            # Contact apex: maximum forward extension (+x in base frame) inside [start_t, end_t)
            seg_rel_x = states_16d[start_t:end_t, 0]
            apex_offset = int(np.argmax(seg_rel_x)) if len(seg_rel_x) > 0 else 0
            contact_t = int(np.clip(start_t + apex_offset, start_t, end_t - 1))
            stages.append({
                **tmpl,
                "stage_idx": k,
                "num_stages": n_tmpl,
                "start_t": int(start_t),
                "grasp_t": int(contact_t),
                "release_t": int(end_t - 1),
                "end_t": int(end_t),
                "grasp_pos_world": eef_world[contact_t].astype(np.float32),
                "place_pos_world": eef_world[end_t - 1].astype(np.float32),
            })
            cursor = end_t

    if not stages:
        stages.append({
            **templates[0],
            "stage_idx": 0,
            "num_stages": 1,
            "start_t": 0,
            "grasp_t": T // 2,
            "release_t": T - 1,
            "end_t": T,
            "grasp_pos_world": eef_world[T // 2].astype(np.float32),
            "place_pos_world": eef_world[T - 1].astype(np.float32),
        })
    else:
        stages[-1]["end_t"] = T

    # Identify pre-contact hand-off frames (for oversampling in data loader)
    handoff_frames: List[int] = []
    for stg in stages:
        gt = stg["grasp_t"]
        lo = max(stg["start_t"], gt - 38)
        hi = min(stg["end_t"] - 1, gt + 22)
        handoff_frames.extend(range(lo, hi + 1))
        if stg["kind"] == "pick_place":
            rt = stg["release_t"]
            r_lo = max(gt, rt - 25)
            r_hi = min(stg["end_t"] - 1, rt + 10)
            handoff_frames.extend(range(r_lo, r_hi + 1))

    return {
        "task_name": task_name,
        "instruction": instruction,
        "num_frames": T,
        "eef_world": eef_world.astype(np.float32),
        "R_bases": R_bases.astype(np.float32),
        "grip_cmd": actions_12d[:, 6].astype(np.float32),
        "stages": stages,
        "handoff_frames": sorted(set(handoff_frames)),
    }


def render_robocasa_dense_prompt(
    meta: Dict,
    frame_idx: int,
    *,
    mode: str = "full",
    pos_noise_std_m: float = 0.018,
    rng: Optional[np.random.Generator] = None,
) -> str:
    """Render a prompt for `frame_idx` in one of three modes:
      - `"raw"`:   original global task instruction
      - `"stage"`: global task + active stage instruction
      - `"full"`:  global task + active stage + local subgoal + 3D base-frame delta + next-10-step motion
    """
    global_inst = str(meta["instruction"]).strip()
    if mode == "raw":
        return global_inst

    T = int(meta["num_frames"])
    t = int(np.clip(frame_idx, 0, T - 1))
    stages = meta["stages"]
    active = stages[-1]
    for stg in stages:
        if stg["start_t"] <= t < stg["end_t"]:
            active = stg
            break

    k_idx = int(active["stage_idx"]) + 1
    n_stg = int(active["num_stages"])
    approach = str(active["approach"])
    stg_inst = str(active["stage_instruction"])

    if mode == "stage":
        return f"{stg_inst} (stage {k_idx}/{n_stg} {approach})"

    eef_now = np.asarray(meta["eef_world"][t], dtype=np.float64)
    R_base = np.asarray(meta["R_bases"][t], dtype=np.float64)
    grip_cmd = meta["grip_cmd"]

    if active["kind"] == "pick_place":
        if t <= active["grasp_t"]:
            tgt_name = str(active["grasp_target"])
            tgt_world = np.asarray(active["grasp_pos_world"], dtype=np.float64).copy()
            # If picking up a bowl after placing food into it, 50% of the time use the bowl center
            # (where the previous stage released the food) as the target reference so the policy
            # learns to shift from the bowl center onto the bowl rim to grasp.
            if (
                rng is not None
                and "bowl" in tgt_name
                and int(active["stage_idx"]) > 0
                and len(stages) > int(active["stage_idx"])
                and rng.random() < 0.50
            ):
                prev_stg = stages[int(active["stage_idx"]) - 1]
                prev_place = np.asarray(prev_stg["place_pos_world"], dtype=np.float64)
                tgt_world[0] = prev_place[0]
                tgt_world[1] = prev_place[1]
            if rng is not None and pos_noise_std_m > 0:
                tgt_world += rng.normal(0.0, pos_noise_std_m, size=3)
            delta_base = R_base.T @ (tgt_world - eef_now)
            dist = float(np.linalg.norm(delta_base))
            if dist <= 0.12:
                subgoal = f"descend to {tgt_name} and grasp"
            else:
                subgoal = f"reach {tgt_name}"
        else:
            tgt_name = str(active["place_target"] or "destination")
            obj_name = str(active["grasp_target"])
            tgt_world = np.asarray(active["place_pos_world"], dtype=np.float64).copy()
            if rng is not None and pos_noise_std_m > 0:
                tgt_world += rng.normal(0.0, pos_noise_std_m, size=3)
            delta_base = R_base.T @ (tgt_world - eef_now)
            dist = float(np.linalg.norm(delta_base))
            if dist <= 0.12:
                subgoal = f"place {obj_name} in {tgt_name} and release"
            else:
                subgoal = f"carry {obj_name} to {tgt_name}"
    else:
        tgt_name = str(active["grasp_target"])
        tgt_world = np.asarray(active["grasp_pos_world"], dtype=np.float64).copy()
        if rng is not None and pos_noise_std_m > 0:
            tgt_world += rng.normal(0.0, pos_noise_std_m, size=3)
        delta_base = R_base.T @ (tgt_world - eef_now)
        subgoal = stg_inst

    geo_str = format_base_delta_3d(delta_base)
    t_end10 = min(T - 1, t + 10)
    delta_10_base = R_base.T @ (
        np.asarray(meta["eef_world"][t_end10], dtype=np.float64) - eef_now
    )
    next10_str = format_next10_base_motion(
        delta_10_base,
        float(grip_cmd[t]),
        grip_cmd[t : t_end10 + 1],
    )
    return (
        f"{stg_inst} [s{k_idx}/{n_stg} {approach}; "
        f"{subgoal}; tgt {geo_str}; "
        f"next {next10_str}]"
    )


def format_runtime_stage_prompt(
    *,
    global_instruction: str,
    stage_idx: int,
    num_stages: int,
    kind: str,
    approach: str,
    stage_instruction: str,
    grasp_target: str,
    place_target: Optional[str],
    eef_pos_world: np.ndarray,
    R_base: np.ndarray,
    holding: bool,
    grasp_pos_world: Optional[np.ndarray],
    place_pos_world: Optional[np.ndarray],
) -> str:
    """Runtime prompt generator called inside `eval_robocasa.py` at every replan step."""
    stg_inst = (stage_instruction or global_instruction).strip().rstrip(".")
    k_idx = stage_idx + 1
    n_stg = max(1, num_stages)

    if kind == "pick_place":
        if not holding:
            tgt_name = grasp_target or "target object"
            tgt_world = grasp_pos_world
            if tgt_world is None:
                return f"{stg_inst} (stage {k_idx}/{n_stg} {approach})"
            delta_base = R_base.T @ (np.asarray(tgt_world, dtype=np.float64) - np.asarray(eef_pos_world, dtype=np.float64))
            dist = float(np.linalg.norm(delta_base))
            subgoal = (
                f"descend to {tgt_name} and grasp"
                if dist <= 0.13
                else f"reach {tgt_name}"
            )
            next_grip = "close gripper to grasp"
        else:
            tgt_name = place_target or "destination"
            obj_name = grasp_target or "object"
            tgt_world = place_pos_world
            if tgt_world is None:
                return f"{stg_inst} (stage {k_idx}/{n_stg} {approach})"
            delta_base = R_base.T @ (np.asarray(tgt_world, dtype=np.float64) - np.asarray(eef_pos_world, dtype=np.float64))
            dist = float(np.linalg.norm(delta_base))
            subgoal = (
                f"place {obj_name} in {tgt_name} and release"
                if dist <= 0.13
                else f"carry {obj_name} to {tgt_name}"
            )
            next_grip = "open gripper to release" if dist <= 0.08 else "keep gripper closed"
    else:
        tgt_name = grasp_target or "fixture"
        tgt_world = grasp_pos_world
        if tgt_world is None:
            return f"{stg_inst} (stage {k_idx}/{n_stg} {approach})"
        delta_base = R_base.T @ (np.asarray(tgt_world, dtype=np.float64) - np.asarray(eef_pos_world, dtype=np.float64))
        subgoal = stg_inst
        next_grip = "keep gripper open"

    geo_str = format_base_delta_3d(delta_base)
    step_vec = np.clip(delta_base, -0.08, 0.08)
    next10_str = format_next10_base_motion(
        step_vec,
        1.0 if holding else -1.0,
        np.array([1.0 if next_grip == "close gripper to grasp" or next_grip == "keep gripper closed" else -1.0]),
    )
    return (
        f"{stg_inst} [s{k_idx}/{n_stg} {approach}; "
        f"{subgoal}; tgt {geo_str}; "
        f"next {next10_str}]"
    )
