"""Oracle-handoff terminal-skill ablation for RoboCasa365.

This auxiliary benchmark isolates contact-rich terminal actions from upstream
VLA perception, grasping, and transport. Simulator state is used only to build
and verify the prerequisite handoff state. Every scored terminal action is then
executed through the same physical 12D PandaOmron controller as the main eval.

Results from this file are an upper-bound diagnostic, not a replacement for
end-to-end VLA results.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
from pathlib import Path
import time
from typing import Optional

import numpy as np
import robocasa.utils.object_utils as OU
import robosuite.utils.transform_utils as T

from robocasa_skill_primitives import (
    _make_fast_adapter,
    close_microwave_and_start,
    fixture_region_center_world,
    grasp_and_place_smart,
    pour_held_pan_and_return,
    turn_sink_water_on_and_align,
)


SUPPORTED_TASKS = (
    "SteamInMicrowave",
    "WashFruitColander",
    "PanTransfer",
)


def _set_free_object_xyz(inner, obj_key: str, xyz: np.ndarray) -> None:
    sim = inner.sim
    joint_name = inner.objects[obj_key].joints[0]
    qpos = np.asarray(sim.data.get_joint_qpos(joint_name), dtype=np.float64).copy()
    qpos[:3] = np.asarray(xyz, dtype=np.float64)
    sim.data.set_joint_qpos(joint_name, qpos)
    qvel = np.asarray(
        sim.data.get_joint_qvel(joint_name),
        dtype=np.float64,
    )
    sim.data.set_joint_qvel(joint_name, np.zeros_like(qvel))
    sim.forward()


def _settle(adapter, steps: int = 60) -> None:
    for _ in range(steps):
        act = np.zeros(12, dtype=np.float64)
        act[6] = -1.0
        act[10] = -1.0
        act[11] = -1.0
        adapter.step_12d(act, mode="oracle_handoff_settle")


def _steam_predicates(inner) -> dict:
    return {
        "vegetable_in_bowl": bool(
            OU.check_obj_in_receptacle(inner, "vegetable", "bowl")
        ),
        "bowl_in_microwave": bool(
            OU.obj_inside_of(inner, "bowl", inner.microwave)
        ),
        "door_closed": bool(inner.microwave.is_closed(inner)),
        "microwave_on": bool(inner.microwave.get_state()["turned_on"]),
    }


def _prepare_steam_handoff(adapter) -> dict:
    inner = adapter.inner
    sim = adapter._sim
    p_tray = fixture_region_center_world(
        inner.microwave,
        inner,
        z_offset=0.025,
    )
    _set_free_object_xyz(inner, "bowl", p_tray)
    _settle(adapter, 45)
    p_bowl = sim.data.body_xpos[inner.obj_body_id["bowl"]].copy()
    _set_free_object_xyz(
        inner,
        "vegetable",
        p_bowl + np.array([0.0, 0.0, 0.11]),
    )
    _settle(adapter, 90)
    return _steam_predicates(inner)


def _run_steam_terminal(adapter) -> tuple[dict, dict]:
    inner = adapter.inner
    details = close_microwave_and_start(adapter, inner.microwave)
    return _steam_predicates(inner), details


def _sink_basin_targets(inner) -> list[np.ndarray]:
    """Return collision-safe colander centers for every available basin."""
    R_sink = T.euler2mat(np.array([0.0, 0.0, inner.sink.rot]))
    sink_pos = np.asarray(inner.sink.pos, dtype=np.float64)
    bottom_z = float(inner.objects["colander"].bottom_offset[2])
    targets = []
    for region in inner.sink.get_reset_regions(inner).values():
        local = np.asarray(region["offset"], dtype=np.float64)
        floor = sink_pos + R_sink @ local
        floor[2] = sink_pos[2] + local[2]
        # Leave enough free-fall clearance to avoid initial interpenetration
        # with thin basin meshes, while staying below the faucet/spout.
        floor[2] += -bottom_z + 0.09
        targets.append(floor)
    if not targets:
        raise ValueError("Sink exposes no basin reset region")
    return targets


def _wash_predicates(inner) -> dict:
    sim = inner.sim
    water_sid = sim.model.site_name2id(
        f"{inner.sink.naming_prefix}water"
    )
    colander_xy = sim.data.body_xpos[
        inner.obj_body_id["colander"]
    ][:2]
    water_xy = sim.data.site_xpos[water_sid][:2]
    fruits_in = [
        bool(
            OU.check_obj_in_receptacle(
                inner,
                f"fruit{idx}",
                "colander",
            )
        )
        for idx in range(inner.num_fruit)
    ]
    handle_state = inner.sink.get_handle_state(inner)
    return {
        "fruit_in_colander": fruits_in,
        "all_fruits_in_colander": bool(all(fruits_in)),
        "water_on": bool(handle_state.get("water_on", False)),
        "colander_under_water": bool(
            inner.sink.check_obj_under_water(inner, "colander")
        ),
        "handle_q": round(float(handle_state.get("handle_joint", 0.0)), 4),
        "spout_q": round(float(handle_state.get("spout_joint", 0.0)), 4),
        "water_distance_m": round(
            float(np.linalg.norm(colander_xy - water_xy)),
            4,
        ),
        "colander_xyz": np.round(
            sim.data.body_xpos[inner.obj_body_id["colander"]],
            4,
        ).tolist(),
    }


def _prepare_wash_handoff(adapter, episode: int) -> dict:
    inner = adapter.inner
    sim = adapter._sim
    prefix = inner.sink.naming_prefix
    spout_joint = f"{prefix}spout_joint"
    spout_jid = sim.model.joint_name2id(spout_joint)
    water_sid = sim.model.site_name2id(f"{prefix}water")

    q_lo, q_hi = map(float, sim.model.jnt_range[spout_jid])

    def _best_spout_q(target_xy: np.ndarray) -> tuple[float, float]:
        q0 = float(sim.data.get_joint_qpos(spout_joint))
        best_q = q0
        best_dist = float("inf")
        for q in np.linspace(q_lo + 0.02, q_hi - 0.02, 81):
            sim.data.set_joint_qpos(spout_joint, float(q))
            sim.forward()
            dist = float(
                np.linalg.norm(
                    sim.data.site_xpos[water_sid][:2] - target_xy
                )
            )
            if dist < best_dist:
                best_q = float(q)
                best_dist = dist
        sim.data.set_joint_qpos(spout_joint, q0)
        sim.forward()
        return best_q, best_dist

    # Select a physically stable basin center that lies on the spout's arc.
    candidates = _sink_basin_targets(inner)
    ranked = [
        (*_best_spout_q(candidate[:2]), candidate)
        for candidate in candidates
    ]
    desired_q, _, p_colander = min(ranked, key=lambda item: item[1])
    _set_free_object_xyz(inner, "colander", p_colander)
    _settle(adapter, 45)
    p_colander = sim.data.body_xpos[inner.obj_body_id["colander"]].copy()
    desired_q, _ = _best_spout_q(p_colander[:2])

    for idx in range(inner.num_fruit):
        theta = 2.0 * np.pi * idx / max(inner.num_fruit, 1)
        offset = 0.012 * np.array([np.cos(theta), np.sin(theta), 0.0])
        _set_free_object_xyz(
            inner,
            f"fruit{idx}",
            p_colander + offset + np.array([0.0, 0.0, 0.12]),
        )
        _settle(adapter, 50)
    # Small randomized fruit meshes can roll out while later fruit settle.
    # Re-seat only failed members until the oracle prerequisite is physically
    # stable; no scored terminal action occurs in this setup loop.
    for _ in range(3):
        failed = [
            idx
            for idx in range(inner.num_fruit)
            if not OU.check_obj_in_receptacle(
                inner,
                f"fruit{idx}",
                "colander",
            )
        ]
        if not failed:
            break
        p_colander = sim.data.body_xpos[inner.obj_body_id["colander"]].copy()
        for idx in failed:
            _set_free_object_xyz(
                inner,
                f"fruit{idx}",
                p_colander + np.array([0.0, 0.0, 0.10]),
            )
            _settle(adapter, 35)

    for _ in range(5):
        failed = [
            idx
            for idx in range(inner.num_fruit)
            if not OU.check_obj_in_receptacle(
                inner,
                f"fruit{idx}",
                "colander",
            )
        ]
        if not failed:
            break
        p_colander = sim.data.body_xpos[inner.obj_body_id["colander"]].copy()
        for idx in failed:
            _set_free_object_xyz(
                inner,
                f"fruit{idx}",
                p_colander + np.array([0.0, 0.0, 0.10]),
            )
            _settle(adapter, 20)

    # Start from the opposite side of the selected basin so stream alignment is
    # a required scored action in both even and odd episodes.
    midpoint = 0.5 * (q_lo + q_hi)
    if (desired_q >= midpoint) ^ bool(episode % 2):
        initial_q = q_lo + 0.08 * (q_hi - q_lo)
    else:
        initial_q = q_hi - 0.08 * (q_hi - q_lo)
    sim.data.set_joint_qpos(spout_joint, initial_q)
    sim.forward()
    _settle(adapter, 5)
    return _wash_predicates(inner)


def _run_wash_terminal(adapter) -> tuple[dict, dict]:
    inner = adapter.inner
    details = turn_sink_water_on_and_align(
        adapter,
        inner.sink,
        target_obj="colander",
    )
    return _wash_predicates(inner), details


def _pan_predicates(inner) -> dict:
    return {
        "vegetable_on_plate": bool(
            OU.check_obj_in_receptacle(inner, "vegetable", "plate")
        ),
        "pan_on_stove": bool(
            inner._check_obj_location_on_stove("vegetable_container")
            is not None
        ),
        "gripper_pan_far": bool(
            OU.gripper_obj_far(
                inner,
                obj_name="vegetable_container",
            )
        ),
        "robot_touched_food": bool(inner._robot_touched_food),
    }


def _run_pan_terminal(adapter) -> tuple[dict, dict]:
    inner = adapter.inner
    sim = adapter._sim
    p_home = sim.data.body_xpos[
        inner.obj_body_id["vegetable_container"]
    ].copy()
    p_plate = sim.data.body_xpos[inner.obj_body_id["plate"]].copy()
    held = grasp_and_place_smart(
        adapter,
        "vegetable_container",
        p_plate + np.array([0.0, 0.0, 0.19]),
        release_z=0.0,
        force_top_grasp=True,
        carry_speed_limit=0.18,
        arm_speed_limit=0.40,
        lift_clearance_m=0.10,
        src_fixture=inner.stove,
        src_ref_obj="vegetable_container",
        release_after_place=False,
    )
    details = {"pan_grasped": bool(held)}
    if held:
        details["tilt_pour"] = pour_held_pan_and_return(
            adapter,
            pan_key="vegetable_container",
            food_key="vegetable",
            plate_key="plate",
            home_xyz=p_home,
        )
    return _pan_predicates(inner), details


def run_terminal_skill_episode(
    task: str,
    episode: int,
    seed: int = 7,
) -> dict:
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported terminal skill task: {task}")
    started = time.monotonic()
    env = None
    try:
        adapter, env = _make_fast_adapter(task, seed + 100 * episode)
        details: Optional[dict] = None
        if task == "SteamInMicrowave":
            before = _prepare_steam_handoff(adapter)
            prerequisite_success = bool(
                before["vegetable_in_bowl"]
                and before["bowl_in_microwave"]
            )
            terminal_start = adapter.t
            after, details = _run_steam_terminal(adapter)
            terminal_success = bool(
                after["vegetable_in_bowl"]
                and after["bowl_in_microwave"]
                and after["door_closed"]
                and after["microwave_on"]
            )
            oracle_handoff = True
        elif task == "WashFruitColander":
            before = _prepare_wash_handoff(adapter, episode)
            prerequisite_success = bool(
                before["all_fruits_in_colander"]
            )
            terminal_start = adapter.t
            after, details = _run_wash_terminal(adapter)
            terminal_success = bool(
                after["all_fruits_in_colander"]
                and after["water_on"]
                and after["colander_under_water"]
            )
            oracle_handoff = True
        else:
            before = _pan_predicates(adapter.inner)
            prerequisite_success = bool(
                before["pan_on_stove"]
                and not before["vegetable_on_plate"]
            )
            terminal_start = adapter.t
            after, details = _run_pan_terminal(adapter)
            terminal_success = bool(
                after["vegetable_on_plate"]
                and after["pan_on_stove"]
                and after["gripper_pan_far"]
                and not after["robot_touched_food"]
            )
            oracle_handoff = False
        return {
            "task": task,
            "episode": int(episode),
            "seed": int(seed),
            "episode_seed": int(seed + 100 * episode),
            "layout_id": int(inner.layout_id),
            "style_id": int(inner.style_id),
            "mode": "terminal_code_policy_ablation",
            "auxiliary_only": True,
            "comparable_to_main_rows": False,
            "oracle_handoff": oracle_handoff,
            "handoff_type": (
                "oracle_prerequisite"
                if oracle_handoff
                else "native_initial_state"
            ),
            "privileged_geometry": True,
            "scored_action_interface": "PandaOmron_12D",
            "prerequisite_success": prerequisite_success,
            "terminal_skill_success": terminal_success,
            "final_task_success": terminal_success,
            "ever_env_success": bool(adapter.success),
            "task_success_after_terminal": terminal_success,
            "end_to_end_success": terminal_success,
            "predicates_before": before,
            "predicates_after": after,
            "skill_details": details,
            "terminal_steps": int(adapter.t - terminal_start),
            "total_steps": int(adapter.t),
            "crash": False,
            "duration_s": round(time.monotonic() - started, 2),
        }
    except Exception as exc:
        return {
            "task": task,
            "episode": int(episode),
            "seed": int(seed),
            "episode_seed": int(seed + 100 * episode),
            "mode": "terminal_code_policy_ablation",
            "auxiliary_only": True,
            "comparable_to_main_rows": False,
            "oracle_handoff": task != "PanTransfer",
            "handoff_type": (
                "oracle_prerequisite"
                if task != "PanTransfer"
                else "native_initial_state"
            ),
            "privileged_geometry": True,
            "scored_action_interface": "PandaOmron_12D",
            "prerequisite_success": False,
            "terminal_skill_success": False,
            "final_task_success": False,
            "ever_env_success": False,
            "task_success_after_terminal": False,
            "end_to_end_success": False,
            "crash": True,
            "reason": f"{type(exc).__name__}: {exc}",
            "duration_s": round(time.monotonic() - started, 2),
        }
    finally:
        if env is not None:
            env.close()


def _worker(args: tuple[str, int, int]) -> dict:
    return run_terminal_skill_episode(*args)


def summarize_rows(rows: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for task in SUPPORTED_TASKS:
        task_rows = [row for row in rows if row["task"] == task]
        if not task_rows:
            continue
        valid = [
            row for row in task_rows if row.get("prerequisite_success", False)
        ]
        terminal_wins = sum(
            bool(row.get("terminal_skill_success", False))
            for row in valid
        )
        task_wins = sum(
            bool(row.get("final_task_success", False))
            for row in task_rows
        )
        ever_wins = sum(
            bool(row.get("ever_env_success", False))
            for row in task_rows
        )
        summary[task] = {
            "episodes": len(task_rows),
            "valid_prerequisites": len(valid),
            "prerequisite_rate": round(len(valid) / len(task_rows), 4),
            "terminal_successes_given_valid": terminal_wins,
            "terminal_success_rate_given_valid": (
                round(terminal_wins / len(valid), 4) if valid else None
            ),
            "final_task_successes": task_wins,
            "final_task_success_rate": round(
                task_wins / len(task_rows),
                4,
            ),
            "ever_env_successes": ever_wins,
            "ever_env_success_rate": round(
                ever_wins / len(task_rows),
                4,
            ),
            "crashes": sum(bool(row.get("crash", False)) for row in task_rows),
        }
    return {
        "mode": "terminal_code_policy_ablation",
        "auxiliary_only": True,
        "comparable_to_main_rows": False,
        "tasks": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=SUPPORTED_TASKS,
        default=list(SUPPORTED_TASKS),
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/robocasa/results_terminal_skill_oracle_v1.jsonl"
        ),
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    jobs = [
        (task, episode, args.seed)
        for task in args.tasks
        for episode in range(args.episodes)
    ]
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=ctx,
    ) as pool:
        for row in pool.map(_worker, jobs):
            rows.append(row)
            print(
                f"{row['task']:24s} ep={row['episode']} "
                f"pre={int(row['prerequisite_success'])} "
                f"terminal={int(row['terminal_skill_success'])} "
                f"e2e={int(row['end_to_end_success'])}",
                flush=True,
            )
    args.output.write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    summary = summarize_rows(rows)
    summary_path = args.output.with_name(
        f"{args.output.stem}.summary.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
