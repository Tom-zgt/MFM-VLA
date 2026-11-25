"""Offline evaluation of the post-grasp stage (analysis only; local π0.5, no GPT).

Replays a recorded episode up to its first confirmed grasp, then runs ``PlaceLoop`` of the
given stage with the live π0.5 policy to the horizon. Episodes that would need GPT again
(a drop followed by a fresh hand-off) end with ``needs_gpt``.

Usage: sim_place_loop.py ROWS.jsonl STAGE[:place_field=json...] PORT OUT.jsonl SUITE TASK [SUITE TASK ...]
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

import libero_compat  # noqa: F401
from eval_g5 import CONFIGS, CarryNeeded, HERE, Lifted, OutOfTime, PlaceLoop, Release
from eval_preposition_pick import _positions
from eval_vla_only import SUITE_HORIZONS
from generic_adapter import GraspMonitor, LiberoAdapter
from harness import TaskEnv
from vla_executor import OpenPiVLAExecutor, VLAConfig


def run(row: dict, cfg, port: int) -> dict:
    events = row.get("events") or []
    watch = next(e for e in events if e.get("why") == "vla_carry")
    profile = json.loads((HERE.parent / cfg.place.policy_profile).read_text())
    goal = np.asarray(watch["goal"], dtype=np.float64)
    crossing = profile.get("held_bottom_above_target_top_crossing_m") or 0.0
    hang = next((e["hang_m"] for e in events if e.get("why") == "carry"), 0.0)
    cruise = float(goal[2] - profile["held_bottom_above_target_top_m"] + crossing)
    pending = {"goal": goal, "hang": hang, "cruise": max(cruise, float(goal[2]))}

    horizon = SUITE_HORIZONS.get(row["suite"]) or SUITE_HORIZONS[row["suite"].rsplit("_", 1)[0]]
    steps = [s for s in map(json.loads, open(f"{row['artifacts']}/steps.jsonl")) if s.get("action")]
    env = TaskEnv(row["suite"], row["task"], img_size=128, seed=row["init"])
    env.allow_goal_shortcut = False
    env.reset(row["init"])

    def on_move(action, mode):
        if env.t >= horizon or env.success:
            raise OutOfTime

    adapter = LiberoAdapter(env, None, cfg.move, on_step=on_move)
    if cfg.verify.hold_min_opening_m is not None:
        adapter.hold_opening_m = (cfg.verify.hold_min_opening_m, adapter.hold_opening_m[1])
    home = adapter.proprio().eef_pos.copy()
    for s in steps:
        if env.t >= watch["t"]:
            break
        env.step(np.asarray(s["action"], dtype=np.float64))
    adapter.gripper_command = 1.0
    sim_events: list = []
    loop = PlaceLoop(cfg, env, adapter, home, horizon, sim_events)
    executor = OpenPiVLAExecutor(VLAConfig(port=port, resize_size=224, replan_steps=5,
                                           max_skill_steps=horizon, prompt_mode="task"))

    state = {"monitor": None}

    def on_vla(step_env, action, call):
        adapter.gripper_command = float(np.sign(action[6]) or adapter.gripper_command)
        if loop.step(action):
            return
        if state["monitor"] is not None and state["monitor"].update(action):
            raise Lifted

    end = "horizon"
    try:
        if adapter.holding():
            carry_now = loop.begin(pending)
        else:
            # Recorded lift is not a hold under this config: π0.5 goes on grasping under a
            # fresh lift detector (the GPT grasp check is assumed to pass).
            sim_events.append({"t": env.t, "why": "not_held"})
            state["monitor"] = GraspMonitor(adapter, settle_m=cfg.place.grasp_settle_m,
                                            settle_steps=cfg.place.grasp_settle_steps)
            carry_now = False
        while env.t < horizon and not env.success:
            if not carry_now:
                try:
                    executor.execute_task(env, max_steps=horizon - env.t, benchmark_stop_on_success=True,
                                          step_callback=on_vla)
                    break
                except Lifted:
                    # The place goal keeps the recorded grasp's hang (approximation).
                    state["monitor"] = None
                    sim_events.append({"t": env.t, "why": "lifted_again"})
                    carry_now = loop.begin(pending)
                    continue
                except CarryNeeded:
                    carry_now = True
                except Release:
                    loop.release()
                    continue
            carry_now = False
            carry = loop.carry()
            if not carry["still_holding"] and cfg.place.max_redo_after_drop > 0:
                end = "needs_gpt"
                break
            loop.hand_back()
    except OutOfTime:
        pass
    finally:
        executor.close()
    target = row.get("eval_target")
    final = {"holding": adapter.holding(),
             "eef_xy_m": round(float(np.linalg.norm(adapter.proprio().eef_pos[:2] - goal[:2])), 3)}
    if target:
        pos = _positions(env)
        name = next((n for n in pos if n.startswith(target) or target.startswith(n)), None)
        if name is not None:
            final["eval_only_target_xy_m"] = round(float(np.linalg.norm(pos[name][:2] - goal[:2])), 3)
            final["eval_only_target_dz_m"] = round(float(pos[name][2] - goal[2]), 3)
    if os.environ.get("SNAP_DIR"):
        from PIL import Image
        obs = env.observe(cameras=["agentview", "robot0_eye_in_hand"], size=384)
        Image.fromarray(np.concatenate([obs.rgb["agentview"], obs.rgb["robot0_eye_in_hand"]], axis=1)).save(
            f"{os.environ['SNAP_DIR']}/{row['suite']}_{row['task']}_{row['init']}_{cfg.stage}.png")
    out = {"suite": row["suite"], "task": row["task"], "init": row["init"], "stage": cfg.stage,
           "recorded": row["success"], "sim": bool(env.success), "end": end, "t": env.t, "final": final,
           "events": [{k: v for k, v in e.items() if k in ("t", "why", "outcome", "t_end", "still_holding", "moves", "move", "unfold",
                                                       "tilt_rad", "tilt_after_rad")}
                      for e in sim_events]}
    env.close()
    return out


def main() -> None:
    rows_path, stage, port, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
    args = sys.argv[5:]
    keys = {(args[i], int(args[i + 1])) for i in range(0, len(args), 2)}
    name, *overrides = stage.split(":")
    cfg = CONFIGS[name]
    for item in overrides:
        field, value = item.split("=")
        section, _, field = field.rpartition(".")
        section = section or "place"
        cfg = replace(cfg, stage=f"{cfg.stage}:{item}",
                      **{section: replace(getattr(cfg, section), **{field: json.loads(value)})})
    done = set()
    if out_path.exists():
        done = {(r["suite"], r["task"], r["init"]) for r in map(json.loads, out_path.open())}
    for row in map(json.loads, open(rows_path)):
        key = (row["suite"], row["task"], row["init"])
        if (row["suite"], row["task"]) not in keys or key in done or row.get("crash") or row.get("carried_over_from"):
            continue
        if not any(e.get("why") == "vla_carry" for e in row.get("events") or []):
            continue
        result = run(row, cfg, port)
        with out_path.open("a") as handle:
            handle.write(json.dumps(result) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
