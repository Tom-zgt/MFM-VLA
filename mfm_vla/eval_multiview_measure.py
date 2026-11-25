"""Perception-only check of second-view top / support measurement (no rollout).

Runs the G1 coarse and fine passes, then the wrist-view pass, at reset. The
simulator geometry is read only to score the estimates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np

import libero_compat  # noqa: F401

from mfm_client import MFMClientSession
from eval_preposition_pick import _target_object
from generic_adapter import LiberoAdapter, libero_calibration
from generic_config import AgentConfig
from generic_grounding import COARSE_PROMPT, FINE_PROMPT, ROLES, _draw, _resize, crop_window, ray_plane
from generic_multiview import measure_in_second_view
from harness import TaskEnv


def object_extent(env, name: str) -> tuple[np.ndarray, float, float]:
    """Evaluation only: object xy centre, lowest and highest z over its geoms' boxes."""
    sim = env._env.env.sim
    model = sim.model
    root = model.body_name2id(env._env.env.objects_dict[name].root_body)
    corners = []
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
    for g in range(model.ngeom):
        body = model.geom_bodyid[g]
        while body not in (0, root):
            body = model.body_parentid[body]
        if body != root:
            continue
        centre, half = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
        local = centre + signs * half
        rot = sim.data.geom_xmat[g].reshape(3, 3)
        corners.append(local @ rot.T + sim.data.geom_xpos[g])
    pts = np.concatenate(corners)
    return pts[:, :2].mean(0), float(pts[:, 2].min()), float(pts[:, 2].max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", required=True, help="suite:task:init")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/generic/multiview"))
    parser.add_argument("--output", type=Path, default=Path("reports/generic/multiview_measure.jsonl"))
    args = parser.parse_args()
    cfg = AgentConfig().grounding
    for case in args.cases:
        suite, task, init = case.split(":")
        out = args.out_dir / f"{suite}_t{task}_i{init}"
        out.mkdir(parents=True, exist_ok=True)
        row = {"suite": suite, "task": int(task), "init": int(init)}
        env = TaskEnv(suite, int(task), img_size=128, seed=int(init))
        session = MFMClientSession(AgentConfig().models.perception, max_tokens=1000)
        started = time.monotonic()
        try:
            env.reset(int(init))
            try:
                calibration = libero_calibration(suite)
            except KeyError:
                calibration = None
            adapter = LiberoAdapter(env, calibration, AgentConfig().move)
            obs = adapter.observe(["agentview"], cfg.render_px)
            frame = obs.rgb["agentview"]
            coarse, cs = _resize(frame, cfg.coarse_px)
            r1 = session.decide_json(
                COARSE_PROMPT.format(width=coarse.shape[1], height=coarse.shape[0],
                                     instruction=env.instruction, k=cfg.max_candidates,
                                     question=ROLES["grasp"]["question"]),
                {"agentview": coarse},
            )
            cands = [c for c in r1["candidates"] if len(c.get("bbox") or []) == 4][: cfg.max_candidates]
            boxes = [np.asarray(c["bbox"], dtype=np.float64) / cs for c in cands]
            left, top, right, bottom = crop_window(boxes, frame.shape[0], cfg)
            crop, fs = _resize(frame[top:bottom, left:right], cfg.fine_px)
            crop_boxes = [(b - np.array([left, top, left, top])) * fs for b in boxes]
            listed = ", ".join(f"{i}: {c.get('name', '?')}" for i, c in enumerate(cands, 1))
            r2 = session.decide_json(
                FINE_PROMPT.format(width=crop.shape[1], height=crop.shape[0], candidates=listed,
                                   instruction=env.instruction, target=ROLES["grasp"]["target"]),
                {"agentview": np.asarray(_draw(crop, crop_boxes, range(1, len(boxes) + 1)))},
            )

            def to_frame(uv):
                return uv / fs + np.array([left, top], dtype=np.float64)

            mv = measure_in_second_view(adapter, session, r2, crop, to_frame, obs, cfg.render_px, out)
            target = _target_object(env)
            xy, zmin, zmax = object_extent(env, target)
            if calibration is not None:
                plane_point = ray_plane(obs, "agentview", to_frame(np.asarray(r2["pixel_base_center"], float)),
                                        calibration.support_eef_z)
                row["g1_xy_error_m"] = round(float(np.linalg.norm(plane_point[:2] - xy)), 4)
            row.update({
                "target": target, "name": r2.get("name"), "visible": mv["visible"],
                "true_xy": np.round(xy, 4).tolist(), "true_bottom_z": round(zmin, 4), "true_top_z": round(zmax, 4),
                "latency_s": round(time.monotonic() - started, 1), "usage": session.usage(),
                "second_view": {k: v for k, v in mv.items() if k != "top"},
            })
            if mv["visible"]:
                row.update({
                    "mv_xy_error_m": round(float(np.linalg.norm(mv["top"][:2] - xy)), 4),
                    "mv_top_z": round(float(mv["top"][2]), 4),
                    "mv_top_error_m": round(float(mv["top"][2] - zmax), 4),
                    "mv_support_error_m": round(float(mv["support_z"] - zmin), 4),
                    "mv_height_error_m": round(float(mv["height_m"] - (zmax - zmin)), 4),
                })
        except Exception as exc:  # noqa: BLE001
            row.update({"crash": True, "reason": f"{type(exc).__name__}: {exc}",
                        "detail": traceback.format_exc(limit=6)})
        finally:
            session.close()
            env.close()
        with args.output.open("a") as handle:
            handle.write(json.dumps(row, default=str) + "\n")
        print(json.dumps(row, default=str), flush=True)


if __name__ == "__main__":
    main()
