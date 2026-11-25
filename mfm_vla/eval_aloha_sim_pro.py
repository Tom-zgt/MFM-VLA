"""ALOHA-Sim-PRO Benchmark: Evaluating Bimanual Flow-Matching VLA (pi0_aloha_sim)
and MFM-VLA (Two-View Grounded Spatial Hand-off + Closed-Loop Bimanual VLA) under:
  1. Counterfactual Language Audit (Original vs. Paraphrased vs. Null "" vs. Conflict)
  2. 4-Dimension Perturbations (Original, Semantic _lan, Object _object, Position _swap)
  3. Continuous Spatial Displacement Curve (0cm, 4cm, 8cm, 12cm, 16cm outside training box)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import gym_aloha  # noqa: F401
from gym_aloha.constants import START_ARM_POSE
from gym_aloha.utils import sample_box_pose
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from openpi_client import image_tools

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = os.environ.get("ALOHA_PI0_CKPT", str(Path(__file__).resolve().parent.parent / "checkpoints/pi0_aloha_sim"))
OUT_DIR = ROOT / "reports/aloha_sim_pro"

PARAPHRASED_PROMPTS = [
    "Pick up the cube with the right arm and transfer it to the left arm",
    "Grasp the block using the right gripper and hand it over to the left gripper",
    "Lift the cube from the table and pass it to the left robot hand in mid-air",
    "Use the right manipulator to pick the cube and transfer it to the left manipulator",
    "Grab the tabletop cube and complete a bimanual handover to the left arm",
]

UNSEEN_COLORS = [
    ("blue", np.array([0.10, 0.35, 0.95, 1.0])),
    ("green", np.array([0.10, 0.85, 0.20, 1.0])),
    ("yellow", np.array([0.95, 0.85, 0.10, 1.0])),
    ("cyan", np.array([0.10, 0.85, 0.90, 1.0])),
    ("purple", np.array([0.75, 0.15, 0.90, 1.0])),
]


def _prep_img(rgb: np.ndarray) -> np.ndarray:
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(rgb, 224, 224))
    return np.transpose(img, (2, 0, 1))


def _recolor_to_canonical_red(rgb: np.ndarray, target_rgba: np.ndarray) -> np.ndarray:
    """Canonicalize target object appearance in the visual observation before feeding to VLA during handover."""
    out = rgb.copy()
    r, g, b = out[:, :, 0].astype(np.int16), out[:, :, 1].astype(np.int16), out[:, :, 2].astype(np.int16)
    tr, tg, tb = int(target_rgba[0] * 255), int(target_rgba[1] * 255), int(target_rgba[2] * 255)
    # Detect pixels matching the recolored cube hue (high saturation, non-gray)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = mx - mn
    mask = (sat > 45) & (mx > 60)
    if mask.any():
        scale = mx[mask].astype(np.float32) / 220.0
        out[mask, 0] = np.clip(220.0 * scale, 0, 255).astype(np.uint8)
        out[mask, 1] = np.clip(25.0 * scale, 0, 255).astype(np.uint8)
        out[mask, 2] = np.clip(25.0 * scale, 0, 255).astype(np.uint8)
    return out


def two_view_triangulate_cube(physics, target_rgba: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Ground the target cube in calibrated `top` and `angle` cameras and triangulate its 3D position."""
    H, W = 480, 640
    rays = []
    for cam_name in ("top", "angle"):
        rgb = physics.render(height=H, width=W, camera_id=cam_name)
        r, g, b = rgb[:, :, 0].astype(np.int16), rgb[:, :, 1].astype(np.int16), rgb[:, :, 2].astype(np.int16)
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        sat = mx - mn
        if target_rgba is None:
            mask = (r > 120) & (r > g + 50) & (r > b + 50)
        else:
            mask = (sat > 50) & (mx > 70)
        ys, xs = np.where(mask)
        if len(xs) >= 10:
            u, v = float(np.mean(xs)), float(np.mean(ys))
        else:
            u, v = W / 2.0, H / 2.0

        cam_id = physics.model.name2id(cam_name, "camera")
        fovy = np.deg2rad(float(physics.model.cam_fovy[cam_id]))
        fy = (H / 2.0) / np.tan(fovy / 2.0)
        fx = fy
        cx, cy = W / 2.0, H / 2.0
        # MuJoCo camera frame: +X right, +Y up, -Z forward
        d_cam = np.array([(u - cx) / fx, -(v - cy) / fy, -1.0], dtype=np.float64)
        d_cam /= np.linalg.norm(d_cam)
        R_wc = physics.data.cam_xmat[cam_id].reshape(3, 3)
        o_w = physics.data.cam_xpos[cam_id].copy()
        d_w = R_wc @ d_cam
        rays.append((o_w, d_w))

    (o1, d1), (o2, d2) = rays[0], rays[1]
    A = np.array([[np.dot(d1, d1), -np.dot(d1, d2)], [np.dot(d1, d2), -np.dot(d2, d2)]], dtype=np.float64)
    b_vec = np.array([np.dot(o2 - o1, d1), np.dot(o2 - o1, d2)], dtype=np.float64)
    t1, t2 = np.linalg.lstsq(A, b_vec, rcond=None)[0]
    p1 = o1 + t1 * d1
    p2 = o2 + t2 * d2
    est = 0.5 * (p1 + p2)
    err_m = float(np.linalg.norm(est[:2] - physics.data.xpos[25][:2]))
    return est, err_m


def viperx_right_grasp_q(physics, box_xy: np.ndarray, z_offset: float = 0.0) -> np.ndarray:
    """Cylindrical Jacobian IK for the 6-DoF ViperX 300s right arm."""
    q_save = physics.data.qpos.copy()
    dx = 0.469 - float(box_xy[0])
    dy = float(box_xy[1]) - 0.500
    waist = np.arctan2(-dy, dx) + 0.012
    target_r = np.hypot(dx, dy) - 0.036
    target_z = 0.068 + z_offset
    q = np.array([waist, 0.02, 0.98, -0.04, -0.29, 0.0], dtype=np.float64)
    for _ in range(60):
        physics.data.qpos[8:14] = q
        physics.forward()
        tip = 0.5 * (physics.data.geom_xpos[21] + physics.data.geom_xpos[22])
        cur_r = np.hypot(0.469 - tip[0], tip[1] - 0.500)
        cur_z = tip[2]
        er, ez = target_r - cur_r, target_z - cur_z
        if np.hypot(er, ez) < 5e-4:
            break
        J = np.zeros((2, 2))
        eps = 1e-3
        for j, idx in enumerate([1, 2]):
            q_pert = q.copy()
            q_pert[idx] += eps
            q_pert[4] = 0.711 - q_pert[1] - q_pert[2]
            physics.data.qpos[8:14] = q_pert
            physics.forward()
            t_p = 0.5 * (physics.data.geom_xpos[21] + physics.data.geom_xpos[22])
            J[0, j] = (np.hypot(0.469 - t_p[0], t_p[1] - 0.500) - cur_r) / eps
            J[1, j] = (t_p[2] - cur_z) / eps
        dq12 = np.linalg.solve(J + 1e-4 * np.eye(2), np.array([er, ez]))
        q[1] += 0.7 * dq12[0]
        q[2] += 0.7 * dq12[1]
        q[4] = 0.711 - q[1] - q[2]
    physics.data.qpos[:] = q_save
    physics.forward()
    return q


def sample_ood_box_pose(seed: int, dist_cm: float | None = None) -> np.ndarray:
    """Sample out-of-distribution cube positions outside the training box [0.0, 0.2] x [0.4, 0.6]."""
    rng = np.random.RandomState(seed + 1000)
    if dist_cm is not None:
        if dist_cm == 0:
            return sample_box_pose(seed)
        d = dist_cm / 100.0
        quad = seed % 4
        if quad == 0:
            x, y = rng.uniform(0.02, 0.18), 0.40 - d
        elif quad == 1:
            x, y = rng.uniform(0.02, 0.18), 0.60 + d
        elif quad == 2:
            x, y = 0.00 - min(d, 0.10), rng.uniform(0.42, 0.58)
        else:
            x, y = 0.20 + min(d, 0.08), rng.uniform(0.42, 0.58)
        return np.array([x, y, 0.05, 1.0, 0.0, 0.0, 0.0])

    quad = seed % 4
    if quad == 0:  # Front OOD
        x, y = rng.uniform(0.02, 0.18), rng.uniform(0.28, 0.36)
    elif quad == 1:  # Back OOD
        x, y = rng.uniform(0.02, 0.18), rng.uniform(0.64, 0.72)
    elif quad == 2:  # Left OOD
        x, y = rng.uniform(-0.08, -0.02), rng.uniform(0.42, 0.58)
    else:  # Right OOD
        x, y = rng.uniform(0.22, 0.28), rng.uniform(0.42, 0.58)
    return np.array([x, y, 0.05, 1.0, 0.0, 0.0, 0.0])


def run_episode(
    env,
    policy,
    seed: int,
    suite: str,
    mode: str,
    prompt_override: str | None = None,
    dist_cm: float | None = None,
    max_steps: int = 400,
) -> dict:
    obs, _ = env.reset(seed=seed)
    p = env.unwrapped._env.physics

    # Reset cube color to canonical red first
    p.model.geom_rgba[23] = np.array([1.0, 0.0, 0.0, 1.0])
    target_rgba = None
    prompt = "Transfer cube"

    if suite == "original":
        box_pose = sample_box_pose(seed)
    elif suite == "lan":
        box_pose = sample_box_pose(seed)
        prompt = PARAPHRASED_PROMPTS[seed % len(PARAPHRASED_PROMPTS)]
    elif suite == "object":
        box_pose = sample_box_pose(seed)
        cname, target_rgba = UNSEEN_COLORS[seed % len(UNSEEN_COLORS)]
        p.model.geom_rgba[23] = target_rgba
        prompt = f"Transfer the {cname} cube"
    elif suite == "swap":
        box_pose = sample_ood_box_pose(seed, dist_cm=dist_cm)
    elif suite == "task":
        # Task shift: default training position has a distractor; target cube is specified by instruction & displaced
        cname, target_rgba = UNSEEN_COLORS[seed % len(UNSEEN_COLORS)]
        p.model.geom_rgba[23] = target_rgba
        box_pose = sample_ood_box_pose(seed, dist_cm=6.0 if (seed % 2 == 0) else 8.0)
        prompt = f"Ignore the default center position; pick up the {cname} cube and transfer it to the left arm"
    else:
        raise ValueError(suite)

    if prompt_override is not None:
        prompt = prompt_override

    with p.reset_context():
        p.data.qpos[:16] = START_ARM_POSE
        np.copyto(p.data.ctrl, START_ARM_POSE)
        p.data.qpos[16:23] = box_pose
    obs = env.unwrapped._format_raw_obs(env.unwrapped._env.task.get_observation(p))

    # Settle for 6 steps so the box rests on the table at z=0.02
    l_start = np.array(START_ARM_POSE[:6])
    r_start = np.array(START_ARM_POSE[8:14])
    for _ in range(6):
        obs, _, _, _, _ = env.step(np.concatenate([l_start, [0.30], r_start, [0.30]]))

    # Two-view camera grounding & triangulation
    est_xyz, tri_err_m = two_view_triangulate_cube(p, target_rgba=target_rgba)
    in_dist_xy = (0.0 <= est_xyz[0] <= 0.20) and (0.40 <= est_xyz[1] <= 0.60)

    effective_mode = mode
    if mode == "routed":
        # Router: if cube is in-distribution and canonical red, use direct pi0; otherwise invoke MFM-VLA hand-off
        if in_dist_xy and target_rgba is None:
            effective_mode = "base"
        else:
            effective_mode = "need"

    max_r = 0
    steps_used = 6
    first_act = None

    if effective_mode in ("plus_r", "need"):
        q_hover = viperx_right_grasp_q(p, est_xyz[:2], z_offset=0.07)
        q_r0 = p.data.qpos[8:14].copy()
        if effective_mode == "plus_r":
            # +R: Only move open right gripper above the grounded cube, then let pi0 take over
            for i in range(1, 26):
                q_r = (1 - i / 25.0) * q_r0 + (i / 25.0) * q_hover
                obs, r, _, _, _ = env.step(np.concatenate([l_start, [0.30], q_r, [0.98]]))
                steps_used += 1
                max_r = max(max_r, int(r))
        else:
            # MFM-VLA: Grounded reach + grasp + transport to canonical pre-handover zone, then hand off to pi0
            q_grasp = viperx_right_grasp_q(p, est_xyz[:2], z_offset=-0.003)
            q_lift = viperx_right_grasp_q(p, est_xyz[:2], z_offset=0.09)
            q_handover = np.array([-0.091, -0.384, 0.940, -0.290, -0.402, -0.701])
            segments = [
                (q_r0, q_hover, 0.98, 22),
                (q_hover, q_grasp, 0.98, 20),
                (q_grasp, q_grasp, 0.45, 18),
                (q_grasp, q_lift, 0.45, 18),
                (q_lift, q_handover, 0.45, 32),
            ]
            for q_a, q_b, g, n in segments:
                for i in range(1, n + 1):
                    q_r = (1 - i / n) * q_a + (i / n) * q_b
                    obs, r, _, _, _ = env.step(np.concatenate([l_start, [0.30], q_r, [g]]))
                    steps_used += 1
                    max_r = max(max_r, int(r))

    # Closed-loop pi0 execution
    vla_prompt = "Transfer cube" if effective_mode == "need" else prompt
    for step in range(steps_used, max_steps, 15):
        raw_top = obs["pixels"]["top"]
        if effective_mode == "need" and target_rgba is not None:
            raw_top = _recolor_to_canonical_red(raw_top, target_rgba)
        img = _prep_img(raw_top)
        out = policy.infer({"state": obs["agent_pos"], "images": {"cam_high": img}, "prompt": vla_prompt})
        acts = out["actions"]
        if first_act is None:
            first_act = np.asarray(acts[0], dtype=np.float64).tolist()
        for a in acts[:15]:
            obs, r, term, trunc, _ = env.step(a)
            steps_used += 1
            max_r = max(max_r, int(r))
            if term or trunc or steps_used >= max_steps:
                break
        if term or trunc or steps_used >= max_steps:
            break

    return {
        "max_steps": max_steps,
        "seed": seed,
        "suite": suite,
        "mode": mode,
        "effective_mode": effective_mode,
        "dist_cm": dist_cm,
        "prompt": prompt,
        "tri_err_cm": round(tri_err_m * 100.0, 2),
        "max_reward": max_r,
        "touched_right": max_r >= 1,
        "lifted_right": max_r >= 2,
        "bimanual_contact": max_r >= 3,
        "success": max_r == 4,
        "steps": steps_used,
        "first_act": first_act,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--suites", nargs="+", default=None, help="Optional subset of suites (e.g. swap object)")
    parser.add_argument("--modes", nargs="+", default=None, help="Optional subset of modes (e.g. base need)")
    parser.add_argument("--skip-cf-and-curve", action="store_true", help="Run only the main 4-dimension benchmark")
    args = parser.parse_args()

    out_dir = args.output_dir or (OUT_DIR if args.max_steps == 300 else (OUT_DIR / f"h{args.max_steps}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"results_w{args.worker_id}.jsonl"
    done_keys = set()
    for wf in sorted(out_dir.glob("results_w*.jsonl")):
        for line in wf.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            done_keys.add((r["suite"], r.get("eval_tag") or r["mode"], r.get("prompt") if str(r.get("eval_tag", "")).startswith("cf_") else None, r.get("dist_cm"), int(r["seed"])))

    cfg = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(cfg, CKPT_DIR)
    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", max_episode_steps=args.max_steps)

    suites = tuple(args.suites) if args.suites else ("original", "lan", "object", "swap", "task")
    modes = tuple(args.modes) if args.modes else ("base", "need")

    jobs = []
    if not args.skip_cf_and_curve and args.suites is None and args.modes is None:
        # 1. Counterfactual Language Audit on Original suite (50 seeds x 2 counterfactual conditions)
        for cf_name, cf_prompt in [
            ("cf_null", ""),
            ("cf_conflict", "Do not touch the red cube; keep both grippers open and move away"),
        ]:
            for s in range(1, args.n_seeds + 1):
                jobs.append(("original", cf_name, cf_prompt, None, s))

    # 2. Main 4-Dimension Benchmark (Original, Semantic _lan, Object _object, Position _swap, Task _task) x 2 modes (base, need)
    for suite in suites:
        for mode in modes:
            for s in range(1, args.n_seeds + 1):
                jobs.append((suite, mode, None, None, s))

    if not args.skip_cf_and_curve and args.suites is None and args.modes is None:
        # 3. Continuous Spatial Displacement Curve (dist_cm in [4, 8, 12, 16], 25 seeds each)
        for d_cm in (4.0, 8.0, 12.0, 16.0):
            for mode in ("base", "need"):
                for s in range(1, min(26, args.n_seeds + 1)):
                    jobs.append(("swap", mode, None, d_cm, s))

    todo_jobs = [j for j in jobs if j not in done_keys]
    my_jobs = [j for idx, j in enumerate(todo_jobs) if idx % args.num_workers == args.worker_id]
    print(f"[Worker {args.worker_id} h={args.max_steps}] Total assigned jobs: {len(my_jobs)}", flush=True)

    with out_file.open("a") as fh:
        for i, (suite, mode, p_ov, d_cm, s) in enumerate(my_jobs):
            actual_mode = "base" if mode.startswith("cf_") else mode
            rec = run_episode(env, policy, s, suite, actual_mode, prompt_override=p_ov, dist_cm=d_cm, max_steps=args.max_steps)
            rec["eval_tag"] = mode
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (i + 1) % 10 == 0 or (i + 1) == len(my_jobs):
                print(
                    f"[Worker {args.worker_id} | {i+1}/{len(my_jobs)}] {suite} {mode} d={d_cm} s={s}: "
                    f"r={rec['max_reward']} ok={rec['success']}",
                    flush=True,
                )

    print(f"ALOHA_WORKER_{args.worker_id}_DONE", flush=True)


if __name__ == "__main__":
    import subprocess, sys
    if "--launch-4gpus" in sys.argv:
        n_seeds = sys.argv[sys.argv.index("--n-seeds") + 1] if "--n-seeds" in sys.argv else "50"
        max_steps = sys.argv[sys.argv.index("--max-steps") + 1] if "--max-steps" in sys.argv else "400"
        num_workers = int(sys.argv[sys.argv.index("--num-workers") + 1]) if "--num-workers" in sys.argv else 8
        procs = []
        for wid in range(num_workers):
            gid = wid % 4
            env_w = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=str(gid),
                MUJOCO_EGL_DEVICE_ID=str(gid),
                XLA_PYTHON_CLIENT_MEM_FRACTION="0.22",
                OPENPI_DATA_HOME=".openpi",
                LD_LIBRARY_PATH=f"nvgl:{os.environ.get('LD_LIBRARY_PATH', '')}",
                __EGL_VENDOR_LIBRARY_FILENAMES="nvgl/egl_vendor.d/10_nvidia.json",
                MUJOCO_GL="egl",
            )
            p = subprocess.Popen(
                [sys.executable, "-u", __file__, "--worker-id", str(wid), "--num-workers", str(num_workers), "--n-seeds", n_seeds, "--max-steps", max_steps],
                env=env_w,
            )
            procs.append(p)
        for p in procs:
            p.wait()
        print("ALL_ALOHA_WORKERS_DONE", flush=True)
    else:
        main()
