"""Evaluate dedicated ALOHA-Sim pretrained baselines (ACT, Diffusion Policy, SmolVLA)
on the exact ALOHA-Sim-PRO 5-condition benchmark (50 seeds each = 250 episodes/model):
  - Original: canonical box [0.0, 0.2] x [0.4, 0.6], red cube, "Transfer cube"
  - Semantic (lan): canonical box, red cube, paraphrased instructions
  - Object (object): canonical box, unseen cube colors (blue/green/yellow/cyan/purple)
  - Position (swap): out-of-distribution spatial quadrants outside training box
  - Task (task): unseen colored cube displaced outside default box + task-shift instruction
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import gym_aloha  # noqa: F401
from gym_aloha.constants import START_ARM_POSE
from gym_aloha.utils import sample_box_pose

ROOT = Path(__file__).resolve().parent.parent
CKPT_BASE = ROOT / "checkpoints/aloha_baselines"
OUT_DIR = ROOT / "reports/aloha_sim_pro/baselines"

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


def sample_ood_box_pose(seed: int, dist_cm: float | None = None) -> np.ndarray:
    """Identical OOD spatial sampling to mfm_vla/eval_aloha_sim_pro.py."""
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
    if quad == 0:
        x, y = rng.uniform(0.02, 0.18), rng.uniform(0.28, 0.36)
    elif quad == 1:
        x, y = rng.uniform(0.02, 0.18), rng.uniform(0.64, 0.72)
    elif quad == 2:
        x, y = rng.uniform(-0.08, -0.02), rng.uniform(0.42, 0.58)
    else:
        x, y = rng.uniform(0.22, 0.28), rng.uniform(0.42, 0.58)
    return np.array([x, y, 0.05, 1.0, 0.0, 0.0, 0.0])


class BaselinePolicyRunner:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.preprocessor = None
        self.postprocessor = None
        ckpt_dir = str(CKPT_BASE / model_name)

        if model_name == "act":
            try:
                from lerobot.common.policies.act.modeling_act import ACTPolicy
                self.policy = ACTPolicy.from_pretrained(ckpt_dir).to(device).eval()
                self.legacy = True
            except ImportError:
                from lerobot.policies.act.modeling_act import ACTPolicy
                from lerobot.policies.factory import make_pre_post_processors
                self.policy = ACTPolicy.from_pretrained(ckpt_dir).to(device).eval()
                self.preprocessor, self.postprocessor = make_pre_post_processors(
                    self.policy.config, pretrained_path=ckpt_dir
                )
                self.legacy = False
        elif model_name == "diffusion":
            from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
            from lerobot.policies.factory import make_pre_post_processors
            self.policy = DiffusionPolicy.from_pretrained(ckpt_dir).to(device).eval()
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                self.policy.config, pretrained_path=ckpt_dir
            )
            self.legacy = False
        elif model_name == "dot":
            from lerobot.common.policies.dot.modeling_dot import DOTPolicy
            self.policy = DOTPolicy.from_pretrained(ckpt_dir).to(device).eval()
            self.legacy = True
        elif model_name == "smolvla":
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from lerobot.policies.factory import make_pre_post_processors
            self.policy = SmolVLAPolicy.from_pretrained(ckpt_dir).to(device).eval()
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                self.policy.config, pretrained_path=ckpt_dir
            )
            self.legacy = False
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

    def reset(self):
        self.policy.reset()

    @torch.no_grad()
    def act(self, rgb_top: np.ndarray, agent_pos: np.ndarray, prompt: str) -> np.ndarray:
        img = torch.from_numpy(rgb_top).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)
        st = torch.from_numpy(agent_pos).float().unsqueeze(0).to(self.device)
        batch = {
            "observation.images.top": img,
            "observation.state": st,
        }
        if self.model_name == "smolvla":
            batch["task"] = [prompt]

        if self.preprocessor is not None:
            batch = self.preprocessor(batch)
        action = self.policy.select_action(batch)
        if self.postprocessor is not None:
            action = self.postprocessor(action)
        if isinstance(action, dict):
            action = action["action"]
        return action.squeeze(0).detach().cpu().numpy()


def run_episode(env, runner: BaselinePolicyRunner, seed: int, suite: str, max_steps: int = 400) -> dict:
    obs, _ = env.reset(seed=seed)
    p = env.unwrapped._env.physics

    p.model.geom_rgba[23] = np.array([1.0, 0.0, 0.0, 1.0])
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
        box_pose = sample_ood_box_pose(seed, dist_cm=None)
    elif suite == "task":
        cname, target_rgba = UNSEEN_COLORS[seed % len(UNSEEN_COLORS)]
        p.model.geom_rgba[23] = target_rgba
        box_pose = sample_ood_box_pose(seed, dist_cm=6.0 if (seed % 2 == 0) else 8.0)
        prompt = f"Ignore the default center position; pick up the {cname} cube and transfer it to the left arm"
    else:
        raise ValueError(suite)

    with p.reset_context():
        p.data.qpos[:16] = START_ARM_POSE
        np.copyto(p.data.ctrl, START_ARM_POSE)
        p.data.qpos[16:23] = box_pose
    obs = env.unwrapped._format_raw_obs(env.unwrapped._env.task.get_observation(p))

    l_start = np.array(START_ARM_POSE[:6])
    r_start = np.array(START_ARM_POSE[8:14])
    for _ in range(6):
        obs, _, _, _, _ = env.step(np.concatenate([l_start, [0.30], r_start, [0.30]]))

    runner.reset()
    torch.manual_seed(seed)
    max_r = 0
    steps_used = 6

    while steps_used < max_steps:
        a = runner.act(obs["pixels"]["top"], obs["agent_pos"], prompt)
        obs, r, term, trunc, _ = env.step(a)
        steps_used += 1
        max_r = max(max_r, int(r))
        if term or trunc or steps_used >= max_steps:
            break

    return {
        "model": runner.model_name,
        "max_steps": max_steps,
        "seed": seed,
        "suite": suite,
        "prompt": prompt,
        "max_reward": max_r,
        "touched_right": max_r >= 1,
        "lifted_right": max_r >= 2,
        "bimanual_contact": max_r >= 3,
        "success": max_r == 4,
        "steps": steps_used,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["act", "diffusion", "dot", "smolvla"])
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=400)
    args = parser.parse_args()

    out_dir = OUT_DIR if args.max_steps == 300 else (ROOT / f"reports/aloha_sim_pro/h{args.max_steps}_baselines")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.model}_w{args.worker_id}.jsonl"
    done_keys = set()
    for wf in sorted(out_dir.glob(f"{args.model}_w*.jsonl")):
        for line in wf.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            done_keys.add((r["suite"], int(r["seed"])))

    runner = BaselinePolicyRunner(args.model, device="cuda")
    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", max_episode_steps=args.max_steps)

    jobs = []
    for suite in ("original", "lan", "object", "swap", "task"):
        for s in range(1, args.n_seeds + 1):
            if (suite, s) not in done_keys:
                jobs.append((suite, s))

    my_jobs = [j for idx, j in enumerate(jobs) if idx % args.num_workers == args.worker_id]
    print(f"[{args.model} h={args.max_steps} Worker {args.worker_id}] Assigned {len(my_jobs)} jobs", flush=True)

    with out_file.open("a") as fh:
        for i, (suite, s) in enumerate(my_jobs):
            rec = run_episode(env, runner, s, suite, max_steps=args.max_steps)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (i + 1) % 10 == 0 or (i + 1) == len(my_jobs):
                print(
                    f"[{args.model} W{args.worker_id} | {i+1}/{len(my_jobs)}] {suite} s={s}: "
                    f"r={rec['max_reward']} ok={rec['success']}",
                    flush=True,
                )
    env.close()
    print(f"{args.model.upper()}_WORKER_{args.worker_id}_DONE", flush=True)


if __name__ == "__main__":
    if "--launch-4gpus" in sys.argv:
        idx = sys.argv.index("--model")
        model_name = sys.argv[idx + 1]
        n_seeds = sys.argv[sys.argv.index("--n-seeds") + 1] if "--n-seeds" in sys.argv else "50"
        max_steps = sys.argv[sys.argv.index("--max-steps") + 1] if "--max-steps" in sys.argv else "400"
        num_workers = int(sys.argv[sys.argv.index("--num-workers") + 1]) if "--num-workers" in sys.argv else 4
        procs = []
        for wid in range(num_workers):
            gid = wid % 4
            env_w = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=str(gid),
                MUJOCO_EGL_DEVICE_ID=str(gid),
                LD_LIBRARY_PATH=f"nvgl:{os.environ.get('LD_LIBRARY_PATH', '')}",
                __EGL_VENDOR_LIBRARY_FILENAMES="nvgl/egl_vendor.d/10_nvidia.json",
                MUJOCO_GL="egl",
            )
            p = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    __file__,
                    "--model",
                    model_name,
                    "--worker-id",
                    str(wid),
                    "--num-workers",
                    str(num_workers),
                    "--n-seeds",
                    n_seeds,
                    "--max-steps",
                    max_steps,
                ],
                env=env_w,
            )
            procs.append(p)
        rc = 0
        for p in procs:
            rc |= p.wait()
        if rc != 0:
            sys.exit(rc)
        print(f"ALL_{num_workers}_{model_name.upper()}_WORKERS_DONE", flush=True)
    else:
        main()
