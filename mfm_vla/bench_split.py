"""Separate physics cost from render cost, and check how much parallelism we have."""

import os
import time

import libero_compat  # noqa: F401
import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def main():
    print(f"cpu_count={os.cpu_count()}")

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=128,
        camera_widths=128,
    )
    env.seed(0)
    env.reset()
    env.set_init_state(suite.get_task_init_states(0)[0])

    inner = env.env  # robosuite env
    sim = inner.sim
    a = np.zeros(7)

    for _ in range(10):
        env.step(a)

    # 1. Full env.step (physics + both camera renders + obs assembly)
    t0 = time.time()
    for _ in range(50):
        env.step(a)
    full = (time.time() - t0) / 50

    # 2. Raw MuJoCo physics only, no rendering at all.
    n_sub = int(inner.control_freq and round(1.0 / (inner.control_freq * inner.model_timestep)))
    t0 = time.time()
    for _ in range(50):
        for _ in range(n_sub):
            sim.step()
    phys = (time.time() - t0) / 50

    # 3. One offscreen render call.
    t0 = time.time()
    for _ in range(50):
        sim.render(width=128, height=128, camera_name="agentview")
    render = (time.time() - t0) / 50

    print(f"\nsim_timestep={inner.model_timestep}  control_freq={inner.control_freq}  "
          f"sub_steps_per_env_step={n_sub}")
    print(f"full env.step      {full*1000:8.2f} ms   ({1/full:6.1f} Hz)")
    print(f"physics only       {phys*1000:8.2f} ms   ({1/phys:6.1f} Hz)")
    print(f"one render (128px) {render*1000:8.2f} ms   ({1/render:6.1f} Hz)")
    print(f"\n-> render is {2*render/full*100:.0f}% of env.step (2 cameras)")

    env.close()


if __name__ == "__main__":
    main()
