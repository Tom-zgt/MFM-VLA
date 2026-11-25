"""Mark providers for the visual-prompt π0.5 executor (``OpenPiVLAExecutor.image_marker``).

``OracleMarker`` reads the BDDL goal and simulator poses. It is privileged and only
gives the upper bound of what marks can buy; deployment uses a vision-language
marker instead.
"""

from __future__ import annotations

import numpy as np

from harness import AGENT_CAM
from visual_prompt import draw_marks, goal_targets, project_target

EVAL_RADIUS = 9.5


class OracleMarker:
    """Marks the first unmet goal predicate, projected when it becomes the active one."""

    def __init__(self) -> None:
        self.active: int | None = None
        self.uv = (None, None)
        self.history: list[dict] = []

    def __call__(self, env, obs, image: np.ndarray) -> np.ndarray:
        inner = env._env.env
        goals = list(inner.parsed_problem["goal_state"])
        pending = [i for i, g in enumerate(goals) if not inner._eval_predicate(g)]
        current = pending[0] if pending else (self.active or 0)
        if current != self.active:
            size = image.shape[1]
            pick, place = goal_targets(goals[current])
            self.uv = (project_target(inner, pick, obs, AGENT_CAM, size),
                       project_target(inner, place, obs, AGENT_CAM, size))
            self.active = current
            self.history.append({"t": int(env.t), "goal": [str(x) for x in goals[current]],
                                 "pick_uv": self.uv[0].tolist(), "place_uv": self.uv[1].tolist()})
        return draw_marks(image, self.uv[0], self.uv[1], radius=EVAL_RADIUS)
