"""M5 verification: GPT checks the real camera images at decision points.

Grasp check: right after a proprioceptive grasp-and-lift, before carrying, is the
object in the gripper the one the instruction asks for? The simulator is paused
while GPT looks, so the check costs no environment steps.
"""

from __future__ import annotations

import time

GRASP_CHECK_PROMPT = """You are the verification module of a robot arm.
Instruction: "{instruction}"
The robot has just closed its gripper and lifted it a little. The first image is a fixed
camera; the second is the camera on the robot's wrist, looking down past the gripper fingers.

Which object, if any, is held between the gripper fingers right now? Is it the object the
instruction asks the robot to pick up? Identify objects by appearance (shape, colour, printed
text), not by location.
Reply with JSON only:
{{"held": "<short name, or nothing>", "correct": true or false, "evidence": "<one short sentence>"}}"""


GRASP_CHECK_WITH_REFERENCE_PROMPT = """You are the verification module of a robot arm.
Instruction: "{instruction}"
Before moving, the perception module identified the object to pick up: the yellow box in the
first image (a crop of the fixed camera at the start) marks it; it was described as
"{name}: {evidence}".
The robot has just closed its gripper and lifted it a little. The second image is the fixed
camera now; the third is the camera on the robot's wrist, looking down past the gripper fingers.

Which object, if any, is held between the gripper fingers right now? Is it the same physical
object as the one in the yellow box? Compare appearance (shape, colour, cap, printed text);
do not rely on the object's name alone.
Reply with JSON only:
{{"held": "<short name, or nothing>", "correct": true or false, "evidence": "<one short sentence>"}}"""


def check_grasp(adapter, session, instruction: str, size: int, reference: dict | None = None,
                wrist: str = "robot0_eye_in_hand", fixed: str = "agentview") -> dict:
    """``reference``: {"image": start crop with the target boxed, "name", "evidence"} from grounding."""
    obs = adapter.observe([fixed, wrist], size)
    asked = time.monotonic()
    if reference is None:
        prompt, images = GRASP_CHECK_PROMPT.format(instruction=instruction), {}
    else:
        prompt = GRASP_CHECK_WITH_REFERENCE_PROMPT.format(
            instruction=instruction, name=reference["name"], evidence=reference["evidence"])
        images = {"reference": reference["image"]}
    images.update({fixed: obs.rgb[fixed], wrist: obs.rgb[wrist]})
    reply = session.decide_json(prompt, images)
    return {
        "t": adapter.t, "held": reply.get("held"), "correct": bool(reply.get("correct")),
        "evidence": reply.get("evidence"), "latency_s": round(time.monotonic() - asked, 2),
    }
