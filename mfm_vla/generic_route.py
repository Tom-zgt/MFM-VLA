"""Route an episode: leave it to the VLA when it is one of the VLA's own training
tasks in the same arrangement, otherwise assist.

Two GPT calls at step 0 (no simulator time): the first plans the instruction into
single-skill steps and names the closest training task, if the instruction asks
for the same thing; the second compares the start image with that training
task's start image. The catalog holds only the VLA's training tasks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from eval_program import parse_program

PLAN_AND_MATCH_PROMPT = """You plan for a robot arm with a parallel gripper, seen from a fixed camera (image).
Instruction: "{instruction}"

1. Split the instruction into the shortest ordered list of single-skill steps. Each step is one of:
- "carry": pick up one object and put it into, onto, or next to something.
- "act": operate one thing without carrying it (open, close, turn on or off, push...).
Order the steps so each is physically possible when its turn comes. Do not add steps the
instruction does not ask for. For each step write "instruction": a short imperative sentence
reusing the instruction's own words and naming objects exactly as it does. For a carry step also
set "onto_object": true if the object goes into or onto another object, false otherwise.
For an act step set "act_kind": "drawer" to slide a drawer open or closed, "door" to swing a
hinged door or lid, "other" for anything else (knobs, buttons, pushing an object...).

2. The robot's policy was trained on exactly these tasks:
{catalog}
Give "match": the number of the training task that asks for the same outcome as the
instruction (same object moved or operated, same destination or state, same
disambiguating description such as "next to X" or "on Y"), allowing only different
wording. Naming an object more or less specifically (a brand or pattern name, a colour,
"the X box" for "the X") is only wording. If none asks for the same outcome, "match": null.

Return JSON only:
{{"steps": [{{"skill": "carry" or "act", "instruction": "...", "onto_object": true or false,
  "act_kind": "drawer" or "door" or "other"}}],
  "match": <number or null>, "match_reason": "<one short sentence>"}}"""

PLAN_PROMPT = PLAN_AND_MATCH_PROMPT.split("\n\n2. ")[0] + """

Return JSON only:
{{"steps": [{{"skill": "carry" or "act", "instruction": "...", "onto_object": true or false,
  "act_kind": "drawer" or "door" or "other"}}]}}"""

LAYOUT_PROMPT = """Images from the same fixed camera above a robot's workspace.
{n_ref} "reference" images are different starts of a task the robot's policy was trained on:
"{reference}". Objects in them are placed with some random variation; that variation is normal
and the policy handles it.
"current" is the start of the task to do now: "{instruction}".

Is the current scene arranged like the references, so that the policy's trained motion would
reach the same objects? For every object the task involves (the object to move or operate and
its destination) and every object standing next to them, check: is it present; is it on the
same supporting surface; is it in the same place relative to the other objects as in the
references? It is different if two objects have traded places, an object was replaced by
another, or an object is clearly outside the range of positions the references show. A shift
no larger than the references' own variation is not a difference. Ignore lighting and objects
far from the task.
Return JSON only:
{{"same_arrangement": true or false, "differences": "<one short sentence>"}}"""


def load_catalog(path: Path) -> tuple[list[dict], Path]:
    entries = json.loads(path.read_text())
    return entries, path.parent


def route(session, instruction: str, image: np.ndarray, catalog: list[dict] | None, catalog_dir: Path | None,
          program_cfg, assist_acts: tuple[str, ...] = ()) -> dict:
    """Returns {"program", "carries", "match", "same_arrangement", "route": "direct"|"assist", ...}.
    ``assist_acts``: act kinds the harness can do (G5e: "drawer"); other act-only
    instructions always go to the VLA. Without a ``catalog`` the route follows from the
    plan of the instruction and the start image alone: any carry or harness act is assisted."""
    if catalog is None:
        reply = session.decide_json(PLAN_PROMPT.format(instruction=instruction), {"current": image})
        catalog = []
    else:
        listing = "\n".join(f"{e['id']}. {e['instruction']}" for e in catalog)
        reply = session.decide_json(
            PLAN_AND_MATCH_PROMPT.format(instruction=instruction, catalog=listing), {"current": image})
    program = parse_program(reply, program_cfg)
    carries = any(step["skill"] == "carry" for step in program)
    acts = [step["act_kind"] for step in program if step.get("act_kind") in assist_acts]
    match = reply.get("match")
    entry = next((e for e in catalog if isinstance(match, int) and e["id"] == match), None)
    out = {"program": program, "carries": carries, "assist_acts": acts,
           "match": None if entry is None else entry["id"],
           "match_instruction": None if entry is None else entry["instruction"],
           "match_reason": reply.get("match_reason"), "same_arrangement": None}
    if not carries and not acts:
        out["route"] = "direct"
        return out
    if entry is None:
        out["route"] = "assist"
        return out
    names = entry.get("images") or [entry["image"]]
    images = {f"reference_{i + 1}": np.asarray(Image.open(catalog_dir / name).convert("RGB"))
              for i, name in enumerate(names)}
    images["current"] = image
    layout = session.decide_json(
        LAYOUT_PROMPT.format(n_ref=len(names), reference=entry["instruction"], instruction=instruction),
        images)
    out["same_arrangement"] = bool(layout.get("same_arrangement"))
    out["differences"] = layout.get("differences")
    out["route"] = "direct" if out["same_arrangement"] else "assist"
    return out
