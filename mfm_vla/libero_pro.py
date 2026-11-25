"""Pinned LIBERO-PRO task discovery for the evaluation harness.

The adapter exposes the four pinned, pre-generated perturbations distributed as
BDDL/init pairs. Environment perturbation is generated dynamically by upstream
and is deliberately kept separate from this static-data runner.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import torch


LIBERO_PRO_SOURCE_COMMIT = "eafdb809426b13153aa1e4c42d6601844217dfec"
LIBERO_PRO_DATA_REVISION = "c86fc3b8293185a6f373677018ff3e37f8391602"
BASE_SUITES = ("spatial", "object", "goal", "10")
PERTURBATION_SUFFIXES = {
    "swap": "position",
    "object": "object",
    "lan": "semantic",
    "task": "task",
}
LIBERO_PRO_SUITES = tuple(
    f"libero_{base}_{suffix}"
    for suffix in PERTURBATION_SUFFIXES
    for base in BASE_SUITES
)
LIBERO_PRO_PERTURBATIONS = {
    f"libero_{base}_{suffix}": perturbation
    for suffix, perturbation in PERTURBATION_SUFFIXES.items()
    for base in BASE_SUITES
}

# This is the official task order in libero_suite_task_map.py at the pinned
# source commit.  Do not infer task IDs from filesystem iteration order.
SPATIAL_TASK_NAMES = (
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
)


@dataclass(frozen=True)
class LiberoProTask:
    suite: str
    perturbation: str
    task_id: int
    task_name: str
    bddl_path: Path
    init_path: Path


def default_data_root() -> Path:
    configured = os.environ.get("LIBERO_PRO_DATA")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "third_party" / "LIBERO-PRO-data"


def is_libero_pro_suite(suite_name: str) -> bool:
    return suite_name in LIBERO_PRO_SUITES


def task_names_for_suite(
    suite_name: str, *, data_root: Path | None = None
) -> tuple[str, ...]:
    if suite_name not in LIBERO_PRO_SUITES:
        raise ValueError(f"Unsupported LIBERO-PRO suite {suite_name!r}")
    if suite_name.startswith("libero_spatial_"):
        return SPATIAL_TASK_NAMES
    root = Path(data_root) if data_root is not None else default_data_root()
    directory = root / "bddl_files" / suite_name
    names = tuple(sorted(path.stem for path in directory.glob("*.bddl")))
    if len(names) != 10:
        raise FileNotFoundError(
            f"Expected 10 pinned BDDL files in {directory}, found {len(names)}"
        )
    return names


def get_libero_pro_task(
    suite_name: str,
    task_id: int,
    *,
    data_root: Path | None = None,
) -> LiberoProTask:
    if suite_name not in LIBERO_PRO_SUITES:
        raise ValueError(
            f"Unsupported LIBERO-PRO suite {suite_name!r}; "
            f"expected one of {LIBERO_PRO_SUITES}"
        )
    task_names = task_names_for_suite(suite_name, data_root=data_root)
    if not 0 <= task_id < len(task_names):
        raise IndexError(f"LIBERO-PRO task id must be in [0, 9], got {task_id}")

    root = Path(data_root) if data_root is not None else default_data_root()
    task_name = task_names[task_id]
    bddl_path = root / "bddl_files" / suite_name / f"{task_name}.bddl"
    init_path = root / "init_files" / suite_name / f"{task_name}.pruned_init"
    missing = [str(path) for path in (bddl_path, init_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing pinned LIBERO-PRO data: "
            + ", ".join(missing)
            + ". Run scripts/setup_libero_pro.sh."
        )
    return LiberoProTask(
        suite=suite_name,
        perturbation=LIBERO_PRO_PERTURBATIONS[suite_name],
        task_id=task_id,
        task_name=task_name,
        bddl_path=bddl_path,
        init_path=init_path,
    )


def load_init_states(task: LiberoProTask):
    """Load public benchmark states, including on newer torch versions."""
    try:
        return torch.load(task.init_path, weights_only=False)
    except TypeError:
        return torch.load(task.init_path)
