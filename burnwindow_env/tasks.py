from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskConfig:
    name: str
    num_units: int
    deterministic_weather: bool
    include_villages: bool
    crews_total: int
    escape_multiplier: float


TASKS: dict[str, TaskConfig] = {
    "easy": TaskConfig(
        name="easy",
        num_units=5,
        deterministic_weather=True,
        include_villages=False,
        crews_total=3,
        escape_multiplier=0.6,
    ),
    "medium": TaskConfig(
        name="medium",
        num_units=10,
        deterministic_weather=False,
        include_villages=True,
        crews_total=3,
        escape_multiplier=1.0,
    ),
    "hard": TaskConfig(
        name="hard",
        num_units=15,
        deterministic_weather=False,
        include_villages=True,
        crews_total=2,
        escape_multiplier=1.5,
    ),
}


def get_task_config(task_name: str) -> TaskConfig:
    if task_name not in TASKS:
        raise ValueError(f"Unknown task '{task_name}'. Choose from: {list(TASKS)}")
    return TASKS[task_name]
