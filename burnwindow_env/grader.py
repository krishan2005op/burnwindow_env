from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpisodeStats:
    total_reward: float
    days_used: int
    burned_units: int
    total_units: int
    escapes: int
    smoke_events: int


def grade_episode(stats: EpisodeStats, task_name: str) -> float:
    """
    Returns a score in [0.0, 1.0].
    Higher is better.
    """
    if stats.total_units <= 0:
        return 0.0

    burn_ratio = stats.burned_units / stats.total_units
    speed_score = max(0.0, 1.0 - (stats.days_used / 30.0))
    reward_norm = (stats.total_reward + 10.0) / 25.0
    reward_norm = max(0.0, min(1.0, reward_norm))

    penalty = 0.0
    penalty += min(0.6, stats.escapes * 0.25)
    penalty += min(0.3, stats.smoke_events * 0.1)

    if task_name == "easy":
        raw = _grade_easy(burn_ratio, reward_norm, speed_score, penalty)
    elif task_name == "medium":
        raw = _grade_medium(burn_ratio, reward_norm, speed_score, penalty)
    else:
        raw = _grade_hard(burn_ratio, reward_norm, speed_score, penalty)

    return max(0.0, min(1.0, raw))


def _grade_easy(burn_ratio: float, reward_norm: float, speed_score: float, penalty: float) -> float:
    return (0.5 * burn_ratio) + (0.3 * reward_norm) + (0.2 * speed_score) - penalty


def _grade_medium(burn_ratio: float, reward_norm: float, speed_score: float, penalty: float) -> float:
    return (0.45 * burn_ratio) + (0.35 * reward_norm) + (0.2 * speed_score) - penalty


def _grade_hard(burn_ratio: float, reward_norm: float, speed_score: float, penalty: float) -> float:
    return (0.4 * burn_ratio) + (0.4 * reward_norm) + (0.2 * speed_score) - penalty


TASK_GRADERS = {
    "easy": _grade_easy,
    "medium": _grade_medium,
    "hard": _grade_hard,
}
