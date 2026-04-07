from __future__ import annotations

import random
from typing import Any

from .models import Action, Observation, Reward, UnitState, Weather
from .tasks import TaskConfig, get_task_config


class BurnWindowEnv:
    def __init__(self, task_name: str = "easy", seed: int | None = None) -> None:
        self.task: TaskConfig = get_task_config(task_name)
        self.rng = random.Random(seed)
        self.max_days = 30
        self.day = 1
        self.crews_total = self.task.crews_total
        self.crew_available = self.crews_total
        self.processed_units: set[int] = set()
        self.units: list[UnitState] = []
        self.weather = Weather(wind_direction="N", wind_speed="low", humidity="high")
        self.reset()

    def reset(self) -> Observation:
        self.day = 1
        self.crew_available = self.crews_total
        self.processed_units = set()
        self.units = self._generate_units()
        self.weather = self._get_weather_for_day(self.day)
        return self.state()

    def state(self) -> Observation:
        active_fires = [u.unit_id for u in self.units if u.on_fire]
        return Observation(
            day=self.day,
            weather=self.weather,
            units=self.units,
            active_fires=active_fires,
            crew_available=self.crew_available,
            crews_total=self.crews_total,
        )

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        reward = 0.0
        details: list[str] = []

        if not (0 <= action.unit_id < len(self.units)):
            reward -= 1.0
            details.append("Invalid unit_id")
            self._advance_day()
            done = self._is_done()
            return self.state(), Reward(value=reward, details=details), done, {}

        target = self.units[action.unit_id]
        self.processed_units.add(action.unit_id)
        self.crew_available = max(0, self.crew_available - 1)

        if action.action_type == "ignite":
            if self._is_safe_to_ignite(target):
                success_p = self._burn_success_probability(target)
                if self.rng.random() < success_p:
                    target.burned = True
                    target.on_fire = True
                    reward += 1.0
                    details.append("Successful controlled burn")
                    if target.risk_score >= 0.7:
                        reward += 0.3
                        details.append("Burned high-risk unit")
                else:
                    details.append("Ignition attempted, burn did not sustain")
                escape = self._simulate_spread(target.unit_id)
                if escape:
                    reward -= 5.0
                    details.append("Fire escaped boundary")
                if target.near_village and self._smoke_hits_village():
                    reward -= 2.0
                    details.append("Smoke affected nearby village")
            else:
                reward -= 1.0
                details.append("Unsafe ignition blocked/penalized")

        elif action.action_type == "hold":
            reward -= 0.5
            details.append("Idle crew")

        elif action.action_type == "monitor":
            if (not target.burned) and target.risk_score >= 0.6:
                reward += 0.2
                details.append("Useful monitoring on risky unit")
            else:
                details.append("Monitoring complete")

        elif action.action_type == "suppress":
            if target.on_fire:
                target.on_fire = False
                reward += 0.5
                details.append("Active fire suppressed")
            else:
                reward -= 0.2
                details.append("Suppression used with no active fire")

        elif action.action_type == "reassign_crew":
            self.crew_available = min(self.crews_total, self.crew_available + 1)
            details.append("Crew reassigned")

        # Fires may naturally die if humidity is high.
        self._cooldown_fires()
        self._advance_day()
        done = self._is_done()
        info = {"processed_units": len(self.processed_units), "task": self.task.name}
        return self.state(), Reward(value=round(reward, 3), details=details), done, info

    def _generate_units(self) -> list[UnitState]:
        units: list[UnitState] = []
        fuel_levels = ["low", "medium", "high"]
        for i in range(self.task.num_units):
            fuel = self.rng.choice(fuel_levels)
            risk = round(self.rng.uniform(0.2, 0.95), 2)
            near_village = False
            if self.task.include_villages:
                near_village = self.rng.random() < 0.35
            firebreak_present = self.rng.random() < 0.75
            units.append(
                UnitState(
                    unit_id=i,
                    fuel_load=fuel,  # type: ignore[arg-type]
                    risk_score=risk,
                    burned=False,
                    near_village=near_village,
                    firebreak_present=firebreak_present,
                    on_fire=False,
                )
            )
        return units

    def _get_weather_for_day(self, day: int) -> Weather:
        if self.task.deterministic_weather:
            cycle = [
                Weather(wind_direction="N", wind_speed="low", humidity="high"),
                Weather(wind_direction="E", wind_speed="medium", humidity="medium"),
                Weather(wind_direction="W", wind_speed="low", humidity="medium"),
            ]
            return cycle[(day - 1) % len(cycle)]

        return Weather(
            wind_direction=self.rng.choice(["N", "S", "E", "W"]),  # type: ignore[arg-type]
            wind_speed=self.rng.choice(["low", "medium", "high"]),  # type: ignore[arg-type]
            humidity=self.rng.choice(["low", "medium", "high"]),  # type: ignore[arg-type]
        )

    def _is_safe_to_ignite(self, unit: UnitState) -> bool:
        wind_safe = self.weather.wind_speed in {"low", "medium"}
        humidity_safe = self.weather.humidity in {"medium", "high"}
        return wind_safe and humidity_safe and unit.firebreak_present

    def _burn_success_probability(self, unit: UnitState) -> float:
        base = {"low": 0.45, "medium": 0.65, "high": 0.8}[unit.fuel_load]
        humidity_factor = {"low": 1.0, "medium": 0.9, "high": 0.75}[self.weather.humidity]
        wind_factor = {"low": 0.95, "medium": 1.0, "high": 0.8}[self.weather.wind_speed]
        return max(0.1, min(0.95, base * humidity_factor * wind_factor))

    def _simulate_spread(self, source_unit_id: int) -> bool:
        source = self.units[source_unit_id]
        if not source.on_fire:
            return False

        base_escape = {"low": 0.04, "medium": 0.1, "high": 0.2}[self.weather.wind_speed]
        humidity_adj = {"low": 1.3, "medium": 1.0, "high": 0.7}[self.weather.humidity]
        spread_p = base_escape * humidity_adj * self.task.escape_multiplier

        escaped = self.rng.random() < spread_p
        if escaped:
            for neighbor_idx in (source_unit_id - 1, source_unit_id + 1):
                if 0 <= neighbor_idx < len(self.units):
                    neighbor = self.units[neighbor_idx]
                    if not neighbor.burned:
                        neighbor.on_fire = True
            return True
        return False

    def _smoke_hits_village(self) -> bool:
        speed_factor = {"low": 0.05, "medium": 0.15, "high": 0.35}[self.weather.wind_speed]
        humidity_factor = {"low": 1.2, "medium": 1.0, "high": 0.8}[self.weather.humidity]
        return self.rng.random() < (speed_factor * humidity_factor)

    def _cooldown_fires(self) -> None:
        if self.weather.humidity == "high":
            for unit in self.units:
                if unit.on_fire and self.rng.random() < 0.6:
                    unit.on_fire = False

    def _advance_day(self) -> None:
        self.day += 1
        self.crew_available = self.crews_total
        if self.day <= self.max_days:
            self.weather = self._get_weather_for_day(self.day)

    def _is_done(self) -> bool:
        all_processed = len(self.processed_units) >= len(self.units)
        day_limit_reached = self.day > self.max_days
        return all_processed or day_limit_reached
