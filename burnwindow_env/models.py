from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


FuelLoad = Literal["low", "medium", "high"]
WindDirection = Literal["N", "S", "E", "W"]
WindSpeed = Literal["low", "medium", "high"]
Humidity = Literal["low", "medium", "high"]
ActionType = Literal["ignite", "hold", "monitor", "suppress", "reassign_crew"]


class UnitState(BaseModel):
    unit_id: int
    fuel_load: FuelLoad
    risk_score: float = Field(ge=0.0, le=1.0)
    burned: bool = False
    near_village: bool = False
    firebreak_present: bool = False
    on_fire: bool = False


class Weather(BaseModel):
    wind_direction: WindDirection
    wind_speed: WindSpeed
    humidity: Humidity


class Observation(BaseModel):
    day: int
    weather: Weather
    units: list[UnitState]
    active_fires: list[int]
    crew_available: int
    crews_total: int


class Action(BaseModel):
    action_type: ActionType
    unit_id: int


class Reward(BaseModel):
    value: float
    details: list[str] = Field(default_factory=list)
