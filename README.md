# BurnWindow: Prescribed Fire Authorization Agent

BurnWindow is a simplified OpenEnv environment for training and evaluating decision-making around controlled forest burns.  
The goal is to reduce wildfire risk while avoiding unsafe ignitions, smoke harm to villages, and fire escapes.

## Problem Overview

Forest managers must decide daily actions for each forest unit:

- When to ignite a controlled burn
- When to hold or monitor
- When to suppress active fire
- How to reassign limited crews

This environment intentionally uses simplified fire behavior:

- Probabilistic burn success
- Semi-random weather (for medium/hard tasks)
- Simple spread and smoke models

## Environment API

Implemented core methods:

- `reset() -> Observation`
- `state() -> Observation`
- `step(action: Action) -> (Observation, Reward, done, info)`

Each `step` equals one day in a 30-day episode.

Episode ends when:

- all units are processed, or
- day limit is reached.

## State (Observation)

Observation includes:

- `day` (1..30)
- `weather`:
  - `wind_direction`: `N|S|E|W`
  - `wind_speed`: `low|medium|high`
  - `humidity`: `low|medium|high`
- `units` list, each with:
  - `fuel_load`: `low|medium|high`
  - `risk_score`: `0.0..1.0`
  - `burned`: bool
  - `near_village`: bool
  - `firebreak_present`: bool
  - `on_fire`: bool
- `active_fires` (list of unit ids)
- `crew_available` and `crews_total`

## Action Space

Pydantic `Action` supports:

- `ignite(unit_id)`
- `hold(unit_id)`
- `monitor(unit_id)`
- `suppress(unit_id)`
- `reassign_crew(unit_id)`

## Reward Design (Dense)

- `+1.0` successful controlled burn
- `+0.3` good planning: high-risk unit burned
- `-0.5` idle crew (`hold`)
- `-1.0` unsafe ignition
- `-2.0` smoke affects village
- `-5.0` fire escapes boundary

## Tasks

Defined in `tasks.py`:

1. **easy**
   - 5 units
   - deterministic weather
   - no villages
2. **medium**
   - 10 units
   - stochastic weather
   - some villages
3. **hard**
   - 15 units
   - stochastic weather
   - limited crews
   - higher escape probability

## Grading

`grader.py` returns a normalized score in `[0.0, 1.0]`, combining:

- burn coverage
- reward quality
- speed
- penalties for escapes and smoke events

## Run Locally

From parent directory (recommended):

```bash
python -m burnwindow_env.inference
```

Optional env vars:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `TASK_NAME` (`easy|medium|hard`)

If model call fails, the script automatically uses a fallback policy.

## Docker

Build:

```bash
docker build -t burnwindow-env ./burnwindow_env
```

Run:

```bash
docker run --rm \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o-mini" \
  -e HF_TOKEN="your_token_here" \
  -e TASK_NAME="medium" \
  burnwindow-env
```

## Required Inference Output Format

`inference.py` prints:

- `[START]` once
- `[STEP]` before each step payload
- `[END]` once at termination

## Example Output

```text
[START]
[STEP]
{"day": 2, "action": {"action_type": "ignite", "unit_id": 4}, "reward": {"value": 1.3, "details": ["Successful controlled burn", "Burned high-risk unit"]}, "active_fires": [4]}
[STEP]
{"day": 3, "action": {"action_type": "suppress", "unit_id": 4}, "reward": {"value": 0.5, "details": ["Active fire suppressed"]}, "active_fires": []}
{"final_score": 0.73, "stats": {"total_reward": 4.1, "days_used": 10, "burned_units": 6, "total_units": 10, "escapes": 1, "smoke_events": 0}}
[END]
```
