from __future__ import annotations

import json
import os
import random
from typing import Any

from openai import OpenAI

from burnwindow_env.env import BurnWindowEnv
from burnwindow_env.grader import EpisodeStats, grade_episode
from burnwindow_env.models import Action

def _fallback_policy(observation: dict[str, Any]) -> dict[str, Any]:
    units = observation["units"]
    weather = observation["weather"]
    rng = random.SystemRandom()

    # Prefer safe ignition of highest-risk unit with firebreak.
    candidates = [
        u
        for u in units
        if (not u["burned"]) and u["firebreak_present"] and u["risk_score"] >= 0.6
    ]
    if candidates and weather["wind_speed"] != "high" and weather["humidity"] != "low":
        top_k = sorted(candidates, key=lambda x: x["risk_score"], reverse=True)[: min(3, len(candidates))]
        chosen = rng.choice(top_k)
        return {"action_type": "ignite", "unit_id": chosen["unit_id"]}

    # Suppress if anything is burning.
    active = observation["active_fires"]
    if active:
        return {"action_type": "suppress", "unit_id": rng.choice(active)}

    # Otherwise monitor a top-risk unburned unit.
    unburned = [u for u in units if not u["burned"]]
    if unburned:
        unburned.sort(key=lambda x: x["risk_score"], reverse=True)
        top_k = unburned[: min(4, len(unburned))]
        chosen = rng.choice(top_k)
        return {"action_type": "monitor", "unit_id": chosen["unit_id"]}

    return {"action_type": "hold", "unit_id": rng.randint(0, max(0, len(units) - 1))}


def _model_action(
    client: OpenAI, model_name: str, observation: dict[str, Any]
) -> dict[str, Any]:
    prompt = (
        "You are selecting one action for a prescribed burn environment.\n"
        "Return JSON only with keys: action_type, unit_id.\n"
        "Valid action_type: ignite, hold, monitor, suppress, reassign_crew.\n"
        "Choose a legal unit_id from observation units."
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(observation)},
        ],
        temperature=0.8,
    )
    content = response.choices[0].message.content or "{}"
    return json.loads(content)


def run_episode(task_name: str = "medium", seed: int | None = None) -> None:
    base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    hf_token = os.getenv("HF_TOKEN", "")

    client = OpenAI(base_url=base_url, api_key=hf_token)
    if seed is None:
        seed = random.SystemRandom().randint(1, 10_000_000)
    env = BurnWindowEnv(task_name=task_name, seed=seed)
    obs = env.reset()

    total_reward = 0.0
    escapes = 0
    smoke_events = 0

    print("[START]")
    print(json.dumps({"task": task_name, "seed": seed}))
    done = False
    while not done:
        obs_dict = obs.model_dump()

        try:
            chosen = _model_action(client, model_name, obs_dict)
        except Exception:
            chosen = _fallback_policy(obs_dict)

        try:
            action = Action(**chosen)
        except Exception:
            action = Action(**_fallback_policy(obs_dict))

        obs, reward, done, _info = env.step(action)
        total_reward += reward.value
        if any("escaped" in d.lower() for d in reward.details):
            escapes += 1
        if any("smoke" in d.lower() for d in reward.details):
            smoke_events += 1

        print("[STEP]")
        print(
            json.dumps(
                {
                    "day": obs.day,
                    "action": action.model_dump(),
                    "reward": reward.model_dump(),
                    "active_fires": obs.active_fires,
                }
            )
        )

    burned_units = sum(1 for u in obs.units if u.burned)
    stats = EpisodeStats(
        total_reward=round(total_reward, 3),
        days_used=min(obs.day, 30),
        burned_units=burned_units,
        total_units=len(obs.units),
        escapes=escapes,
        smoke_events=smoke_events,
    )
    score = grade_episode(stats, task_name=task_name)
    print(json.dumps({"final_score": round(score, 3), "stats": stats.__dict__}))
    print("[END]")


if __name__ == "__main__":
    task_name = os.getenv("TASK_NAME", "medium")
    seed_env = os.getenv("SEED")
    seed = int(seed_env) if seed_env is not None else None
    run_episode(task_name=task_name, seed=seed)
