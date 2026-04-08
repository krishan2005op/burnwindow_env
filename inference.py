from burnwindow_env.env import BurnWindowEnv
import json
import os

if __name__ == "__main__":
    task_name = os.getenv("TASK_NAME", "medium")

    env = BurnWindowEnv(task_name=task_name)
    obs = env.reset()

    # REQUIRED: print initial state
    print("[START]")
    print(json.dumps(obs.model_dump()))

    done = False
    while not done:
        # simple random policy
        import random

        action = {
            "action_type": random.choice(["ignite", "hold", "monitor", "suppress"]),
            "unit_id": random.randint(0, len(obs.units) - 1),
        }

        obs, reward, done, _ = env.step(type("A", (), action)())

        print("[STEP]")
        print(json.dumps({
            "day": obs.day,
            "action": action,
            "reward": reward.model_dump(),
            "active_fires": obs.active_fires,
        }))

    print("[END]")