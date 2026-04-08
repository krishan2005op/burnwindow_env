from burnwindow_env.inference import run_episode
import os

if __name__ == "__main__":
    task_name = os.getenv("TASK_NAME", "medium")
    run_episode(task_name=task_name)