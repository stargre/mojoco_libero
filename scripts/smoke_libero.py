"""Run a no-op LIBERO episode to validate reset, rendering, stepping and logging."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libero_saycan.env import LiberoTaskEnv
from libero_saycan.observations import camera_frame, observation_shapes


def main():
    parser = argparse.ArgumentParser(
        description="LIBERO environment smoke test (zero action; not a task-solving policy)."
    )
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory. Defaults to outputs/smoke/<suite>_task<id>_init<id>.",
    )
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    if args.steps < 0 or args.settle_steps < 0:
        raise SystemExit("--steps and --settle-steps must be non-negative")

    output_dir = (
        Path(args.output)
        if args.output
        else PROJECT_ROOT
        / "outputs"
        / "smoke"
        / "{}_task{}_init{}".format(
            args.suite, args.task_id, args.init_state_id
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    import imageio.v2 as imageio

    started = time.time()
    frames = []
    total_reward = 0.0
    episode_success = False
    executed_steps = 0

    env = LiberoTaskEnv(
        suite_name=args.suite,
        task_id=args.task_id,
        init_state_id=args.init_state_id,
        seed=args.seed,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        settle_steps=args.settle_steps,
        render_gpu_device_id=args.render_gpu_device_id,
    )

    try:
        initial_observation = env.observation
        frames.append(camera_frame(initial_observation, "agentview_image"))
        imageio.imwrite(str(output_dir / "initial_agentview.png"), frames[0])

        neutral_action = np.zeros(env.action_dim, dtype=np.float32)
        for _ in range(args.steps):
            result = env.step(neutral_action)
            executed_steps += 1
            total_reward += result["reward"]
            frames.append(camera_frame(result["observation"], "agentview_image"))
            episode_success = bool(result["success"])
            if episode_success or result["done"]:
                break

        final_observation = env.observation
        final_frame = camera_frame(final_observation, "agentview_image")
        imageio.imwrite(str(output_dir / "final_agentview.png"), final_frame)

        if frames:
            imageio.mimsave(
                str(output_dir / "trajectory.gif"),
                frames,
                duration=1.0 / max(1, args.fps),
                loop=0,
            )

        result_data = {
            "status": (
                "task_success"
                if episode_success
                else "environment_smoke_passed_task_not_solved"
            ),
            "episode_success": bool(episode_success),
            "done": bool(env.last_done),
            "total_reward": float(total_reward),
            "requested_steps": int(args.steps),
            "executed_steps": int(executed_steps),
            "settle_steps": int(args.settle_steps),
            "total_sim_steps": int(env.total_steps),
            "frame_count": len(frames),
            "frame_shape": list(frames[0].shape) if frames else None,
            "observation_shapes": observation_shapes(final_observation),
            "metadata": env.metadata,
            "outputs": {
                "result": "result.json",
                "gif": "trajectory.gif",
                "initial_image": "initial_agentview.png",
                "final_image": "final_agentview.png",
            },
            "elapsed_seconds": round(time.time() - started, 3),
        }
        (output_dir / "result.json").write_text(
            json.dumps(result_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(result_data, indent=2, ensure_ascii=False))
        print("Artifacts:", str(output_dir.resolve()))
        print(
            "NOTE: no-op actions only validate the LIBERO runtime; "
            "they are not expected to solve the language task."
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
