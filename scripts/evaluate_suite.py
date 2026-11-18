"""Evaluate SayCan across official LIBERO task suites and fixed initial states."""

import argparse
import csv
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libero_saycan.agent import LiberoSayCanAgent
from libero_saycan.bootstrap import configure_libero
from libero_saycan.env import LiberoTaskEnv
from libero_saycan.metrics import summarize_episodes
from libero_saycan.qwen import KeywordRelevanceScorer, QwenRelevanceScorer


ALL_SUITES = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_90",
    "libero_10",
]


def parse_task_ids(value, task_count):
    if not value:
        return list(range(task_count))
    result = []
    for part in value.split(","):
        part = part.strip()
        if part:
            index = int(part)
            if index < 0 or index >= task_count:
                raise ValueError("task id {} outside [0, {})".format(index, task_count))
            result.append(index)
    return result


def make_scorer(kind):
    return QwenRelevanceScorer() if kind == "qwen" else KeywordRelevanceScorer()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen-SayCan on official LIBERO task suites."
    )
    parser.add_argument(
        "--suite",
        choices=ALL_SUITES + ["all"],
        default="libero_spatial",
    )
    parser.add_argument("--task-ids", default=None, help="Comma-separated task IDs.")
    parser.add_argument("--task-limit", type=int, default=0, help="0 means all tasks.")
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--scorer", choices=["qwen", "keyword"], default="qwen")
    parser.add_argument("--max-rounds", type=int, default=40)
    parser.add_argument("--max-motion-steps", type=int, default=100)
    parser.add_argument("--gripper-steps", type=int, default=110)
    parser.add_argument("--render-gpu-device-id", type=int, default=0)
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--output", default="outputs/evaluations")
    args = parser.parse_args()

    configure_libero()
    from libero.libero import benchmark
    registry = benchmark.get_benchmark_dict()
    suites = ALL_SUITES if args.suite == "all" else [args.suite]
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    all_episodes = []
    started = time.time()

    for suite_name in suites:
        if suite_name not in registry:
            raise ValueError("LIBERO suite not registered: {}".format(suite_name))
        task_suite = registry[suite_name]()
        task_count = task_suite.get_num_tasks()
        task_ids = parse_task_ids(args.task_ids, task_count) if len(suites) == 1 else list(range(task_count))
        if args.task_limit > 0:
            task_ids = task_ids[:args.task_limit]

        print("\n=== {}: {} tasks selected ===".format(suite_name, len(task_ids)))
        suite_episodes = []

        for task_id in task_ids:
            # Load the official init-state tensor once to determine its length.
            init_states = task_suite.get_task_init_states(task_id)
            episode_count = min(max(1, args.episodes_per_task), len(init_states))
            for init_state_id in range(episode_count):
                episode_started = time.time()
                env = None
                try:
                    env = LiberoTaskEnv(
                        suite_name=suite_name,
                        task_id=task_id,
                        init_state_id=init_state_id,
                        seed=init_state_id,
                        camera_height=args.camera_height,
                        camera_width=args.camera_width,
                        horizon=args.horizon,
                        render_gpu_device_id=args.render_gpu_device_id,
                    )
                    scorer = make_scorer(args.scorer)
                    agent = LiberoSayCanAgent(
                        env=env,
                        scorer=scorer,
                        recorder=None,
                        max_rounds=args.max_rounds,
                        max_motion_steps=args.max_motion_steps,
                        gripper_steps=args.gripper_steps,
                    )
                    result = agent.run()
                    episode = {
                        "suite": suite_name,
                        "task_id": int(task_id),
                        "task_name": env.task_name,
                        "instruction": env.instruction,
                        "init_state_id": int(init_state_id),
                        "seed": int(init_state_id),
                        "success": bool(result["success"]),
                        "status": result["status"],
                        "rounds": int(result.get("rounds", 0)),
                        "qwen_request_count": int(result.get("qwen_request_count", 0)),
                        "total_sim_steps": int(env.total_steps),
                        "elapsed_seconds": round(time.time() - episode_started, 3),
                        "error": None,
                    }
                except Exception as exc:
                    episode = {
                        "suite": suite_name,
                        "task_id": int(task_id),
                        "task_name": getattr(getattr(env, "task", None), "name", ""),
                        "instruction": getattr(env, "instruction", ""),
                        "init_state_id": int(init_state_id),
                        "seed": int(init_state_id),
                        "success": False,
                        "status": "exception",
                        "rounds": 0,
                        "qwen_request_count": 0,
                        "total_sim_steps": int(getattr(env, "total_steps", 0)),
                        "elapsed_seconds": round(time.time() - episode_started, 3),
                        "error": "{}: {}".format(type(exc).__name__, str(exc)),
                    }
                finally:
                    if env is not None:
                        env.close()

                suite_episodes.append(episode)
                all_episodes.append(episode)
                print(
                    "[{}] task {}/{} init {} -> {} ({})".format(
                        suite_name, task_id + 1, task_count, init_state_id,
                        "SUCCESS" if episode["success"] else "FAIL",
                        episode["status"],
                    )
                )

        suite_summary = summarize_episodes([
            {
                "suite": suite_name,
                "episode_success": item["success"],
            }
            for item in suite_episodes
        ])
        (output_root / "{}_episodes.json".format(suite_name)).write_text(
            json.dumps(suite_episodes, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_root / "{}_summary.json".format(suite_name)).write_text(
            json.dumps(suite_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print("Suite summary:", json.dumps(suite_summary, ensure_ascii=False))

    summary = summarize_episodes(all_episodes)
    summary["suites"] = suites
    summary["scorer"] = args.scorer
    summary["elapsed_seconds"] = round(time.time() - started, 3)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output_root / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "suite", "task_id", "task_name", "instruction", "init_state_id",
            "seed", "success", "status", "rounds", "qwen_request_count",
            "total_sim_steps", "elapsed_seconds", "error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for episode in all_episodes:
            writer.writerow(episode)

    print("\n=== Overall summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Results:", str(output_root.resolve()))


if __name__ == "__main__":
    main()
