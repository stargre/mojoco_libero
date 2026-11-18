"""Run one BDDL-grounded Qwen-SayCan episode in official LIBERO."""

import argparse
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libero_saycan.agent import LiberoSayCanAgent
from libero_saycan.env import LiberoTaskEnv
from libero_saycan.qwen import KeywordRelevanceScorer, QwenRelevanceScorer
from libero_saycan.recorder import TrajectoryRecorder


def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen + SayCan on one official LIBERO task."
    )
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scorer", choices=["qwen", "keyword"], default="qwen")
    parser.add_argument("--max-rounds", type=int, default=40)
    parser.add_argument("--max-motion-steps", type=int, default=100)
    parser.add_argument("--gripper-steps", type=int, default=110)
    parser.add_argument("--render-gpu-device-id", type=int, default=0)
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--record-stride", type=int, default=8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output = (
        Path(args.output)
        if args.output
        else PROJECT_ROOT / "outputs" / "saycan" /
        "{}_task{}_init{}_{}".format(
            args.suite, args.task_id, args.init_state_id, args.scorer
        )
    )
    output.mkdir(parents=True, exist_ok=True)

    env = LiberoTaskEnv(
        suite_name=args.suite,
        task_id=args.task_id,
        init_state_id=args.init_state_id,
        seed=args.seed,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        horizon=args.horizon,
        render_gpu_device_id=args.render_gpu_device_id,
    )
    recorder = None if args.no_record else TrajectoryRecorder(
        output, fps=args.fps, stride=args.record_stride
    )
    scorer = (
        QwenRelevanceScorer()
        if args.scorer == "qwen"
        else KeywordRelevanceScorer()
    )

    started = time.time()
    try:
        agent = LiberoSayCanAgent(
            env=env,
            scorer=scorer,
            recorder=recorder,
            max_rounds=args.max_rounds,
            max_motion_steps=args.max_motion_steps,
            gripper_steps=args.gripper_steps,
        )
        result = agent.run()
        result["elapsed_seconds"] = round(time.time() - started, 3)
        result["environment_metadata"] = env.metadata
        (output / "task_spec.json").write_text(
            json.dumps(env.spec.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("Artifacts:", str(output.resolve()))
        if not result["success"]:
            raise SystemExit(1)
    finally:
        env.close()


if __name__ == "__main__":
    main()
