"""High-level LIBERO SayCan agent composition."""

from libero_saycan.belief import BeliefState
from libero_saycan.can import CanEvaluator
from libero_saycan.planner import SayCanPlanner
from libero_saycan.skills import SkillLibrary


class LiberoSayCanAgent:
    def __init__(
        self,
        env,
        scorer,
        recorder=None,
        max_rounds=40,
        max_retries_per_skill=2,
        max_motion_steps=100,
        gripper_steps=110,
    ):
        self.env = env
        self.scorer = scorer
        self.recorder = recorder
        self.belief = BeliefState(env.instruction)
        on_step = recorder.capture if recorder is not None else None
        self.skills = SkillLibrary(
            env,
            self.belief,
            on_step=on_step,
            max_motion_steps=max_motion_steps,
            gripper_steps=gripper_steps,
        )
        self.planner = SayCanPlanner(
            env=env,
            scorer=scorer,
            can_evaluator=CanEvaluator(),
            skill_library=self.skills,
            max_rounds=max_rounds,
            max_retries_per_skill=max_retries_per_skill,
        )

    def run(self):
        if self.recorder is not None:
            self.recorder.capture(self.env.observation)
        result = self.planner.run(self.env.instruction, self.belief)
        if self.recorder is not None:
            self.recorder.capture(self.env.observation)
            result["visualization"] = self.recorder.finalize()
        result["run_metadata"] = {
            "suite": self.env.suite_name,
            "task_id": self.env.task_id,
            "task_name": self.env.task_name,
            "instruction": self.env.instruction,
            "scorer": getattr(self.scorer, "model", type(self.scorer).__name__),
            "qwen_model": getattr(self.scorer, "model", None),
            "qwen_base_url": getattr(self.scorer, "base_url", None),
            "qwen_request_count": int(getattr(self.scorer, "request_count", 0)),
            "qwen_last_latency_s": float(getattr(self.scorer, "last_latency_s", 0.0)),
            "total_sim_steps": int(self.env.total_steps),
            "rollout_steps": int(self.env.rollout_steps),
            "init_state_id": int(self.env.init_state_id),
            "seed": int(self.env.seed),
        }
        return result
