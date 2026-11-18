import unittest

from libero_saycan.belief import BeliefState
from libero_saycan.planner import SayCanPlanner
from libero_saycan.spec import Predicate, TaskSpec


class FakeEnv:
    def __init__(self):
        self.success = False
        self.spec = TaskSpec(
            task_name="fake",
            instruction="pick the bowl",
            objects_by_type={"bowl": ["bowl_1"]},
            fixtures_by_type={},
            regions={},
            initial_predicates=[],
            goal_predicates=[Predicate("on", ("bowl_1", "plate_1"))],
            objects_of_interest=["bowl_1"],
        )
        self.instruction = "put the bowl on the plate"

    def get_state_snapshot(self):
        return {"objects": {}, "robot": {}, "held_object": None}

    def get_goal_status(self):
        return [{
            "index": 0,
            "predicate": {"name": "on", "args": ["bowl_1", "plate_1"]},
            "satisfied": self.success,
        }]

    def check_success(self):
        return self.success


class FakeScorer:
    request_count = 0
    last_latency_s = 0.0

    def score(self, task, candidates, belief):
        self.request_count += 1
        return {
            item["id"]: {"relevance": 0.9, "rationale": "fake"}
            for item in candidates
        }


class FakeCan:
    def evaluate(self, candidate, belief, env):
        return 0.0 if candidate["name"] == "Done" and not env.success else 1.0


class FailingThenSuccessfulSkills:
    def __init__(self, env):
        self.env = env
        self.calls = 0

    def execute(self, candidate):
        self.calls += 1
        if self.calls == 1:
            return {"success": False, "status": "temporary_failure", "reason": "test"}
        self.env.success = True
        return {"success": True, "status": "completed", "reason": "test"}


class ReplanningTests(unittest.TestCase):
    def test_failed_skill_triggers_replan_instead_of_terminating(self):
        env = FakeEnv()
        skills = FailingThenSuccessfulSkills(env)
        planner = SayCanPlanner(
            env=env,
            scorer=FakeScorer(),
            can_evaluator=FakeCan(),
            skill_library=skills,
            max_rounds=4,
            max_retries_per_skill=2,
        )
        result = planner.run(env.instruction, BeliefState(env.instruction))
        self.assertTrue(result["success"])
        self.assertEqual(skills.calls, 2)
        self.assertEqual(result["rounds"], 2)


if __name__ == "__main__":
    unittest.main()
