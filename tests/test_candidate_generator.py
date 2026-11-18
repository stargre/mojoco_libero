import unittest

from libero_saycan.belief import BeliefState
from libero_saycan.planner import CandidateGenerator
from libero_saycan.spec import Predicate, TaskSpec


class CandidateGeneratorTests(unittest.TestCase):
    def make_spec(self, predicate, instruction=""):
        return TaskSpec(
            task_name="test",
            instruction=instruction,
            objects_by_type={"bowl": ["bowl_1"], "plate": ["plate_1"]},
            fixtures_by_type={"cabinet": ["cabinet_1"]},
            regions={},
            initial_predicates=[],
            goal_predicates=[predicate],
            objects_of_interest=["bowl_1"],
        )

    def make_belief(self, predicate, held=None, objects=None):
        belief = BeliefState("test")
        belief.held_object = held
        belief.objects = objects or {}
        belief.goal_status = [{
            "index": 0,
            "predicate": predicate.to_dict(),
            "satisfied": False,
        }]
        return belief

    def names(self, candidates):
        return [item["name"] for item in candidates]

    def test_on_goal_generates_pick_when_gripper_empty(self):
        predicate = Predicate("on", ("bowl_1", "plate_1"))
        spec = self.make_spec(predicate)
        belief = self.make_belief(predicate)
        candidates = CandidateGenerator().generate(spec, belief)
        self.assertEqual(self.names(candidates), ["Pick", "Done"])
        self.assertEqual(candidates[0]["args"]["object"], "bowl_1")

    def test_on_goal_generates_put_when_object_is_held(self):
        predicate = Predicate("on", ("bowl_1", "plate_1"))
        spec = self.make_spec(predicate)
        belief = self.make_belief(predicate, held="bowl_1")
        candidates = CandidateGenerator().generate(spec, belief)
        self.assertEqual(self.names(candidates), ["PutOn", "Done"])

    def test_in_goal_opens_closed_parent_before_pick(self):
        predicate = Predicate("in", ("bowl_1", "microwave_1_heating_region"))
        spec = self.make_spec(predicate)
        belief = self.make_belief(
            predicate,
            objects={
                "microwave_1_heating_region": {
                    "parent": "microwave_1",
                    "has_articulation": False,
                },
                "microwave_1": {
                    "has_articulation": True,
                    "is_open": False,
                },
            },
        )
        candidates = CandidateGenerator().generate(spec, belief)
        self.assertEqual(self.names(candidates), ["Open", "Done"])
        self.assertEqual(candidates[0]["args"]["target"], "microwave_1")

    def test_plate_goal_can_offer_push_alternative(self):
        predicate = Predicate("on", ("plate_1", "main_table_stove_front_region"))
        spec = self.make_spec(predicate, instruction="push the plate to the front of the stove")
        belief = self.make_belief(predicate)
        candidates = CandidateGenerator().generate(spec, belief)
        self.assertIn("Pick", self.names(candidates))
        self.assertIn("Push", self.names(candidates))


if __name__ == "__main__":
    unittest.main()
