import unittest
from types import SimpleNamespace

from libero_saycan.spec import TaskSpec, flatten_predicates


class TaskSpecTests(unittest.TestCase):
    def test_flatten_nested_goal_conjunction(self):
        tree = [
            "and",
            ["on", "bowl_1", "plate_1"],
            ["open", "cabinet_1_middle_region"],
            ["turnon", "stove_1"],
        ]
        predicates = flatten_predicates(tree)
        self.assertEqual(
            [(item.name, item.args) for item in predicates],
            [
                ("on", ("bowl_1", "plate_1")),
                ("open", ("cabinet_1_middle_region",)),
                ("turnon", ("stove_1",)),
            ],
        )

    def test_task_spec_extracts_categories_and_regions(self):
        task = SimpleNamespace(name="test_task", language="put the bowl on the plate")
        parsed = {
            "problem_name": "libero_tabletop_manipulation",
            "objects": {"bowl": ["bowl_1"]},
            "fixtures": {"table": ["main_table"]},
            "regions": {
                "main_table_plate_region": {"target": "main_table", "ranges": []}
            },
            "initial_state": [["on", "bowl_1", "main_table_plate_region"]],
            "goal_state": [["on", "bowl_1", "main_table_plate_region"]],
            "obj_of_interest": ["bowl_1"],
        }
        spec = TaskSpec.from_parsed_problem(task, parsed)
        self.assertEqual(spec.entity_kind("bowl_1"), "object")
        self.assertEqual(spec.entity_kind("main_table"), "fixture")
        self.assertEqual(spec.entity_kind("main_table_plate_region"), "region")
        self.assertEqual(spec.region_parent("main_table_plate_region"), "main_table")
        self.assertEqual(spec.goal_predicates[0].name, "on")


if __name__ == "__main__":
    unittest.main()
