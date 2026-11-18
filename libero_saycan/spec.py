"""BDDL task specification extraction using LIBERO's official parser output."""
from dataclasses import dataclass

SUPPORTED_PREDICATES = {"on", "in", "open", "close", "turnon", "turnoff"}

@dataclass(frozen=True)
class Predicate:
    name: str
    args: tuple

    def to_dict(self):
        return {"name": self.name, "args": list(self.args)}

    @property
    def key(self):
        return "{}({})".format(self.name, ",".join(self.args))


def flatten_predicates(tree):
    """Flatten BDDL parser output into atomic predicate records."""
    found = []

    def visit(node):
        if not isinstance(node, (list, tuple)) or not node:
            return
        head = node[0]
        if isinstance(head, str) and head.lower() in SUPPORTED_PREDICATES:
            found.append(Predicate(head.lower(), tuple(str(x) for x in node[1:])))
            return
        for child in node:
            visit(child)

    visit(tree)
    return found


class TaskSpec:
    def __init__(self, task_name, instruction, objects_by_type, fixtures_by_type,
                 regions, initial_predicates, goal_predicates, objects_of_interest,
                 problem_name=""):
        self.task_name = str(task_name)
        self.instruction = str(instruction)
        self.objects_by_type = {str(k): list(v) for k, v in (objects_by_type or {}).items()}
        self.fixtures_by_type = {str(k): list(v) for k, v in (fixtures_by_type or {}).items()}
        self.regions = dict(regions or {})
        self.initial_predicates = list(initial_predicates or [])
        self.goal_predicates = list(goal_predicates or [])
        self.objects_of_interest = list(objects_of_interest or [])
        self.problem_name = str(problem_name)
        self.object_categories = {}
        self.fixture_categories = {}
        for category, names in self.objects_by_type.items():
            for name in names:
                self.object_categories[str(name)] = category
        for category, names in self.fixtures_by_type.items():
            for name in names:
                self.fixture_categories[str(name)] = category
        self.region_targets = {
            str(name): str(value.get("target", ""))
            for name, value in self.regions.items()
        }

    @classmethod
    def from_parsed_problem(cls, task, parsed_problem):
        instruction = getattr(task, "language", None) or parsed_problem.get("language_instruction", "")
        return cls(
            task_name=getattr(task, "name", ""),
            instruction=instruction,
            objects_by_type=parsed_problem.get("objects", {}),
            fixtures_by_type=parsed_problem.get("fixtures", {}),
            regions=parsed_problem.get("regions", {}),
            initial_predicates=flatten_predicates(parsed_problem.get("initial_state", [])),
            goal_predicates=flatten_predicates(parsed_problem.get("goal_state", [])),
            objects_of_interest=parsed_problem.get("obj_of_interest", []),
            problem_name=parsed_problem.get("problem_name", ""),
        )

    def entity_kind(self, name):
        name = str(name)
        if name in self.object_categories:
            return "object"
        if name in self.fixture_categories:
            return "fixture"
        if name in self.regions:
            return "region"
        return "unknown"

    def category(self, name):
        name = str(name)
        return self.object_categories.get(name) or self.fixture_categories.get(name) or ""

    def region_parent(self, name):
        return self.region_targets.get(str(name), "")

    def to_dict(self):
        return {
            "task_name": self.task_name,
            "instruction": self.instruction,
            "problem_name": self.problem_name,
            "objects_by_type": self.objects_by_type,
            "fixtures_by_type": self.fixtures_by_type,
            "regions": self.regions,
            "initial_predicates": [item.to_dict() for item in self.initial_predicates],
            "goal_predicates": [item.to_dict() for item in self.goal_predicates],
            "objects_of_interest": self.objects_of_interest,
        }
