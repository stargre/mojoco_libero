"""Unified symbolic / geometric belief state for LIBERO SayCan."""
import copy


class BeliefState:
    """Compact state consumed by candidate generation and the Qwen Say scorer.

    Geometry is populated from LIBERO simulator-side object-state wrappers.
    The source is retained explicitly so values are not confused with
    camera-only perception.
    """
    def __init__(self, instruction=""):
        self.instruction = str(instruction)
        self.objects = {}
        self.robot = {}
        self.held_object = None
        self.goal_status = []
        self.revision = 0
        self.last_failure = None

    def refresh(self, snapshot, goal_status):
        self.objects = copy.deepcopy(snapshot.get("objects", {}))
        self.robot = copy.deepcopy(snapshot.get("robot", {}))
        if "held_object" in snapshot:
            self.held_object = snapshot.get("held_object")
        self.goal_status = copy.deepcopy(goal_status or [])
        self.revision += 1

    def all_goals_satisfied(self):
        return bool(self.goal_status) and all(
            bool(item.get("satisfied")) for item in self.goal_status
        )

    def first_unsatisfied_goal(self):
        for item in self.goal_status:
            if not item.get("satisfied"):
                return item
        return None

    def summary(self):
        objects = {}
        for name, state in self.objects.items():
            objects[name] = {
                "kind": state.get("kind", "unknown"),
                "category": state.get("category", ""),
                "position": state.get("position"),
                "is_open": state.get("is_open"),
                "is_closed": state.get("is_closed"),
                "is_on": state.get("is_on"),
                "parent": state.get("parent"),
                "has_articulation": bool(state.get("has_articulation", False)),
            }
        return {
            "instruction": self.instruction,
            "objects": objects,
            "robot": copy.deepcopy(self.robot),
            "held_object": self.held_object,
            "goal_status": copy.deepcopy(self.goal_status),
            "revision": self.revision,
            "last_failure": copy.deepcopy(self.last_failure),
            "state_source": "LIBERO simulator object-state API",
        }

    def set_held(self, object_name):
        self.held_object = object_name

    def mark_failure(self, failure):
        self.last_failure = copy.deepcopy(failure)
