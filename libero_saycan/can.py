"""Rule-based Can affordance estimator for the LIBERO simulator adapter."""

import math
import numpy as np


class CanEvaluator:
    """Estimate skill feasibility from preconditions and simulator geometry.

    This is an explicit, deterministic engineering baseline. It is not a
    learned affordance/value function as used in the original SayCan paper.
    """
    def __init__(self, max_reach_m=0.88, min_object_z=0.78, max_object_z=1.85):
        self.max_reach_m = float(max_reach_m)
        self.min_object_z = float(min_object_z)
        self.max_object_z = float(max_object_z)

    def _reach_score(self, position, belief):
        if position is None:
            return 0.0
        point = np.asarray(position, dtype=float)
        eef = np.asarray(belief.robot.get("eef_position", []), dtype=float)
        if point.shape != (3,) or eef.shape != (3,):
            return 0.0
        if not np.all(np.isfinite(point)) or not np.all(np.isfinite(eef)):
            return 0.0
        if point[2] < self.min_object_z or point[2] > self.max_object_z:
            return 0.0
        distance = float(np.linalg.norm(point - eef))
        if distance > self.max_reach_m:
            return 0.0
        return max(0.15, min(1.0, 1.0 - distance / (self.max_reach_m * 1.35)))

    def evaluate(self, candidate, belief, env):
        name = candidate.get("name")
        args = candidate.get("args", {})
        if name == "Done":
            return 1.0 if env.check_success() else 0.0

        if name == "Pick":
            target = args.get("object")
            state = belief.objects.get(target)
            if state is None or state.get("kind") != "object":
                return 0.0
            if belief.held_object is not None:
                return 0.0
            return self._reach_score(state.get("position"), belief)

        if name in ("PutOn", "PutInside"):
            object_name = args.get("object")
            target = args.get("target")
            if belief.held_object != object_name:
                return 0.0
            target_state = belief.objects.get(target)
            if target_state is None:
                return 0.0
            parent = target_state.get("parent")
            parent_state = belief.objects.get(parent, {}) if parent else {}
            if name == "PutInside":
                if target_state.get("is_open") is False and target_state.get("has_articulation"):
                    return 0.0
                if parent_state.get("is_open") is False and parent_state.get("has_articulation"):
                    return 0.0
            return self._reach_score(target_state.get("position"), belief)

        if name in ("Open", "Close", "TurnOn", "TurnOff"):
            if belief.held_object is not None:
                return 0.0
            target = args.get("target")
            state = belief.objects.get(target)
            if state is None or not state.get("has_articulation"):
                return 0.0
            if name == "Open" and state.get("is_open") is True:
                return 0.0
            if name == "Close" and state.get("is_closed") is True:
                return 0.0
            if name == "TurnOn" and state.get("is_on") is True:
                return 0.0
            if name == "TurnOff" and state.get("is_on") is False:
                return 0.0
            try:
                info = env.get_articulation_info(target)
            except Exception:
                return 0.0
            if not info or info.get("handle_position") is None:
                return 0.0
            return self._reach_score(info.get("handle_position"), belief)

        if name == "Push":
            target = args.get("object")
            state = belief.objects.get(target)
            if state is None or state.get("kind") != "object":
                return 0.0
            if belief.held_object is not None:
                return 0.0
            return self._reach_score(state.get("position"), belief)

        return 0.0
