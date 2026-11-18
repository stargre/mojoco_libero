"""BDDL-grounded candidate generation and closed-loop SayCan replanning."""

import copy


class CandidateGenerator:
    def __init__(self, max_goal_lookahead=4):
        self.max_goal_lookahead = int(max_goal_lookahead)

    @staticmethod
    def _candidate(name, args, goal_index, predicate):
        payload = {
            "name": name,
            "args": dict(args),
            "goal_index": int(goal_index),
            "goal_predicate": predicate.to_dict(),
        }
        payload["id"] = "{}:{}:{}".format(
            name, goal_index,
            "|".join(str(value) for value in payload["args"].values()),
        )
        return payload

    def generate(self, spec, belief):
        candidates = []
        seen = set()
        unsatisfied = [
            item for item in belief.goal_status if not item.get("satisfied")
        ][:self.max_goal_lookahead]

        for item in unsatisfied:
            index = int(item["index"])
            predicate = item["predicate"]
            name = predicate.get("name", "")
            args = list(predicate.get("args", []))
            generated = []

            if name in ("on", "in") and len(args) >= 2:
                object_name, target_name = args[0], args[1]
                target_state = belief.objects.get(target_name, {})
                parent_name = target_state.get("parent")
                parent_state = belief.objects.get(parent_name, {}) if parent_name else {}

                # Open an articulated container before picking the object, so
                # the gripper remains free to manipulate its handle.
                closed_target = (
                    target_state.get("has_articulation")
                    and target_state.get("is_open") is False
                )
                closed_parent = (
                    parent_state.get("has_articulation")
                    and parent_state.get("is_open") is False
                )
                if name == "in" and (closed_target or closed_parent) and belief.held_object is None:
                    open_target = target_name if closed_target else parent_name
                    generated.append(("Open", {"target": open_target}))
                elif belief.held_object == object_name:
                    skill = "PutInside" if name == "in" else "PutOn"
                    generated.append((
                        skill,
                        {"object": object_name, "target": target_name, "relation": name},
                    ))
                elif belief.held_object is None:
                    generated.append(("Pick", {"object": object_name}))
                    category = spec.category(object_name).lower()
                    instruction = spec.instruction.lower()
                    if name == "on" and ("plate" in category or "push" in instruction):
                        generated.append(("Push", {"object": object_name, "target": target_name}))

            elif name in ("open", "close", "turnon", "turnoff") and args:
                skill = {
                    "open": "Open",
                    "close": "Close",
                    "turnon": "TurnOn",
                    "turnoff": "TurnOff",
                }[name]
                generated.append((skill, {"target": args[0]}))

            for skill, skill_args in generated:
                candidate = self._candidate(skill, skill_args, index, predicate)
                signature = (
                    candidate["name"],
                    tuple(sorted(candidate["args"].items())),
                )
                if signature not in seen:
                    seen.add(signature)
                    candidates.append(candidate)

        candidates.append({
            "id": "Done",
            "name": "Done",
            "args": {},
            "goal_index": -1,
            "goal_predicate": None,
        })
        return candidates


class SayCanPlanner:
    def __init__(self, env, scorer, can_evaluator, skill_library,
                 max_rounds=40, max_retries_per_skill=2):
        self.env = env
        self.scorer = scorer
        self.can = can_evaluator
        self.skills = skill_library
        self.max_rounds = int(max_rounds)
        self.max_retries_per_skill = int(max_retries_per_skill)
        self.generator = CandidateGenerator()
        self.history = []
        self.failure_counts = {}

    @staticmethod
    def _signature(candidate):
        return (
            candidate["name"],
            tuple(sorted(candidate.get("args", {}).items())),
        )

    def _refresh(self, belief):
        belief.refresh(self.env.get_state_snapshot(), self.env.get_goal_status())
        return belief

    def run(self, instruction, belief):
        for round_index in range(self.max_rounds):
            self._refresh(belief)
            if self.env.check_success():
                return self._result(True, "task_complete", belief)

            candidates = self.generator.generate(self.env.spec, belief)
            say_scores = self.scorer.score(instruction, candidates, belief)
            scored = []
            candidate_log = []

            for candidate in candidates:
                say_data = say_scores[candidate["id"]]
                say = float(say_data["relevance"])
                can = float(self.can.evaluate(candidate, belief, self.env))
                score = say * can
                signature = self._signature(candidate)
                retries = self.failure_counts.get(signature, 0)
                if retries >= self.max_retries_per_skill:
                    score = 0.0
                log_item = {
                    "id": candidate["id"],
                    "name": candidate["name"],
                    "args": copy.deepcopy(candidate.get("args", {})),
                    "goal_index": candidate.get("goal_index", -1),
                    "say": say,
                    "can": can,
                    "score": score,
                    "retries": retries,
                    "rationale": say_data.get("rationale", ""),
                }
                candidate_log.append(log_item)
                scored.append((score, say, can, candidate))

            scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
            if not scored or scored[0][2] <= 0.0 or scored[0][0] <= 0.0:
                self.history.append({
                    "round": round_index + 1,
                    "belief_before": belief.summary(),
                    "candidate_scores": candidate_log,
                    "selected": None,
                    "status": "no_feasible_candidate",
                })
                return self._result(False, "no_feasible_candidate", belief)

            score, say, can, selected = scored[0]
            if selected["name"] == "Done":
                if self.env.check_success():
                    return self._result(True, "task_complete", belief)
                signature = self._signature(selected)
                self.failure_counts[signature] = self.failure_counts.get(signature, 0) + 1
                continue

            belief_before = belief.summary()
            execution = self.skills.execute(selected)
            self._refresh(belief)
            self.history.append({
                "round": round_index + 1,
                "belief_before": belief_before,
                "candidate_scores": candidate_log,
                "selected": copy.deepcopy(selected),
                "say": say,
                "can": can,
                "combined_score": score,
                "execution": copy.deepcopy(execution),
                "belief_after": belief.summary(),
            })

            signature = self._signature(selected)
            if execution.get("success"):
                self.failure_counts.pop(signature, None)
                if self.env.check_success():
                    return self._result(True, "task_complete", belief)
            else:
                self.failure_counts[signature] = self.failure_counts.get(signature, 0) + 1
                belief.mark_failure({
                    "skill": selected["name"],
                    "args": selected.get("args", {}),
                    "status": execution.get("status"),
                    "reason": execution.get("reason"),
                })
                # A failed action does not terminate the episode. The next
                # iteration refreshes the simulator state and replans.

        self._refresh(belief)
        if self.env.check_success():
            return self._result(True, "task_complete", belief)
        return self._result(False, "max_rounds", belief)

    def _result(self, success, status, belief):
        return {
            "success": bool(success),
            "status": status,
            "rounds": len(self.history),
            "history": self.history,
            "final_belief": belief.summary(),
            "qwen_request_count": int(getattr(self.scorer, "request_count", 0)),
            "qwen_last_latency_s": float(getattr(self.scorer, "last_latency_s", 0.0)),
        }
