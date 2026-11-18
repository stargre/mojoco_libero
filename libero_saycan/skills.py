"""LIBERO skill library using robosuite OSC_POSE actions."""

import math
import numpy as np


class SkillLibrary:
    """Composable manipulation primitives executed through LIBERO's Panda OSC controller.

    All arm commands are normalized OSC_POSE deltas (5 cm translation and
    0.5 rad rotation per full-scale command in the pinned robosuite config).
    PandaGripper uses -1 for open and +1 for close.
    """

    def __init__(self, env, belief, on_step=None, max_motion_steps=100,
                 gripper_steps=110):
        self.env = env
        self.belief = belief
        self.on_step = on_step
        self.max_motion_steps = int(max_motion_steps)
        self.gripper_steps = int(gripper_steps)

    def _step(self, action):
        result = self.env.step(action)
        if self.on_step is not None:
            self.on_step(result.get("observation"))
        return result

    def _translation_scale(self):
        try:
            controller = self.env.domain.robots[0].controller
            values = np.asarray(controller.output_max[:3], dtype=float)
            if values.shape == (3,) and np.all(values > 1e-6):
                return values
        except Exception:
            pass
        return np.array([0.05, 0.05, 0.05], dtype=float)

    def _move_to(self, target, tolerance=0.018, max_steps=None):
        target = np.asarray(target, dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            return {"success": False, "status": "invalid_target_pose"}
        limit = self.max_motion_steps if max_steps is None else int(max_steps)
        scale = self._translation_scale()
        last_distance = float("inf")

        for step_index in range(limit):
            robot = self.env.get_robot_state()
            current = np.asarray(robot.get("eef_position", []), dtype=float)
            if current.shape != (3,):
                return {"success": False, "status": "eef_observation_missing"}
            delta = target - current
            last_distance = float(np.linalg.norm(delta))
            if last_distance <= tolerance:
                return {
                    "success": True,
                    "status": "target_reached",
                    "distance_m": last_distance,
                    "steps": step_index,
                }

            action = np.zeros(self.env.action_dim, dtype=np.float32)
            action[:3] = np.clip(delta / scale, -1.0, 1.0)
            action[3:6] = 0.0
            action[6] = 0.0
            result = self._step(action)
            if result.get("success"):
                return {
                    "success": True,
                    "status": "task_completed_during_motion",
                    "distance_m": last_distance,
                    "steps": step_index + 1,
                }

        robot = self.env.get_robot_state()
        current = np.asarray(robot.get("eef_position", []), dtype=float)
        if current.shape == (3,):
            last_distance = float(np.linalg.norm(target - current))
        return {
            "success": last_distance <= tolerance * 1.5,
            "status": "target_reached" if last_distance <= tolerance * 1.5 else "motion_timeout",
            "distance_m": last_distance,
            "steps": limit,
        }

    def _set_gripper(self, opened):
        command = -1.0 if opened else 1.0
        steps = max(0, self.gripper_steps)
        executed = 0
        for _ in range(steps):
            action = np.zeros(self.env.action_dim, dtype=np.float32)
            action[6] = command
            result = self._step(action)
            executed += 1
            if result.get("success"):
                break
        return {
            "success": executed > 0,
            "status": "gripper_opened" if opened else "gripper_closed",
            "steps": executed,
        }

    def _wait(self, steps=12):
        action = np.zeros(self.env.action_dim, dtype=np.float32)
        for _ in range(max(0, int(steps))):
            result = self._step(action)
            if result.get("success"):
                break

    def _predicate_satisfied(self, predicate_name, args):
        try:
            return bool(self.env.domain._eval_predicate(
                [predicate_name] + [str(value) for value in args]
            ))
        except Exception:
            return False

    def _pick(self, object_name):
        if self.belief.held_object is not None:
            return self._result("Pick", False, "gripper_not_empty", "The gripper already holds an object.")
        if object_name not in self.env.domain.objects_dict:
            return self._result("Pick", False, "not_a_movable_object", object_name)

        initial_position = self.env.get_object_position(object_name)
        pregrasp = initial_position + np.array([0.0, 0.0, 0.13])
        grasp = initial_position + np.array([0.0, 0.0, 0.005])

        approach = self._move_to(pregrasp)
        if not approach["success"]:
            return self._result("Pick", False, "pregrasp_unreachable", "Could not reach the pre-grasp pose.", motion=approach)

        descend = self._move_to(grasp, tolerance=0.022)
        if not descend["success"]:
            self._move_to(pregrasp)
            return self._result("Pick", False, "grasp_pose_unreachable", "Could not reach the grasp pose.", motion=descend)

        self._set_gripper(opened=False)
        self._wait(8)
        lift_target = grasp + np.array([0.0, 0.0, 0.16])
        lift = self._move_to(lift_target)
        self._wait(8)

        try:
            final_position = self.env.get_object_position(object_name)
            lift_delta = float(final_position[2] - initial_position[2])
        except Exception:
            final_position = None
            lift_delta = 0.0

        verified = lift["success"] and lift_delta >= 0.035
        if not verified:
            self._set_gripper(opened=True)
            self._move_to(pregrasp)
            return self._result(
                "Pick", False, "grasp_or_lift_not_verified",
                "The object did not rise sufficiently after closing the gripper.",
                lift_delta_m=lift_delta,
                final_position=None if final_position is None else final_position.tolist(),
                motion=lift,
            )

        self.belief.set_held(object_name)
        return self._result(
            "Pick", True, "pick_verified",
            "The simulator object pose confirms that the object was lifted.",
            object=object_name,
            lift_delta_m=lift_delta,
        )

    def _put(self, skill_name, object_name, target_name, relation):
        if self.belief.held_object != object_name:
            return self._result(skill_name, False, "object_not_held", object_name)
        try:
            target_position = self.env.compute_placement_position(
                object_name, target_name, relation
            )
        except Exception as exc:
            return self._result(
                skill_name, False, "placement_target_unavailable",
                "{}: {}".format(type(exc).__name__, str(exc)),
            )

        preplace = target_position + np.array([0.0, 0.0, 0.13])
        approach = self._move_to(preplace)
        if not approach["success"]:
            return self._result(skill_name, False, "preplace_unreachable", "Could not reach the pre-place pose.", motion=approach)

        descend = self._move_to(target_position, tolerance=0.022)
        if not descend["success"]:
            return self._result(skill_name, False, "place_pose_unreachable", "Could not reach the release pose.", motion=descend)

        self._set_gripper(opened=True)
        self.belief.set_held(None)
        self._wait(18)
        retreat = self._move_to(preplace)

        verified = self._predicate_satisfied(relation, [object_name, target_name])
        return self._result(
            skill_name, verified,
            "placement_verified" if verified else "placement_not_verified",
            "LIBERO BDDL predicate {}({}, {}) evaluated to {}.".format(
                relation, object_name, target_name, verified
            ),
            object=object_name,
            target=target_name,
            target_position=target_position.tolist(),
            retreat=retreat,
        )

    def _desired_joint_position(self, info, skill_name):
        if skill_name == "Open":
            ranges = info.get("open_range") or []
            if not ranges:
                ranges = info["joint_range"]
            return float(sum(ranges[:2]) / 2.0)
        if skill_name == "Close":
            ranges = info.get("close_range") or []
            if not ranges:
                ranges = [0.0, 0.0]
            return float(sum(ranges[:2]) / 2.0)
        if skill_name == "TurnOn":
            ranges = info.get("turnon_range") or info.get("open_range") or []
            if not ranges:
                ranges = info["joint_range"]
            return float(sum(ranges[:2]) / 2.0)
        ranges = info.get("turnoff_range") or info.get("close_range") or []
        if not ranges:
            ranges = [0.0, 0.0]
        return float(sum(ranges[:2]) / 2.0)

    def _articulate(self, skill_name, target_name):
        if self.belief.held_object is not None:
            return self._result(skill_name, False, "gripper_not_empty", "The gripper must be free to manipulate a handle.")
        try:
            info = self.env.get_articulation_info(target_name)
        except Exception as exc:
            return self._result(skill_name, False, "articulation_lookup_failed", str(exc))
        if not info:
            return self._result(skill_name, False, "articulation_not_found", target_name)

        target_qpos = self._desired_joint_position(info, skill_name)
        current_qpos = float(info["current_qpos"])
        if abs(target_qpos - current_qpos) < 0.015:
            return self._result(skill_name, True, "already_in_target_state", target_name)

        handle = np.asarray(info["handle_position"], dtype=float)
        pregrasp = handle + np.array([0.0, 0.0, 0.11])
        approach = self._move_to(pregrasp)
        if not approach["success"]:
            return self._result(skill_name, False, "handle_pregrasp_unreachable", "Could not approach the articulation handle.", motion=approach)

        grasp = self._move_to(handle, tolerance=0.025)
        if not grasp["success"]:
            return self._result(skill_name, False, "handle_unreachable", "Could not reach the articulation handle.", motion=grasp)

        self._set_gripper(opened=False)
        target_handle = self.env.target_handle_position(info, target_qpos)
        distance = float(np.linalg.norm(target_handle - handle))
        segments = max(3, min(24, int(math.ceil(distance / 0.025))))
        path_results = []
        for index in range(1, segments + 1):
            alpha = float(index) / float(segments)
            waypoint = handle + alpha * (target_handle - handle)
            motion = self._move_to(waypoint, tolerance=0.025, max_steps=35)
            path_results.append(motion)
            if not motion["success"]:
                break

        self._set_gripper(opened=True)
        self._wait(12)
        retreat = self._move_to(target_handle + np.array([0.0, 0.0, 0.11]))

        state = self.env.domain.object_states_dict.get(target_name)
        if state is None and target_name in self.env.domain.object_sites_dict:
            state = self.env.domain.object_states_dict.get(target_name)

        if skill_name == "Open":
            verified = bool(state.is_open()) if state is not None else False
        elif skill_name == "Close":
            verified = bool(state.is_close()) if state is not None else False
        elif skill_name == "TurnOn":
            try:
                verified = bool(state.turn_on()) if state is not None else False
            except Exception:
                verified = False
        else:
            try:
                verified = bool(state.turn_off()) if state is not None else False
            except Exception:
                verified = False

        return self._result(
            skill_name, verified,
            "articulation_verified" if verified else "articulation_not_verified",
            "The LIBERO articulation predicate was checked after physical handle motion.",
            target=target_name,
            joint_name=info["joint_name"],
            initial_qpos=current_qpos,
            target_qpos=target_qpos,
            handle_path_segments=len(path_results),
            path_results=path_results,
            retreat=retreat,
        )

    def _push(self, object_name, target_name):
        if self.belief.held_object is not None:
            return self._result("Push", False, "gripper_not_empty", "Cannot push while holding an object.")
        start = self.env.get_object_position(object_name)
        target = self.env.compute_placement_position(object_name, target_name, "on")
        direction = target[:2] - start[:2]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-5:
            return self._result("Push", False, "zero_push_distance", "Source and target are coincident.")
        direction = direction / norm
        contact = start + np.array([0.0, 0.0, 0.035])
        precontact = contact + np.array([0.0, 0.0, 0.10])
        approach = self._move_to(precontact)
        if not approach["success"]:
            return self._result("Push", False, "push_precontact_unreachable", "Could not reach push pose.", motion=approach)
        descend = self._move_to(contact, tolerance=0.025)
        if not descend["success"]:
            return self._result("Push", False, "push_contact_unreachable", "Could not reach contact pose.", motion=descend)
        push_target = contact.copy()
        push_target[:2] += direction * min(norm, 0.18)
        motion = self._move_to(push_target, tolerance=0.03, max_steps=120)
        self._move_to(push_target + np.array([0.0, 0.0, 0.12]))
        verified = self._predicate_satisfied("on", [object_name, target_name])
        return self._result(
            "Push", verified,
            "push_verified" if verified else "push_not_verified",
            "The requested On predicate was checked after pushing.",
            motion=motion,
        )

    def execute(self, call):
        name = call.get("name")
        args = call.get("args", {})
        if name == "Pick":
            return self._pick(args.get("object"))
        if name == "PutOn":
            return self._put("PutOn", args.get("object"), args.get("target"), "on")
        if name == "PutInside":
            return self._put("PutInside", args.get("object"), args.get("target"), "in")
        if name in ("Open", "Close", "TurnOn", "TurnOff"):
            return self._articulate(name, args.get("target"))
        if name == "Push":
            return self._push(args.get("object"), args.get("target"))
        if name == "Done":
            success = self.env.check_success()
            return self._result(
                "Done", success,
                "task_complete" if success else "task_not_complete",
                "Checked the official LIBERO BDDL success predicate.",
            )
        return self._result(name or "unknown", False, "unknown_skill", str(name))

    @staticmethod
    def _result(skill, success, status, reason, **extra):
        return {
            "skill": skill,
            "success": bool(success),
            "status": status,
            "reason": reason,
            **extra
        }
