"""LIBERO task loader, BDDL bridge and simulator-state adapter."""

import os

import numpy as np

from libero_saycan.bootstrap import configure_libero
from libero_saycan.spec import TaskSpec


class LiberoTaskEnv:
    """Load one official LIBERO task with a reproducible fixed initial state.

    LIBERO uses robosuite's legacy four-value step API. This adapter keeps
    that API behind a small normalized interface and exposes BDDL predicates,
    object-state wrappers and articulation metadata for the SayCan layers.
    """

    def __init__(
        self,
        suite_name="libero_spatial",
        task_id=0,
        init_state_id=0,
        seed=0,
        camera_height=128,
        camera_width=128,
        horizon=600,
        settle_steps=10,
        render_gpu_device_id=0,
    ):
        self.bootstrap_info = configure_libero()
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        self.suite_name = str(suite_name).lower()
        self.task_id = int(task_id)
        self.init_state_id = int(init_state_id)
        self.seed = int(seed)
        self.settle_steps = int(settle_steps)
        self.rollout_steps = 0
        self.total_steps = 0
        self.closed = False
        self.observation = None
        self.last_info = {}
        self.last_done = False

        benchmark_dict = benchmark.get_benchmark_dict()
        if self.suite_name not in benchmark_dict:
            raise ValueError(
                "Unknown LIBERO suite {!r}. Available suites: {}".format(
                    self.suite_name, sorted(benchmark_dict.keys())
                )
            )

        self.task_suite = benchmark_dict[self.suite_name]()
        task_count = self.task_suite.get_num_tasks()
        if not 0 <= self.task_id < task_count:
            raise IndexError(
                "task_id {} is out of range for {} ({} tasks)".format(
                    self.task_id, self.suite_name, task_count
                )
            )

        self.task = self.task_suite.get_task(self.task_id)
        self.instruction = str(self.task.language)
        self.task_name = str(self.task.name)
        self.problem_folder = str(self.task.problem_folder)
        self.bddl_file = os.path.join(
            get_libero_path("bddl_files"),
            self.problem_folder,
            self.task.bddl_file,
        )
        if not os.path.isfile(self.bddl_file):
            raise FileNotFoundError(
                "LIBERO BDDL task file does not exist: {}".format(self.bddl_file)
            )

        self.init_states = self.task_suite.get_task_init_states(self.task_id)
        self.initial_state_count = len(self.init_states)
        if not 0 <= self.init_state_id < self.initial_state_count:
            raise IndexError(
                "init_state_id {} is out of range; {} initial states are available".format(
                    self.init_state_id, self.initial_state_count
                )
            )

        self.camera_height = int(camera_height)
        self.camera_width = int(camera_width)
        self.horizon = int(horizon)
        self.render_gpu_device_id = int(render_gpu_device_id)

        self.env = OffScreenRenderEnv(
            bddl_file_name=self.bddl_file,
            robots=["Panda"],
            controller="OSC_POSE",
            use_camera_obs=True,
            use_object_obs=True,
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=self.camera_height,
            camera_widths=self.camera_width,
            control_freq=20,
            horizon=self.horizon,
            render_gpu_device_id=self.render_gpu_device_id,
            renderer="mujoco",
        )
        self.domain = self.env.env
        self.parsed_problem = self.domain.parsed_problem
        self.spec = TaskSpec.from_parsed_problem(self.task, self.parsed_problem)
        self.action_dim = int(getattr(self.domain, "action_dim", 7))
        if self.action_dim != 7:
            self.close()
            raise RuntimeError(
                "Expected LIBERO Panda OSC_POSE action_dim=7, got {}".format(
                    self.action_dim
                )
            )

        self.reset(
            init_state_id=self.init_state_id,
            seed=self.seed,
            settle_steps=self.settle_steps,
        )

    @staticmethod
    def _raw_model_data(domain):
        model = domain.sim.model
        data = domain.sim.data
        return (
            getattr(model, "_model", model),
            getattr(data, "_data", data),
        )

    def _step_raw(self, action):
        if self.closed:
            raise RuntimeError("Cannot step a closed LIBERO environment")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.action_dim,):
            raise ValueError(
                "Expected action shape ({},), got {}".format(
                    self.action_dim, action.shape
                )
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("Action contains NaN or Inf")
        action = np.clip(action, -1.0, 1.0)
        observation, reward, done, info = self.env.step(action)
        self.observation = observation
        self.last_info = info if isinstance(info, dict) else {}
        self.last_done = bool(done)
        self.total_steps += 1
        return observation, float(reward), bool(done), self.last_info

    def reset(self, init_state_id=None, seed=None, settle_steps=None):
        if self.closed:
            raise RuntimeError("Cannot reset a closed LIBERO environment")
        if init_state_id is not None:
            self.init_state_id = int(init_state_id)
        if seed is not None:
            self.seed = int(seed)
        if settle_steps is not None:
            self.settle_steps = int(settle_steps)
        if not 0 <= self.init_state_id < self.initial_state_count:
            raise IndexError("Invalid initial-state index")

        self.env.seed(self.seed)
        self.env.reset()
        self.observation = self.env.set_init_state(
            self.init_states[self.init_state_id]
        )
        self.last_info = {}
        self.last_done = False
        self.rollout_steps = 0

        neutral_action = np.zeros(self.action_dim, dtype=np.float32)
        for _ in range(max(0, self.settle_steps)):
            if self.check_success():
                break
            self._step_raw(neutral_action)
        self.rollout_steps = 0
        return self.observation

    def step(self, action):
        observation, reward, done, info = self._step_raw(action)
        self.rollout_steps += 1
        success = self.check_success()
        return {
            "observation": observation,
            "reward": reward,
            "done": bool(done),
            "success": bool(success),
            "info": info,
            "rollout_step": self.rollout_steps,
            "total_sim_steps": self.total_steps,
        }

    def check_success(self):
        if self.closed:
            return False
        return bool(self.env.check_success())

    def get_goal_status(self):
        status = []
        for index, predicate in enumerate(self.spec.goal_predicates):
            raw = [predicate.name] + list(predicate.args)
            try:
                satisfied = bool(self.domain._eval_predicate(raw))
                error = None
            except Exception as exc:
                satisfied = False
                error = "{}: {}".format(type(exc).__name__, str(exc))
            status.append({
                "index": index,
                "predicate": predicate.to_dict(),
                "satisfied": satisfied,
                "evaluation_error": error,
            })
        return status

    def _safe_state_method(self, state, method_name):
        method = getattr(state, method_name, None)
        if method is None:
            return None
        try:
            return bool(method())
        except Exception:
            return None

    def _joint_names(self, name):
        if name in self.domain.object_sites_dict:
            values = getattr(self.domain.object_sites_dict[name], "joints", [])
        else:
            obj = self.domain.get_object(name)
            values = getattr(obj, "joints", []) if obj is not None else []
        return [str(value) for value in (values or [])]

    def _joint_values(self, name):
        values = []
        for joint_name in self._joint_names(name):
            try:
                address = self.domain.sim.model.get_joint_qpos_addr(joint_name)
                qpos = self.domain.sim.data.qpos[address]
                if np.isscalar(qpos):
                    values.append(float(qpos))
                else:
                    values.extend(np.asarray(qpos, dtype=float).reshape(-1).tolist())
            except Exception:
                continue
        return values

    def get_robot_state(self):
        observation = self.observation or {}
        result = {}
        for key, output in (
            ("robot0_eef_pos", "eef_position"),
            ("robot0_eef_quat", "eef_quaternion"),
            ("robot0_gripper_qpos", "gripper_qpos"),
            ("robot0_joint_pos", "joint_positions"),
        ):
            if key in observation:
                result[output] = np.asarray(observation[key], dtype=float).reshape(-1).tolist()
        return result

    def get_object_position(self, name):
        state = self.domain.object_states_dict.get(name)
        if state is None:
            raise KeyError("Unknown LIBERO object/site: {}".format(name))
        geom = state.get_geom_state()
        position = np.asarray(geom["pos"], dtype=float)
        if position.shape != (3,):
            raise ValueError("Invalid position for {}: {}".format(name, position))
        return position.copy()

    def _is_descendant_body(self, body_id, root_body_id, model):
        current = int(body_id)
        root = int(root_body_id)
        while current >= 0:
            if current == root:
                return True
            if current == 0:
                break
            current = int(model.body_parentid[current])
        return False

    def get_object_half_height(self, name):
        if name not in self.domain.obj_body_id:
            return 0.035
        try:
            import mujoco

            model, data = self._raw_model_data(self.domain)
            root_id = int(self.domain.obj_body_id[name])
            root_z = float(data.xpos[root_id][2])
            extents = []
            for geom_id in range(model.ngeom):
                body_id = int(model.geom_bodyid[geom_id])
                if not self._is_descendant_body(body_id, root_id, model):
                    continue
                geom_type = int(model.geom_type[geom_id])
                size = np.asarray(model.geom_size[geom_id], dtype=float)
                rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
                if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
                    half = (
                        abs(rotation[2, 0]) * size[0]
                        + abs(rotation[2, 1]) * size[1]
                        + abs(rotation[2, 2]) * size[2]
                    )
                elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                    half = float(size[0])
                elif geom_type in (
                    int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                    int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                ):
                    half = (
                        abs(rotation[2, 2]) * size[1]
                        + size[0] * np.sqrt(rotation[2, 0] ** 2 + rotation[2, 1] ** 2)
                    )
                else:
                    half = float(model.geom_rbound[geom_id])
                center_offset = abs(float(data.geom_xpos[geom_id][2]) - root_z)
                extents.append(center_offset + float(half))
            if extents:
                return float(np.clip(max(extents), 0.015, 0.18))
        except Exception:
            pass
        return 0.035

    def get_site_geometry(self, site_name):
        import mujoco

        model, data = self._raw_model_data(self.domain)
        site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, str(site_name)
        )
        if site_id < 0:
            raise KeyError("Unknown LIBERO site: {}".format(site_name))
        return {
            "position": np.asarray(data.site_xpos[site_id], dtype=float).copy(),
            "rotation": np.asarray(data.site_xmat[site_id], dtype=float).reshape(3, 3).copy(),
            "size": np.asarray(model.site_size[site_id], dtype=float).copy(),
        }

    def compute_placement_position(self, object_name, target_name, relation):
        object_half = self.get_object_half_height(object_name)
        if target_name in self.domain.object_sites_dict:
            site = self.get_site_geometry(target_name)
            position = site["position"].copy()
            if relation == "in":
                return position
            if float(site["size"][2]) <= 0.012:
                position[2] += object_half + 0.006
            else:
                position[2] += object_half + 0.006
            return position

        if target_name in self.domain.object_states_dict:
            target_position = self.get_object_position(target_name)
            target_half = self.get_object_half_height(target_name)
            return np.array([
                target_position[0],
                target_position[1],
                target_position[2] + target_half + object_half + 0.006,
            ], dtype=float)
        raise KeyError("Unknown placement target: {}".format(target_name))

    def _descendant_geoms(self, body_id, model):
        found = []
        for geom_id in range(model.ngeom):
            geom_body = int(model.geom_bodyid[geom_id])
            if self._is_descendant_body(geom_body, body_id, model):
                found.append(geom_id)
        return found

    def get_articulation_info(self, target_name):
        """Resolve an articulated BDDL fixture/site to a movable joint and handle."""
        import mujoco

        if target_name in self.domain.object_sites_dict:
            site_object = self.domain.object_sites_dict[target_name]
            parent_name = str(site_object.parent_name)
            joint_names = list(getattr(site_object, "joints", []) or [])
        else:
            parent_name = str(target_name)
            obj = self.domain.get_object(parent_name)
            joint_names = list(getattr(obj, "joints", []) or []) if obj is not None else []

        if not joint_names:
            return None

        model, data = self._raw_model_data(self.domain)
        selected = None
        for joint_name in joint_names:
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, str(joint_name)
            )
            if joint_id < 0:
                continue
            joint_type = int(model.jnt_type[joint_id])
            if joint_type in (
                int(mujoco.mjtJoint.mjJNT_SLIDE),
                int(mujoco.mjtJoint.mjJNT_HINGE),
            ):
                selected = (str(joint_name), joint_id)
                break
        if selected is None:
            return None

        joint_name, joint_id = selected
        body_id = int(model.jnt_bodyid[joint_id])
        geom_ids = self._descendant_geoms(body_id, model)
        named = []
        collidable = []
        for geom_id in geom_ids:
            geom_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            ) or ""
            lower_name = geom_name.lower()
            if any(word in lower_name for word in ("handle", "knob", "button")):
                named.append(geom_id)
            if int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id]):
                collidable.append(geom_id)

        choices = named or collidable or geom_ids
        handle_position = None
        if choices:
            # Prefer the smallest contactable geometry when the asset does not
            # expose an explicit handle/knob geom name.
            if not named:
                choices = sorted(
                    choices,
                    key=lambda gid: float(model.geom_rbound[gid]),
                )
            handle_position = np.asarray(
                data.geom_xpos[int(choices[0])], dtype=float
            ).copy()
        if handle_position is None:
            handle_position = np.asarray(data.xpos[body_id], dtype=float).copy()

        parent_body = int(model.body_parentid[body_id])
        parent_rotation = np.asarray(data.xmat[parent_body], dtype=float).reshape(3, 3)
        axis_world = parent_rotation.dot(np.asarray(model.jnt_axis[joint_id], dtype=float))
        axis_norm = float(np.linalg.norm(axis_world))
        if axis_norm <= 1e-9:
            return None
        axis_world = axis_world / axis_norm

        body_rotation = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
        anchor = (
            np.asarray(data.xpos[body_id], dtype=float)
            + body_rotation.dot(np.asarray(model.jnt_pos[joint_id], dtype=float))
        )
        qpos_address = int(model.jnt_qposadr[joint_id])
        current_qpos = float(data.qpos[qpos_address])
        joint_range = np.asarray(model.jnt_range[joint_id], dtype=float).copy()

        parent_object = self.domain.get_object(parent_name)
        properties = getattr(parent_object, "object_properties", {}) or {}
        articulation = properties.get("articulation", {}) or {}
        return {
            "target": str(target_name),
            "parent": parent_name,
            "joint_name": joint_name,
            "joint_id": int(joint_id),
            "joint_type": int(model.jnt_type[joint_id]),
            "joint_kind": "slide" if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_SLIDE) else "hinge",
            "body_id": body_id,
            "qpos_address": qpos_address,
            "current_qpos": current_qpos,
            "joint_range": joint_range.tolist(),
            "axis_world": axis_world,
            "anchor_world": anchor,
            "handle_position": handle_position,
            "open_range": list(articulation.get("default_open_ranges", []) or []),
            "close_range": list(articulation.get("default_close_ranges", []) or []),
            "turnon_range": list(articulation.get("default_turnon_ranges", []) or []),
            "turnoff_range": list(articulation.get("default_turnoff_ranges", []) or []),
        }

    @staticmethod
    def target_handle_position(info, target_qpos):
        delta = float(target_qpos) - float(info["current_qpos"])
        handle = np.asarray(info["handle_position"], dtype=float)
        axis = np.asarray(info["axis_world"], dtype=float)
        if info.get("joint_kind") == "slide":
            return handle + axis * delta
        # MuJoCo hinge joint: rotate the handle around the joint anchor.
        anchor = np.asarray(info["anchor_world"], dtype=float)
        vector = handle - anchor
        angle = delta
        return (
            anchor
            + vector * np.cos(angle)
            + np.cross(axis, vector) * np.sin(angle)
            + axis * np.dot(axis, vector) * (1.0 - np.cos(angle))
        )

    def get_state_snapshot(self):
        objects = {}
        for name, state in self.domain.object_states_dict.items():
            try:
                geometry = state.get_geom_state()
                position = np.asarray(geometry["pos"], dtype=float).reshape(3).tolist()
                quaternion = np.asarray(geometry["quat"], dtype=float).reshape(4).tolist()
            except Exception:
                position = None
                quaternion = None

            if name in self.domain.object_sites_dict:
                kind = "region"
                parent = str(self.domain.object_sites_dict[name].parent_name)
            elif name in self.spec.object_categories:
                kind = "object"
                parent = None
            elif name in self.spec.fixture_categories:
                kind = "fixture"
                parent = None
            else:
                kind = "unknown"
                parent = None

            joint_names = self._joint_names(name)
            articulated = False
            if joint_names:
                try:
                    import mujoco
                    model, _ = self._raw_model_data(self.domain)
                    for joint_name in joint_names:
                        jid = mujoco.mj_name2id(
                            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                        )
                        if jid >= 0 and int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_FREE):
                            articulated = True
                            break
                except Exception:
                    articulated = True

            is_on = None
            if kind == "fixture":
                obj = self.domain.get_object(name)
                if obj is not None and hasattr(obj, "turn_on"):
                    try:
                        is_on = bool(state.turn_on())
                    except Exception:
                        is_on = None

            objects[str(name)] = {
                "kind": kind,
                "category": self.spec.category(name),
                "position": position,
                "quaternion": quaternion,
                "parent": parent,
                "joint_names": joint_names,
                "joint_values": self._joint_values(name),
                "has_articulation": bool(articulated),
                "is_open": self._safe_state_method(state, "is_open"),
                "is_closed": self._safe_state_method(state, "is_close"),
                "is_on": is_on,
                "source": "simulator_ground_truth",
            }

        return {
            "objects": objects,
            "robot": self.get_robot_state(),
            "held_object": None,
        }

    @property
    def metadata(self):
        return {
            "suite": self.suite_name,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "instruction": self.instruction,
            "problem_folder": self.problem_folder,
            "bddl_file": self.bddl_file,
            "init_state_id": self.init_state_id,
            "initial_state_count": self.initial_state_count,
            "seed": self.seed,
            "camera_height": self.camera_height,
            "camera_width": self.camera_width,
            "horizon": self.horizon,
            "settle_steps": self.settle_steps,
            "action_dim": self.action_dim,
            "mujoco_gl": os.environ.get("MUJOCO_GL"),
            "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
            "bddl_goal_predicates": [
                item.to_dict() for item in self.spec.goal_predicates
            ],
        }

    def close(self):
        if not self.closed:
            try:
                self.env.close()
            finally:
                self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
