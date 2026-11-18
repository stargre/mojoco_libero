"""Thin adapter around the official LIBERO benchmark and OffScreenRenderEnv."""

import os

import numpy as np

from libero_saycan.bootstrap import configure_libero


class LiberoTaskEnv:
    """Load one official LIBERO task with a reproducible fixed initial state.

    The official LIBERO environment uses the robosuite legacy four-value step
    API: observation, reward, done, info. This adapter normalizes the result
    and exposes the benchmark's explicit check_success() postcondition.
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
        render_gpu_device_id=-1,
    ):
        self.bootstrap_info = configure_libero()

        # Import only after configure_libero has written config.yaml.
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
                "init_state_id {} is out of range; {} initial states are available"
                .format(self.init_state_id, self.initial_state_count)
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

        self.action_dim = int(getattr(self.env.env, "action_dim", 7))
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

    def _step_raw(self, action):
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
            self._step_raw(neutral_action)

        self.rollout_steps = 0
        return self.observation

    def step(self, action):
        if self.closed:
            raise RuntimeError("Cannot step a closed LIBERO environment")
        observation, reward, done, info = self._step_raw(action)
        self.rollout_steps += 1
        success = self.check_success()
        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "success": success,
            "info": info,
            "rollout_step": self.rollout_steps,
            "total_sim_steps": self.total_steps,
        }

    def check_success(self):
        if self.closed:
            return False
        return bool(self.env.check_success())

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
