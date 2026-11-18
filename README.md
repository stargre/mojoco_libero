# MuJoCo LIBERO SayCan

将 LIBERO（Lifelong Robot Learning Benchmark）接入独立的 MuJoCo SayCan 实验工程。项目以官方 LIBERO 环境为仿真后端，逐步接入 Qwen Say、Can affordance 评估和可执行技能库。

> 当前里程碑：Stage 1 — LIBERO 环境接入与可复现 smoke test。当前代码能够加载官方任务、固定初始状态、执行 MuJoCo step、检查任务成功条件并保存图像/GIF/JSON。**当前 smoke test 使用 no-op action，不代表已经实现 Qwen SayCan 任务求解。**

## 1. 为什么使用独立环境

官方 LIBERO README 推荐 Python 3.8.13，并固定了较老版本的 robosuite、Gym、NumPy 和 PyTorch。为了避免破坏原来的 MuJoCo 环境，也避免和 mujoco_saycan 的 Python 3.11 环境发生依赖冲突，本项目使用独立 Conda 环境 libero-saycan。

当前参考依赖：

| Component | Version |
|---|---:|
| Python | 3.8.13 |
| LIBERO | commit 8f1084e3132a |
| robosuite | 1.4.0 |
| MuJoCo | 2.3.7 |
| NumPy | 1.22.4 |
| Gym | 0.25.2 |
| PyTorch | 1.11.0 + CUDA 11.3 |

MuJoCo 2.3.7 是本项目为 robosuite 1.4.0 增加的兼容性 pin；如果升级 MuJoCo 或 robosuite，需要重新执行渲染与环境 smoke test。

## 2. 安装

在 Linux 服务器执行：

~~~bash
conda env create -f environment.yml
conda activate libero-saycan

cd /mnt/data2/zhiyu/embodied-lab/mojoco_libero
bash scripts/install_libero.sh
~~~

安装脚本会把官方 LIBERO 固定版本 clone 到 third_party/LIBERO（该目录已加入 .gitignore），安装官方 requirements、MuJoCo/PyTorch 兼容版本，并以 editable 模式安装 LIBERO 和本项目。

如果服务器没有 EGL/OpenGL 开发运行库，需要先由管理员安装系统依赖。运行前设置：

~~~bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
~~~

先运行与 LIBERO 无关的工具单测，再检查安装：

~~~bash
python -m unittest discover -s tests -v
python scripts/check_install.py
~~~

## 3. 查看任务

LIBERO 提供 Spatial、Object、Goal，以及 LIBERO-100 拆分出的 LIBERO-90 / LIBERO-10。官方 benchmark 注册表中的 suite 名称为小写：

~~~bash
python scripts/list_tasks.py --suite libero_spatial
python scripts/list_tasks.py --suite libero_object
python scripts/list_tasks.py --suite libero_goal
python scripts/list_tasks.py --suite libero_10
python scripts/list_tasks.py --suite libero_90
~~~

列出全部 suite：

~~~bash
python scripts/list_tasks.py --all
~~~

## 4. 第一个环境测试

先运行一个最小任务：加载 libero_spatial 的第 0 个任务，使用官方固定初始状态 0，执行 20 步零动作，并保存相机画面。

~~~bash
python scripts/smoke_libero.py \
  --suite libero_spatial \
  --task-id 0 \
  --init-state-id 0 \
  --steps 20 \
  --render-gpu-device-id 0
~~~

输出目录默认为：

~~~text
outputs/smoke/libero_spatial_task0_init0/
├── result.json
├── trajectory.gif
├── initial_agentview.png
└── final_agentview.png
~~~

注意：零动作不会主动完成语言任务。因此 episode_success=false 是正常的；本阶段的验收标准是环境成功加载、固定初始状态成功恢复、相机图像可读、MuJoCo step 正常执行、check_success() 可调用，以及结果文件成功保存。

## 5. 代码结构

~~~text
mojoco_libero/
├── libero_saycan/
│   ├── __init__.py
│   ├── bootstrap.py       # LIBERO 路径配置与 EGL 初始化
│   ├── env.py             # LIBERO task / fixed-init-state 环境适配器
│   ├── observations.py    # 相机帧标准化
│   └── metrics.py         # episode / suite 成功率统计
├── scripts/
│   ├── install_libero.sh
│   ├── check_install.py
│   ├── list_tasks.py
│   └── smoke_libero.py
├── tests/
│   ├── test_metrics.py
│   └── test_observations.py
├── environment.yml
├── pyproject.toml
└── README.md
~~~

## 6. SayCan architecture and current implementation

This repository contains the Stage 2–6 implementation layers. It is an engineering baseline built on LIBERO simulator ground truth and deterministic motion primitives; it is not a reproduction of the original paper's learned affordance value functions, nor a trained visuomotor policy.

### Stage 2 — BDDL task semantics and BeliefState

- Reads LIBERO's parsed BDDL problem: movable object categories, fixture categories, named regions, initial predicates, goal predicates and objects of interest.
- Normalizes supported predicates (currently `on`, `in`, `open`, `close`, `turnon`, `turnoff`) into typed records.
- Builds a unified `BeliefState` with object/site positions, articulation state, robot end-effector state, held-object estimate, goal predicate status, revision counter and last failure.
- The current state snapshot is simulator-side ground truth. It is not camera-only perception; do not describe it as a learned visual belief model.

### Stage 3 — Qwen + SayCan planner

- Uses DashScope's OpenAI-compatible endpoint. Configure `DASHSCOPE_API_KEY` (or `QWEN_API_KEY`), optionally `QWEN_MODEL` and `QWEN_BASE_URL`.
- Generates candidate skills from the earliest unsatisfied BDDL goals and a bounded look-ahead.
- Qwen returns one semantic relevance score per candidate; a separate deterministic Can evaluator checks state preconditions and reachability.
- Combines them as `score = Say × Can`, logs every candidate and selected action.
- `--scorer keyword` is an explicit offline baseline; it is never silently substituted for Qwen.

### Stage 4 — Skill library

Implemented high-level skills:

| Skill | Execution / verification |
|---|---|
| `Pick` | Panda OSC_POSE approach, descend, close gripper, lift; verifies object height change |
| `PutOn` | Computes a target pose, releases the held object, verifies the BDDL `on` predicate |
| `PutInside` | Opens articulated container first when needed, places into the BDDL region and verifies `in` |
| `Open` / `Close` | Approaches an articulation handle and moves along a joint-derived handle path; verifies state |
| `TurnOn` / `TurnOff` | Uses articulation metadata and checks the corresponding fixture state |
| `Push` | Contact-height push toward a target region; verifies the BDDL `on` predicate |
| `Done` | Calls the official LIBERO success check |

These are geometric closed-loop primitives, not general-purpose motion planning. A task may fail because a grasp pose, handle estimate, placement point, collision-free path or Can threshold is unsuitable for that asset. Such failures should be recorded and used to improve the relevant skill rather than reported as benchmark success.

### Stage 5 — Dynamic replanning

- Refreshes the simulator snapshot and all BDDL goal predicates before each planning round and after skill execution.
- Records pre/post belief, Say, Can, combined score, execution result and retry count.
- Failed skills increment a per-skill retry counter; the planner excludes a repeatedly failing candidate after the configured retry budget and regenerates candidates from the updated state.
- The planner terminates on official LIBERO success, no feasible candidate, or the maximum planning-round budget.

### Stage 6 — Official suite evaluation

The evaluator supports `libero_spatial`, `libero_object`, `libero_goal`, `libero_90` and `libero_10`. It writes per-episode JSON, per-suite JSON summaries, aggregate JSON and CSV. Use a small task subset first; full-suite evaluation can take substantial time and will make many Qwen API calls.

## 7. Running Qwen-SayCan

Set the API key in the shell; do not commit it to Git:

~~~bash
export DASHSCOPE_API_KEY="YOUR_DASHSCOPE_API_KEY"
export QWEN_MODEL="qwen-plus"
# Optional:
# export QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
~~~

Run one official task:

~~~bash
python scripts/run_saycan_libero.py \
  --suite libero_spatial \
  --task-id 0 \
  --init-state-id 0 \
  --scorer qwen \
  --max-rounds 40 \
  --render-gpu-device-id 0
~~~

Offline planner smoke test (no API key):

~~~bash
python scripts/run_saycan_libero.py \
  --suite libero_spatial \
  --task-id 0 \
  --init-state-id 0 \
  --scorer keyword \
  --max-rounds 40 \
  --render-gpu-device-id 0
~~~

A keyword-baseline success is not a Qwen result. A successful episode is only counted when the official LIBERO success predicate returns true.

## 8. Running evaluations

Start with a single task and one fixed initial state:

~~~bash
python scripts/evaluate_suite.py \
  --suite libero_spatial \
  --task-ids 0 \
  --episodes-per-task 1 \
  --scorer keyword \
  --task-limit 1 \
  --render-gpu-device-id 0
~~~

Then run a small Qwen sample:

~~~bash
python scripts/evaluate_suite.py \
  --suite libero_spatial \
  --task-ids 0,1,2 \
  --episodes-per-task 2 \
  --scorer qwen \
  --render-gpu-device-id 0
~~~

Expand suite-by-suite only after inspecting failures and trajectory GIFs:

~~~bash
python scripts/evaluate_suite.py --suite libero_spatial --scorer qwen
python scripts/evaluate_suite.py --suite libero_object --scorer qwen
python scripts/evaluate_suite.py --suite libero_goal --scorer qwen
python scripts/evaluate_suite.py --suite libero_90 --scorer qwen
python scripts/evaluate_suite.py --suite libero_10 --scorer qwen
~~~

For a quick coverage check without spending API quota, use `--scorer keyword` first. Keep Qwen and keyword results in separate output directories when comparing them.

## 9. Verification status and limitations

- Unit tests cover BDDL predicate flattening, task-spec extraction, candidate generation, Qwen response validation, metrics and failed-skill replanning.
- Static unit tests do not prove MuJoCo contacts, grasping, articulation or placement work in a real LIBERO episode.
- The LIBERO simulator, EGL renderer, physical skills and official suites must be smoke-tested on the target A40 server before claiming end-to-end success.
- The current Can evaluator is rule-based and uses simulator state; the original SayCan paper's affordance/value functions are learned from robot interaction data.
- Current skills use direct geometric waypoints and do not provide general collision-aware motion planning or camera-only perception.
- Report success rate only from `env.check_success()`; distinguish infrastructure errors, no-feasible-skill, max-rounds and physical skill failures.

## 7. 参考资料

- [LIBERO official repository](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [LIBERO project website](https://libero-project.github.io)
- [LIBERO paper](https://arxiv.org/abs/2306.03310)
- [LIBERO official README](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/README.md)
