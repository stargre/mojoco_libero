# MuJoCo LIBERO SayCan

将 LIBERO（Lifelong Robot Learning Benchmark）接入独立的 MuJoCo SayCan 实验工程。项目以官方 LIBERO 环境为仿真后端，逐步接入 Qwen Say、Can affordance 评估和可执行技能库。

> 当前里程碑：Stage 1 — LIBERO 环境接入与可复现 smoke test。当前代码能够加载官方任务、固定初始状态、执行 MuJoCo step、检查任务成功条件并保存图像/GIF/JSON。**当前 smoke test 使用 no-op action，不代表已经实现 Qwen SayCan 任务求解。**

## 1. 为什么使用独立环境

官方 LIBERO README 推荐 Python 3.8.13，并固定了较老版本的 robosuite、Gym、NumPy 和 PyTorch。为了避免破坏原来的 mojoco 环境，也避免和 mujoco_saycan 的 Python 3.11 环境发生依赖冲突，本项目使用独立 Conda 环境 libero-saycan。

当前参考依赖：

| Component | Version |
|---|---:|
| Python | 3.8.13 |
| LIBERO | official master |
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

安装脚本会把官方 LIBERO clone 到 third_party/LIBERO（该目录已加入 .gitignore），安装官方 requirements、MuJoCo/PyTorch 兼容版本，并以 editable 模式安装 LIBERO 和本项目。

如果服务器没有 EGL/OpenGL 开发运行库，需要先由管理员安装系统依赖。运行前设置：

~~~bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
~~~

建议先检查安装：

~~~bash
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

## 6. 当前阶段边界与后续路线

当前阶段完成的是 LIBERO simulator adapter，不是完整 SayCan 复现，也不包含可执行的通用机械臂技能。

后续按以下顺序开发：

1. **Stage 1 — LIBERO 环境接入（当前）**：suite/task 加载、固定初始状态、观测、step、成功判定、GIF 和 JSON。
2. **Stage 2 — 状态与任务语义适配**：解析 BDDL task、对象/fixture、初始状态与 goal predicates；建立统一 BeliefState。
3. **Stage 3 — SayCan Planner**：接入 Qwen relevance scorer，生成多个候选技能，计算 Say × Can 并记录每轮决策。
4. **Stage 4 — 可执行技能库**：先实现并验证 Pick / Place，再扩展 Open / Close / PutOn / PutInside 等任务所需技能；低层控制由 LIBERO/robosuite action 接口执行。
5. **Stage 5 — 动态重规划**：执行失败后刷新观测与 belief，重新生成候选技能。
6. **Stage 6 — 基准评测**：固定 task、initial-state、seed 和 episode budget，统计成功率、步骤数、规划请求数和失败类型。

LIBERO 的任务初始状态文件可用于固定初始条件；人类示范 HDF5 数据集是训练/示范分析所需的独立下载项，不是运行本仓库第一个环境 smoke test 的前置条件。

## 7. 参考资料

- [LIBERO official repository](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [LIBERO project website](https://libero-project.github.io)
- [LIBERO paper](https://arxiv.org/abs/2306.03310)
- [LIBERO official README](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/README.md)
