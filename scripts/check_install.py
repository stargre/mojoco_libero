"""Check the pinned LIBERO runtime and verify benchmark task metadata."""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libero_saycan.bootstrap import configure_libero


def main():
    config = configure_libero()

    import gym
    import mujoco
    import numpy
    import robosuite
    import torch
    from libero.libero import benchmark, get_libero_path

    versions = {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "gym": gym.__version__,
        "mujoco": mujoco.__version__,
        "robosuite": getattr(robosuite, "__version__", "unknown"),
        "torch": torch.__version__,
    }

    paths = {
        key: get_libero_path(key)
        for key in ("benchmark_root", "bddl_files", "init_states", "datasets", "assets")
    }
    path_status = {key: Path(value).exists() for key, value in paths.items()}

    suites = {}
    for name, suite_class in sorted(benchmark.get_benchmark_dict().items()):
        suite = suite_class()
        suites[name] = {
            "tasks": suite.get_num_tasks(),
            "first_task": suite.get_task(0).language if suite.get_num_tasks() else None,
        }

    report = {
        "versions": versions,
        "paths": paths,
        "path_exists": path_status,
        "suites": suites,
        "config": config,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    missing = [
        key for key, exists in path_status.items()
        if not exists and key != "datasets"
    ]
    if missing:
        raise SystemExit("Missing LIBERO paths: " + ", ".join(missing))

    expected_mujoco = "2.3.7"
    if mujoco.__version__ != expected_mujoco:
        print(
            "WARNING: expected mujoco {}, found {}. Re-run the smoke test before "
            "using this environment for benchmark results.".format(
                expected_mujoco, mujoco.__version__
            )
        )

    print("LIBERO import and benchmark registry: OK")


if __name__ == "__main__":
    main()
