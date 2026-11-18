"""List task IDs and language instructions in an official LIBERO suite."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libero_saycan.bootstrap import configure_libero


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--all", action="store_true", dest="list_all")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    configure_libero()
    from libero.libero import benchmark

    registry = benchmark.get_benchmark_dict()
    names = sorted(registry.keys()) if args.list_all else [args.suite.lower()]
    output = []

    for name in names:
        if name not in registry:
            raise SystemExit(
                "Unknown suite {!r}. Available: {}".format(name, sorted(registry))
            )
        suite = registry[name]()
        tasks = []
        for task_id in range(suite.get_num_tasks()):
            task = suite.get_task(task_id)
            tasks.append({
                "suite": name,
                "task_id": task_id,
                "task_name": task.name,
                "instruction": task.language,
                "problem_folder": task.problem_folder,
                "bddl_file": task.bddl_file,
                "init_states_file": task.init_states_file,
            })
        output.extend(tasks)

    if args.as_json:
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    current_suite = None
    for item in output:
        if item["suite"] != current_suite:
            current_suite = item["suite"]
            print("\n{} ({} tasks)".format(
                current_suite, registry[current_suite]().get_num_tasks()
            ))
        print("  {:02d}  {}".format(item["task_id"], item["instruction"]))


if __name__ == "__main__":
    main()
