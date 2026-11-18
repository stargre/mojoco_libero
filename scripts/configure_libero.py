from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libero_saycan.bootstrap import configure_libero


if __name__ == "__main__":
    info = configure_libero()
    print("LIBERO repository:", info["repo_root"])
    print("LIBERO package:", info["package_root"])
    print("LIBERO config:", info["config_file"])
    for key, value in sorted(info["paths"].items()):
        print("{}: {}".format(key, value))
