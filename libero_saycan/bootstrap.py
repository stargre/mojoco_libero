"""Configure LIBERO paths and headless rendering before importing LIBERO."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configure_libero():
    """Create a project-local LIBERO config and set rendering environment vars.

    LIBERO reads LIBERO_CONFIG_PATH while importing libero.libero. Therefore
    this function must run before importing libero.libero or libero.libero.envs.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    if os.environ.get("MUJOCO_GL", "").lower() == "egl":
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    repo_value = os.environ.get("LIBERO_REPO_PATH")
    repo_root = Path(repo_value).expanduser() if repo_value else PROJECT_ROOT / "third_party" / "LIBERO"
    repo_root = repo_root.resolve()
    package_root = repo_root / "libero" / "libero"

    required_dirs = {
        "benchmark_root": package_root,
        "bddl_files": package_root / "bddl_files",
        "init_states": package_root / "init_files",
        "datasets": repo_root / "libero" / "datasets",
        "assets": package_root / "assets",
    }
    missing = [
        str(path)
        for key, path in required_dirs.items()
        if key != "datasets" and not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Official LIBERO source/assets were not found. Run "
            "'bash scripts/install_libero.sh' first, or set LIBERO_REPO_PATH. "
            "Missing: " + ", ".join(missing)
        )

    config_value = os.environ.get("LIBERO_CONFIG_PATH")
    config_dir = (
        Path(config_value).expanduser()
        if config_value
        else PROJECT_ROOT / ".libero_config"
    )
    config_dir = config_dir.resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is missing. Install the official LIBERO requirements first."
        ) from exc

    config_file = config_dir / "config.yaml"
    config = {key: str(path) for key, path in required_dirs.items()}
    config_file.write_text(
        yaml.safe_dump(config, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "repo_root": str(repo_root),
        "package_root": str(package_root),
        "config_file": str(config_file),
        "paths": config,
    }
