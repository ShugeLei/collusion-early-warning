#!/usr/bin/env python3
"""Validate the lightweight, CPU-only Phase 0 environment."""

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "external" / "narcbench"
DATA_ROOT = PROJECT_ROOT / "data" / "narcbench"

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
}


def directory_size(path):
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def physical_memory_bytes():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None


def git_commit(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def package_versions():
    versions = {}
    missing = []
    for distribution, module in REQUIRED_PACKAGES.items():
        try:
            versions[distribution] = metadata.version(distribution)
            __import__(module)
        except (metadata.PackageNotFoundError, ImportError):
            missing.append(distribution)
    return versions, missing


def main():
    versions, missing_packages = package_versions()
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    memory = physical_memory_bytes()
    heavy_modules = {
        name: importlib.util.find_spec(name) is not None
        for name in ("torch", "transformers", "vllm", "mlx")
    }

    failures = []
    warnings = []
    if platform.system() != "Darwin":
        failures.append("Expected macOS (Darwin).")
    if platform.machine() != "arm64":
        failures.append("Expected Apple Silicon (arm64).")
    if not in_venv:
        failures.append("Python is not running from a virtual environment.")
    if missing_packages:
        failures.append("Missing packages: " + ", ".join(missing_packages))
    if not UPSTREAM_ROOT.joinpath("probes", "reproduce.py").is_file():
        failures.append("Official NARCBench checkout is incomplete.")
    if any(heavy_modules.values()):
        warnings.append(
            "A model-runtime package is importable, although none is needed for Phase 0: "
            + ", ".join(name for name, present in heavy_modules.items() if present)
        )
    if sys.version_info < (3, 10):
        failures.append(
            "Python 3.10 or newer is required because upstream NARCBench uses PEP 604 "
            "union annotations at import time."
        )

    report = {
        "status": "PASS" if not failures else "FAIL",
        "project_root": str(PROJECT_ROOT),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "physical_memory_bytes": memory,
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "in_virtual_environment": in_venv,
            "prefix": sys.prefix,
            "base_prefix": getattr(sys, "base_prefix", sys.prefix),
        },
        "packages": versions,
        "model_runtime_modules_importable": heavy_modules,
        "nvidia_smi_present": shutil.which("nvidia-smi") is not None,
        "upstream": {
            "path": str(UPSTREAM_ROOT),
            "git_commit": git_commit(UPSTREAM_ROOT),
            "official_probe_present": UPSTREAM_ROOT.joinpath(
                "probes", "reproduce.py"
            ).is_file(),
        },
        "paths": {
            "data_root": str(DATA_ROOT),
            "data_root_exists": DATA_ROOT.is_dir(),
            "venv_size_bytes": directory_size(PROJECT_ROOT / ".venv"),
        },
        "warnings": warnings,
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
