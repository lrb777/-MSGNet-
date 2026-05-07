# -*- coding: utf-8 -*-
"""
统一结果目录管理。

默认结构：
  facts/runs/<run_id>/
    checkpoints/
    factors/
    validation/
    backtest/
    walk_forward/
    rolling_update/

环境变量：
  MSGNET_RUN_ID   指定 run_id，例如 20260507_test
  MSGNET_RUN_DIR  指定完整结果目录，优先级最高
"""

import os
from datetime import datetime

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(_SRC_DIR)
FACTS_DIR = os.path.join(PROJECT_DIR, "facts")
RUNS_DIR = os.path.join(FACTS_DIR, "runs")
LATEST_FILE = os.path.join(RUNS_DIR, "_latest_run.txt")


def _timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_latest(run_dir: str):
    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(LATEST_FILE, "w", encoding="utf-8") as f:
        f.write(os.path.abspath(run_dir))


def _read_latest() -> str | None:
    if not os.path.exists(LATEST_FILE):
        return None
    with open(LATEST_FILE, "r", encoding="utf-8") as f:
        path = f.read().strip()
    return path or None


def get_run_dir(create: bool = True, new: bool = False) -> str:
    env_dir = os.environ.get("MSGNET_RUN_DIR")
    if env_dir:
        run_dir = os.path.abspath(env_dir)
        if create:
            os.makedirs(run_dir, exist_ok=True)
            _write_latest(run_dir)
        return run_dir

    env_id = os.environ.get("MSGNET_RUN_ID")
    if env_id:
        run_dir = os.path.join(RUNS_DIR, env_id)
        if create:
            os.makedirs(run_dir, exist_ok=True)
            _write_latest(run_dir)
        return run_dir

    latest = None if new else _read_latest()
    run_dir = latest or os.path.join(RUNS_DIR, _timestamp_id())

    if create:
        os.makedirs(run_dir, exist_ok=True)
        _write_latest(run_dir)
    return run_dir


def get_subdir(name: str, create: bool = True, new_run: bool = False) -> str:
    path = os.path.join(get_run_dir(create=create, new=new_run), name)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def describe_run() -> str:
    return get_run_dir(create=False, new=False)
