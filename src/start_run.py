# -*- coding: utf-8 -*-
"""手动创建一轮新的统一结果目录。"""

from run_config import get_run_dir


if __name__ == "__main__":
    run_dir = get_run_dir(create=True, new=True)
    print(f"新结果目录: {run_dir}")
