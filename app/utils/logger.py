from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"


def setup_logging(log_path: str | None = None) -> None:
    """
    配置全局日志系统：
    - 同时输出到控制台与文件 `logs/ost.log`
    - 使用统一格式：`[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s`

    注意：
    - 为避免重复添加 handler，本函数可被多次调用，但只会初始化一次。
    """

    root = logging.getLogger()
    if getattr(root, "_ost_configured", False) is True:  # type: ignore[attr-defined]
        return

    # 默认日志文件路径：项目根目录 ./logs/ost.log
    if log_path is None:
        # app/utils/logger.py -> app -> project root
        root_dir = Path(__file__).resolve().parents[2]
        log_dir = root_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = (log_dir / "ost.log").as_posix()
    else:
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        log_path = p.as_posix()

    root.setLevel(logging.INFO)

    formatter = logging.Formatter(LOG_FORMAT)

    # 控制台输出
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    # 文件输出（滚动日志，避免无限增长）
    fh = RotatingFileHandler(log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    root.addHandler(sh)
    root.addHandler(fh)

    setattr(root, "_ost_configured", True)  # type: ignore[attr-defined]

