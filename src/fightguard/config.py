"""
config.py — 配置读取与校验
==========================
负责读取 configs/default.yaml，并向其他模块提供统一的配置访问接口。
所有模块需要读取配置时，统一从此模块导入 get_config()，
禁止在业务代码中直接硬编码路径或阈值数字。

使用示例：
    from fightguard.config import get_config
    cfg = get_config()
    threshold = cfg["rules"]["proximity_threshold"]  # 正确
    threshold = 0.3                                   # 禁止硬编码！
"""

import os
import yaml
from typing import Any, Dict, Optional

# ============================================================
# 模块级缓存：配置只读取一次，后续复用同一个对象
# ============================================================
_config_cache: Optional[Dict[str, Any]] = None

# 默认配置文件路径（相对于项目根目录）
_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),   # src/fightguard/
    "..", "..",                   # 退到 src/，再退到 kidguard/
    "configs", "default.yaml"
)


def get_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    读取并返回全局配置字典。
    
    第一次调用时从 YAML 文件读取并缓存；
    后续调用直接返回缓存，不重复读取文件。
    
    参数：
        config_path: 可选，指定配置文件路径。
                     不传则使用默认路径 configs/default.yaml。
    
    返回：
        配置字典，结构与 default.yaml 完全一致。
    
    异常：
        FileNotFoundError : 配置文件不存在时抛出。
        ValueError        : 配置文件格式错误时抛出。
    """
    global _config_cache

    # 如果已缓存且没有指定新路径，直接返回缓存
    if _config_cache is not None and config_path is None:
        return _config_cache

    # 确定实际使用的路径
    path = config_path if config_path else _DEFAULT_CONFIG_PATH
    path = os.path.normpath(path)  # 统一路径分隔符

    # 检查文件是否存在
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"配置文件不存在：{path}\n"
            f"请确认 configs/default.yaml 文件已正确创建。"
        )

    # 读取 YAML
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(
            f"配置文件格式错误：{path}\n"
            f"期望得到字典（dict），实际得到 {type(config)}"
        )

    # 校验必要字段是否存在
    _validate_config(config, path)

    # 写入缓存
    _config_cache = config
    return _config_cache


def reload_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    强制重新读取配置文件，清除缓存。
    在调参调试时使用，无需重启程序即可加载新阈值。
    """
    global _config_cache
    _config_cache = None
    return get_config(config_path)


def _validate_config(config: Dict[str, Any], path: str) -> None:
    """
    校验配置文件中必要字段是否存在。
    如有缺失，抛出清晰的错误信息提示用户补充。
    """
    required_keys = ["paths", "rules", "dataset", "output"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(
            f"配置文件缺少必要字段：{missing}\n"
            f"请检查：{path}"
        )

    # 校验 rules 子字段
    required_rules = [
        "proximity_threshold",
        "wrist_intrusion_threshold",
        "velocity_threshold",
        "conflict_duration_frames",
    ]
    missing_rules = [k for k in required_rules if k not in config.get("rules", {})]
    if missing_rules:
        raise ValueError(
            f"配置文件 rules 部分缺少字段：{missing_rules}\n"
            f"请检查：{path}"
        )