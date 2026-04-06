"""
math_utils.py — 基础几何与数学工具函数
======================================================
纯数学计算函数，避免循环导入问题。
"""

import math
from typing import List, Optional, Dict

def euclidean_distance(p1: List[float], p2: List[float]) -> float:
    """计算两点之间的欧氏距离"""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def get_neck_approx(kp: Dict[str, List[float]]) -> List[float]:
    """获取颈部近似坐标（使用鼻子关键点）"""
    return kp.get("nose", [0.0, 0.0, 0.0])

def get_pelvis_approx(kp: Dict[str, List[float]]) -> List[float]:
    """获取骨盆近似坐标（左右髋关节的中点）"""
    lh = kp.get("left_hip", [0.0, 0.0, 0.0])
    rh = kp.get("right_hip", [0.0, 0.0, 0.0])
    if lh[:2] == [0.0, 0.0] and rh[:2] == [0.0, 0.0]:
        return [0.0, 0.0, 0.0]
    return [(lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0, (lh[2] + rh[2]) / 2.0]

def get_body_center_formula(kp: Dict[str, List[float]]) -> Optional[List[float]]:
    """计算人体中心点
    公式：颈部（鼻子）和骨盆（左右髋关节中点）的平均值
    如果颈部或骨盆坐标无效，则返回 None
    """
    neck = get_neck_approx(kp)
    pelvis = get_pelvis_approx(kp)
    if neck[:2] == [0.0, 0.0] or pelvis[:2] == [0.0, 0.0]:
        return None
    return [(neck[0] + pelvis[0]) / 2.0, (neck[1] + pelvis[1]) / 2.0]

def get_shoulder_scale(kp: Dict[str, List[float]]) -> float:
    """计算肩宽尺度（物理标尺）
    公式：S_i = 欧氏距离(左肩坐标, 右肩坐标) + epsilon
    """
    ls = kp.get("left_shoulder", [0.0, 0.0, 0.0])
    rs = kp.get("right_shoulder", [0.0, 0.0, 0.0])
    if ls[:2] == [0.0, 0.0] or rs[:2] == [0.0, 0.0]:
        return 1.0  # 当肩点不可见时，使用默认值
    epsilon = 1e-6
    return euclidean_distance(ls, rs) + epsilon

def normalize_feature(value: float, min_val: float, max_val: float) -> float:
    """特征归一化"""
    if max_val <= min_val:
        return 0.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))