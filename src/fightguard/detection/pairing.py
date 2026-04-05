"""
pairing.py — 多人交互配对模块
==============================
从一帧或一段轨迹中，选出最可能发生交互的两个人组成配对。
配对策略：计算所有人两两之间的躯干中心距离，选距离最近的一对。

输入：TrackSet（一个clip内所有人的轨迹）
输出：List[Tuple[SkeletonTrack, SkeletonTrack]]（候选交互对列表）
"""

import math
from typing import List, Optional, Tuple

from fightguard.contracts import SkeletonTrack, TrackSet
from fightguard.config import get_config


# ============================================================
# 第一部分：基础几何工具函数
# ============================================================

def euclidean_distance(p1: List[float], p2: List[float]) -> float:
    """
    计算两个二维点之间的欧氏距离。

    参数：
        p1, p2: [x, y] 坐标列表

    返回：
        浮点数距离值
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_body_center(keypoints: dict) -> Optional[List[float]]:
    """
    计算躯干中心点（左右髋关节的中点）。
    用于衡量两人之间的整体距离。

    参数：
        keypoints: COCO-17 格式的关键点字典

    返回：
        [x, y] 中心坐标，如果髋关节数据缺失则返回 None
    """
    lh = keypoints.get("left_hip")
    rh = keypoints.get("right_hip")
    if lh is None or rh is None:
        return None
    # 左右髋关节坐标均为零时视为无效数据
    if lh == [0.0, 0.0] and rh == [0.0, 0.0]:
        return None
    return [(lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0]


def get_shoulder_center(keypoints: dict) -> Optional[List[float]]:
    """
    计算肩部中心点（左右肩关节的中点）。
    当髋关节数据不可用时作为备用躯干中心。

    参数：
        keypoints: COCO-17 格式的关键点字典

    返回：
        [x, y] 中心坐标，如果肩关节数据缺失则返回 None
    """
    ls = keypoints.get("left_shoulder")
    rs = keypoints.get("right_shoulder")
    if ls is None or rs is None:
        return None
    if ls == [0.0, 0.0] and rs == [0.0, 0.0]:
        return None
    return [(ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0]


def get_torso_center(keypoints: dict) -> Optional[List[float]]:
    """
    获取躯干中心点，优先用髋关节，髋关节无效时用肩部中心。

    参数：
        keypoints: COCO-17 格式的关键点字典

    返回：
        [x, y] 中心坐标，两者都无效则返回 None
    """
    center = get_body_center(keypoints)
    if center is not None:
        return center
    return get_shoulder_center(keypoints)


# ============================================================
# 第二部分：单帧配对距离计算
# ============================================================

def compute_pair_distance_at_frame(
    track_a: SkeletonTrack,
    track_b: SkeletonTrack,
    frame_idx: int
) -> Optional[float]:
    """
    计算两条轨迹在指定帧的躯干中心距离。

    参数：
        track_a, track_b: 两个人的骨骼轨迹
        frame_idx       : 帧索引（在各自轨迹中的位置）

    返回：
        距离值，如果任一轨迹该帧无效则返回 None
    """
    if frame_idx >= len(track_a.keypoints) or frame_idx >= len(track_b.keypoints):
        return None

    center_a = get_torso_center(track_a.keypoints[frame_idx])
    center_b = get_torso_center(track_b.keypoints[frame_idx])

    if center_a is None or center_b is None:
        return None

    return euclidean_distance(center_a, center_b)


# ============================================================
# 第三部分：轨迹级平均距离计算
# ============================================================

def compute_pair_mean_distance(
    track_a: SkeletonTrack,
    track_b: SkeletonTrack
) -> float:
    """
    计算两条轨迹在所有共同帧上的平均躯干中心距离。
    用于从多人中筛选出"最近的一对"作为交互候选。

    参数：
        track_a, track_b: 两个人的骨骼轨迹

    返回：
        平均距离值；如果没有有效帧则返回正无穷（表示无法配对）
    """
    n_frames = min(len(track_a.keypoints), len(track_b.keypoints))
    distances = []

    for i in range(n_frames):
        dist = compute_pair_distance_at_frame(track_a, track_b, i)
        if dist is not None:
            distances.append(dist)

    if not distances:
        return float("inf")  # 无有效帧，距离视为无穷大

    return sum(distances) / len(distances)


# ============================================================
# 第四部分：主配对函数
# ============================================================

def get_interaction_pairs(
    track_set: TrackSet,
    cfg: Optional[dict] = None,
    top_k: int = 1
) -> List[Tuple[SkeletonTrack, SkeletonTrack]]:
    """
    从 TrackSet 中选出最可能发生交互的人员配对。

    策略：
      1. 枚举所有人员两两组合
      2. 计算每对的平均躯干中心距离
      3. 返回距离最近的 top_k 对

    参数：
        track_set: 一个clip内所有人的轨迹集合
        cfg      : 配置字典，不传则自动读取
        top_k    : 返回最近的前k对，默认为1

    返回：
        配对列表，每个元素是 (SkeletonTrack, SkeletonTrack) 元组
        如果人数不足2人，返回空列表
    """
    if cfg is None:
        cfg = get_config()

    tracks = track_set.tracks
    if len(tracks) < 2:
        return []  # 少于2人无法配对

    # 枚举所有两两组合，计算平均距离
    pairs_with_dist = []
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            dist = compute_pair_mean_distance(tracks[i], tracks[j])
            pairs_with_dist.append((dist, tracks[i], tracks[j]))

    # 按距离升序排列，取前 top_k 对
    pairs_with_dist.sort(key=lambda x: x[0])
    result = [(a, b) for _, a, b in pairs_with_dist[:top_k]]

    return result


def get_proximity_frames(
    track_a: SkeletonTrack,
    track_b: SkeletonTrack,
    proximity_threshold: float
) -> List[int]:
    """
    找出两人距离小于阈值的所有帧索引。
    用于在规则模块中快速定位"近身帧"，缩小判断范围。

    参数：
        track_a, track_b    : 两个人的骨骼轨迹
        proximity_threshold : 距离阈值（来自 default.yaml）

    返回：
        满足近身条件的帧索引列表
    """
    n_frames = min(len(track_a.keypoints), len(track_b.keypoints))
    proximity_frames = []

    for i in range(n_frames):
        dist = compute_pair_distance_at_frame(track_a, track_b, i)
        if dist is not None and dist < proximity_threshold:
            proximity_frames.append(i)

    return proximity_frames
