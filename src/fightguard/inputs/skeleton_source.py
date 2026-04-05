"""
skeleton_source.py — NTU RGBD 骨骼数据读取模块
===============================================
负责读取 NTU RGBD 的 .skeleton 文件，
将其转换为 contracts.py 定义的 TrackSet 格式。

核心工作：
  1. 解析 .skeleton 文件的帧结构
  2. 将 NTU 25点 3D 坐标映射到 COCO-17 字典格式
  3. 从文件名提取动作类别标签
  4. 返回标准化的 TrackSet 对象

队长要求：禁止硬编码数字索引，全部用字典键名访问。
NTU 原始数组索引只在本文件的映射表中出现一次，
映射完成后外部代码永远只看到 COCO-17 键名。
"""

import os
import glob
from typing import List, Optional, Tuple

from fightguard.contracts import (
    COCO17_KEYPOINT_NAMES,
    Keypoints,
    SkeletonTrack,
    TrackSet,
    keypoints_from_array,
)
from fightguard.config import get_config


# ============================================================
# NTU 25点 → COCO-17 映射表
# key  : COCO-17 关键点名称
# value: NTU 原始数组中的索引（0起）
# 此映射表是项目中唯一允许出现 NTU 数字索引的地方
# 参考：https://github.com/shahroudy/NTURGB-D
# ============================================================
NTU_TO_COCO17: dict = {
    "nose":            3,   # NTU: head (近似)
    "left_eye":        3,   # NTU: head (无独立眼部点，用head近似)
    "right_eye":       3,   # NTU: head (同上)
    "left_ear":        3,   # NTU: head (同上)
    "right_ear":       3,   # NTU: head (同上)
    "left_shoulder":   4,   # NTU: left shoulder
    "right_shoulder":  8,   # NTU: right shoulder
    "left_elbow":      5,   # NTU: left elbow
    "right_elbow":     9,   # NTU: right elbow
    "left_wrist":      6,   # NTU: left wrist
    "right_wrist":     10,  # NTU: right wrist
    "left_hip":        12,  # NTU: left hip
    "right_hip":       16,  # NTU: right hip
    "left_knee":       13,  # NTU: left knee
    "right_knee":      17,  # NTU: right knee
    "left_ankle":      14,  # NTU: left ankle
    "right_ankle":     18,  # NTU: right ankle
}


# ============================================================
# 第一部分：文件名解析
# ============================================================

def parse_clip_id(filename: str) -> dict:
    """
    从 NTU .skeleton 文件名解析元信息。

    文件名格式：S001C001P001R001A049.skeleton
      S: 场景(setup)   C: 摄像机   P: 受试者   R: 重复   A: 动作类别

    参数：
        filename: 文件名（含或不含路径均可）

    返回：
        字典，包含 clip_id / setup / camera / subject / repeat / action_id
    """
    basename = os.path.splitext(os.path.basename(filename))[0]  # 去掉路径和扩展名
    try:
        info = {
            "clip_id":  basename,
            "setup":    int(basename[1:4]),    # S001 → 1
            "camera":   int(basename[5:8]),    # C001 → 1
            "subject":  int(basename[9:12]),   # P001 → 1
            "repeat":   int(basename[13:16]),  # R001 → 1
            "action_id": int(basename[17:20]), # A049 → 49
        }
    except (ValueError, IndexError):
        raise ValueError(f"无法解析文件名：{filename}，请确认是标准 NTU 命名格式")
    return info


def is_conflict_clip(action_id: int, cfg: Optional[dict] = None) -> int:
    """
    根据动作类别编号判断该 clip 是否为冲突样本。

    参数：
        action_id: NTU 动作类别编号（如 49）
        cfg      : 配置字典，不传则自动读取

    返回：
        1 = 冲突样本，0 = 正常样本，-1 = 其他（不参与评测）
    """
    if cfg is None:
        cfg = get_config()
    conflict_actions = cfg["dataset"]["ntu_conflict_actions"]   # [49, 50, 51]
    normal_actions   = cfg["dataset"]["ntu_normal_actions"]     # [1..10]

    if action_id in conflict_actions:
        return 1
    elif action_id in normal_actions:
        return 0
    else:
        return -1  # 既不是冲突也不是正常对照，跳过


# ============================================================
# 第二部分：单帧骨骼解析
# ============================================================

def _parse_one_body(lines: List[str], start_idx: int) -> Tuple[Keypoints, int]:
    """
    解析单帧中一个人的骨骼数据。

    参数：
        lines    : 文件所有行的列表
        start_idx: 从第几行开始解析（指向帧头行）

    返回：
        (Keypoints字典, 解析结束后的行索引)
    """
    # 跳过帧头行（10个元数据，我们不需要）
    idx = start_idx + 1

    # 读取关键点数量（应为25）
    num_joints = int(lines[idx].strip())
    idx += 1

    # 读取25个关键点的原始坐标
    raw_joints = []
    for _ in range(num_joints):
        vals = lines[idx].strip().split()
        # 只取前两个值：x, y（归一化处理在后面做）
        x = float(vals[0])
        y = float(vals[1])
        raw_joints.append([x, y])
        idx += 1

    # 按映射表构建 COCO-17 字典
    # 注意：NTU 坐标是3D世界坐标（单位：米），不是归一化的
    # 这里先保留原始值，归一化在 pairing.py 中按需处理
    keypoints: Keypoints = {}
    for coco_name in COCO17_KEYPOINT_NAMES:
        ntu_idx = NTU_TO_COCO17[coco_name]
        if ntu_idx < len(raw_joints):
            keypoints[coco_name] = raw_joints[ntu_idx]
        else:
            keypoints[coco_name] = [0.0, 0.0]  # 缺失点用零填充

    return keypoints, idx


# ============================================================
# 第三部分：单文件读取 → TrackSet
# ============================================================

def load_skeleton_file(filepath: str, cfg: Optional[dict] = None) -> Optional[TrackSet]:
    """
    读取单个 .skeleton 文件，返回 TrackSet 对象。

    参数：
        filepath: .skeleton 文件的完整路径
        cfg     : 配置字典，不传则自动读取

    返回：
        TrackSet 对象；如果该 clip 不在评测范围内（label=-1）则返回 None
    """
    if cfg is None:
        cfg = get_config()

    # 解析文件名，获取动作标签
    meta    = parse_clip_id(filepath)
    label   = is_conflict_clip(meta["action_id"], cfg)
    if label == -1:
        return None  # 不在评测范围，跳过

    # 读取文件所有行
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total_frames = int(lines[0].strip())  # 第1行：总帧数
    idx = 1                               # 当前解析位置

    # track_id → SkeletonTrack 的映射（最多支持2人）
    tracks_dict: dict = {}

    for frame_no in range(total_frames):
        if idx >= len(lines):
            break

        # 本帧检测到的人数
        num_bodies = int(lines[idx].strip())
        idx += 1

        for body_no in range(num_bodies):
            if idx >= len(lines):
                break

            # 解析该人的骨骼
            keypoints, idx = _parse_one_body(lines, idx)

            # 用 body_no 作为临时 track_id（NTU单文件内人员编号固定）
            track_id = body_no
            if track_id not in tracks_dict:
                tracks_dict[track_id] = SkeletonTrack(
                    track_id=track_id,
                    role="child",   # NTU数据集中统一视为child，无teacher区分
                )
            tracks_dict[track_id].frames.append(frame_no)
            tracks_dict[track_id].keypoints.append(keypoints)

    # 构建 TrackSet
    track_set = TrackSet(
        clip_id=meta["clip_id"],
        label=label,
        tracks=list(tracks_dict.values()),
        fps=30.0,
        total_frames=total_frames,
    )
    return track_set


# ============================================================
# 第四部分：批量加载数据集目录 → TrackSet 列表
# ============================================================

def load_dataset(data_dirs: List[str], cfg: Optional[dict] = None,
                 max_clips: Optional[int] = None) -> List[TrackSet]:
    """
    批量读取一个或多个目录下的所有 .skeleton 文件。

    参数：
        data_dirs: 目录路径列表，如 ["D:/dataset_1/nturgbd_s001_to_s017",
                                     "D:/dataset_1/nturgbd_s018_to_s032"]
        cfg      : 配置字典，不传则自动读取
        max_clips: 调试用，限制最多读取多少个clip（None=全部读取）

    返回：
        TrackSet 列表（已过滤掉 label=-1 的clip）
    """
    if cfg is None:
        cfg = get_config()

    all_track_sets: List[TrackSet] = []
    skipped = 0
    loaded  = 0

    for data_dir in data_dirs:
        pattern = os.path.join(data_dir, "*.skeleton")
        files   = sorted(glob.glob(pattern))

        if not files:
            print(f"[WARNING] 目录下未找到 .skeleton 文件：{data_dir}")
            continue

        print(f"[INFO] 扫描目录：{data_dir}，共找到 {len(files)} 个文件")

        for filepath in files:
            if max_clips is not None and loaded >= max_clips:
                break

            try:
                track_set = load_skeleton_file(filepath, cfg)
                if track_set is None:
                    skipped += 1
                    continue
                all_track_sets.append(track_set)
                loaded += 1
            except Exception as e:
                print(f"[ERROR] 读取失败：{filepath}，原因：{e}")
                skipped += 1

    print(f"[INFO] 加载完成：{loaded} 个clip，跳过 {skipped} 个（不在评测范围或读取失败）")
    print(f"[INFO] 冲突样本：{sum(1 for t in all_track_sets if t.label == 1)}，"
          f"正常样本：{sum(1 for t in all_track_sets if t.label == 0)}")
    return all_track_sets
