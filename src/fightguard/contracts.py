"""
contracts.py — 数据契约
=======================
定义项目中所有模块之间传递数据的统一格式。
队长要求：统一采用字典（Dict）传参，拒绝硬编码数字索引。
关键点命名统一参考 COCO-17 标准。

核心数据结构：
  - Keypoints   : 单帧单人的骨骼关键点字典
  - SkeletonTrack: 单人的多帧骨骼序列（轨迹）
  - TrackSet    : 一个片段内所有人的轨迹集合
  - InteractionEvent: 一次冲突/异常事件的结构化描述
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ============================================================
# 第一部分：COCO-17 关键点名称常量
# 使用方式：直接用字符串键名访问，如 kp["left_wrist"]
# ============================================================

# COCO-17 标准关键点名称列表（顺序与官方索引对应）
COCO17_KEYPOINT_NAMES: List[str] = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]

# 快速查找：关键点名称 -> COCO-17 索引
COCO17_NAME_TO_IDX: Dict[str, int] = {
    name: idx for idx, name in enumerate(COCO17_KEYPOINT_NAMES)
}


# ============================================================
# 第二部分：单帧骨骼关键点类型别名
# Keypoints = {"nose": [x, y], "left_shoulder": [x, y], ...}
# 坐标为归一化值（0~1），相对于画面宽高
# ============================================================

# 类型别名：关键点字典
# key  : COCO-17 关键点名称（字符串）
# value: [x, y] 归一化坐标，可选第三维 confidence
Keypoints = Dict[str, List[float]]


def make_empty_keypoints() -> Keypoints:
    """生成一个所有关键点坐标为 [0.0, 0.0] 的空白字典，用于占位或缺失处理。"""
    return {name: [0.0, 0.0] for name in COCO17_KEYPOINT_NAMES}


def keypoints_from_array(array: List[List[float]],
                         names: List[str] = COCO17_KEYPOINT_NAMES) -> Keypoints:
    """
    将数组格式的关键点转换为字典格式。
    
    参数：
        array : 关键点坐标列表，如 [[x0,y0], [x1,y1], ...]
        names : 对应的关键点名称列表，默认使用 COCO-17 标准
    
    返回：
        Keypoints 字典，如 {"nose": [x, y], "left_eye": [x, y], ...}
    
    示例：
        # NTU 原始数据是25个点的数组，经过映射后调用此函数
        kp = keypoints_from_array(raw_array, COCO17_KEYPOINT_NAMES)
        dist = kp["left_wrist"][0] - kp["right_wrist"][0]  # 正确写法
        # 禁止写成: raw_array[9][0] - raw_array[10][0]     # 禁止！
    """
    if len(array) != len(names):
        raise ValueError(
            f"关键点数量不匹配：array有{len(array)}个点，names有{len(names)}个名称"
        )
    return {name: list(coords) for name, coords in zip(names, array)}


# ============================================================
# 第三部分：SkeletonTrack — 单人多帧骨骼轨迹
# ============================================================

@dataclass
class SkeletonTrack:
    """
    单个人在一段时间内的骨骼轨迹。
    
    属性：
        track_id   : 人员唯一标识（整数）
        role       : 角色标签，"child"（幼儿）或 "teacher"（教师）
        frames     : 帧索引列表，如 [0, 1, 2, 3, ...]
        keypoints  : 每帧对应的关键点字典列表，与 frames 等长
        confidences: 每帧的整体置信度（可选），与 frames 等长
    
    使用示例：
        track.keypoints[i]["left_wrist"]  # 第i帧的左手腕坐标
    """
    track_id:    int
    role:        str = "child"           # "child" 或 "teacher"
    frames:      List[int] = field(default_factory=list)
    keypoints:   List[Keypoints] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)

    def get_keypoint_at(self, frame_idx: int, keypoint_name: str) -> Optional[List[float]]:
        """
        获取指定帧、指定关键点的坐标。
        
        参数：
            frame_idx    : 帧索引（在 self.frames 中的位置，不是绝对帧号）
            keypoint_name: COCO-17 关键点名称，如 "left_wrist"
        
        返回：
            [x, y] 坐标，如果该帧或关键点不存在则返回 None
        """
        if frame_idx >= len(self.keypoints):
            return None
        return self.keypoints[frame_idx].get(keypoint_name, None)

    def get_body_center(self, frame_idx: int) -> Optional[List[float]]:
        """
        计算指定帧的躯干中心点（左右髋关节的中点）。
        用于判断两人之间的距离。
        """
        kp = self.keypoints[frame_idx] if frame_idx < len(self.keypoints) else None
        if kp is None:
            return None
        lh = kp.get("left_hip")
        rh = kp.get("right_hip")
        if lh is None or rh is None:
            return None
        return [(lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0]

    def __len__(self) -> int:
        return len(self.frames)


# ============================================================
# 第四部分：TrackSet — 一个片段内所有人的轨迹集合
# ============================================================

@dataclass
class TrackSet:
    """
    一个视频片段（clip）内所有被检测到的人的轨迹集合。
    
    属性：
        clip_id   : 片段唯一标识（字符串，如 "S001C001P001R001A049"）
        label     : 片段级标签，1=冲突，0=正常（来自数据集标注）
        tracks    : 所有人的轨迹列表
        fps       : 视频帧率，用于时间换算
        total_frames: 片段总帧数
    """
    clip_id:      str
    label:        int = -1               # -1 表示未标注，0=正常，1=冲突
    tracks:       List[SkeletonTrack] = field(default_factory=list)
    fps:          float = 30.0
    total_frames: int = 0

    def get_children(self) -> List[SkeletonTrack]:
        """返回所有角色为 'child' 的轨迹。"""
        return [t for t in self.tracks if t.role == "child"]

    def get_teachers(self) -> List[SkeletonTrack]:
        """返回所有角色为 'teacher' 的轨迹。"""
        return [t for t in self.tracks if t.role == "teacher"]

    def get_track_by_id(self, track_id: int) -> Optional[SkeletonTrack]:
        """根据 track_id 查找轨迹。"""
        for t in self.tracks:
            if t.track_id == track_id:
                return t
        return None


# ============================================================
# 第五部分：InteractionEvent — 一次冲突/异常事件
# ============================================================

@dataclass
class InteractionEvent:
    """
    一次被规则流检测到的交互事件（冲突或异常）。
    
    属性：
        clip_id       : 来源片段 ID
        event_type    : 事件类型，如 "child_conflict"、"teacher_misconduct"
        start_frame   : 事件开始帧
        end_frame     : 事件结束帧
        track_ids     : 涉及的人员 track_id 列表
        score         : 规则触发的置信度分数（0~1）
        triggered_rules: 触发了哪些具体规则（便于可解释性）
        teacher_present: 事件发生时教师是否在场
        region        : 事件发生的功能区域（如 "activity_zone"）
    """
    clip_id:         str
    event_type:      str
    start_frame:     int
    end_frame:       int
    track_ids:       List[int] = field(default_factory=list)
    score:           float = 0.0
    triggered_rules: List[str] = field(default_factory=list)
    teacher_present: bool = False
    region:          str = "unknown"

    @property
    def duration_frames(self) -> int:
        """事件持续帧数。"""
        return self.end_frame - self.start_frame

    def duration_seconds(self, fps: float = 30.0) -> float:
        """事件持续秒数。"""
        return self.duration_frames / fps

    def to_dict(self) -> Dict:
        """转换为字典，用于写入 CSV/JSON 日志。"""
        return {
            "clip_id":         self.clip_id,
            "event_type":      self.event_type,
            "start_frame":     self.start_frame,
            "end_frame":       self.end_frame,
            "duration_frames": self.duration_frames,
            "track_ids":       str(self.track_ids),
            "score":           round(self.score, 4),
            "triggered_rules": str(self.triggered_rules),
            "teacher_present": self.teacher_present,
            "region":          self.region,
        }
