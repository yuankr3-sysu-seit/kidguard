"""
video_source.py — 视频骨骼提取模块 (阶段二核心)
==============================================
负责读取真实视频文件，使用 YOLOv8-Pose 模型逐帧提取人体骨骼关键点，
并将其转换为 contracts.py 定义的 TrackSet 格式。

核心工作：
  1. 调用 OpenCV 读取视频帧
  2. 调用 Ultralytics YOLOv8-Pose 进行推理
  3. 提取 17 个关键点坐标，并进行归一化（0~1）
  4. 将逐帧数据组装为 SkeletonTrack 和 TrackSet
"""

import os
import cv2
from typing import List, Optional, Dict
from ultralytics import YOLO

from fightguard.contracts import (
    COCO17_KEYPOINT_NAMES,
    Keypoints,
    SkeletonTrack,
    TrackSet,
)
from fightguard.config import get_config

# ============================================================
# 模块级缓存：避免每次处理视频都重新加载模型
# ============================================================
_yolo_model = None

#def get_yolo_model():
   # """懒加载 YOLO 模型"""
   # global _yolo_model
    #if _yolo_model is None:
        #print("[INFO] 正在加载 YOLOv8-Pose 模型 (首次加载可能需要下载)...")
        # 使用轻量级的 yolov8n-pose.pt，适合 CPU 运行
        #_yolo_model = YOLO("yolov8n-pose.pt")
    #return _yolo_model

def get_yolo_model():
    """懒加载 YOLO 模型 (OpenVINO 硬件加速版)"""
    global _yolo_model
    if _yolo_model is None:
        print("[INFO] 正在加载 YOLOv8-Pose (OpenVINO 加速引擎)...")
        # 直接加载刚才导出的 openvino 模型文件夹
        # Ultralytics 会自动识别并调用 Intel GPU/NPU 进行加速
        _yolo_model = YOLO("yolov8n-pose_openvino_model", task="pose")
    return _yolo_model



# ============================================================
# 视频处理核心函数
# ============================================================

def process_video_to_trackset(
    video_path: str,
    label: int = -1,
    cfg: Optional[dict] = None,
    max_frames: Optional[int] = None
) -> Optional[TrackSet]:
    """
    读取视频，用 YOLO 提取骨骼，返回 TrackSet。
    
    参数：
        video_path : 视频文件路径
        label      : 视频标签（1=冲突，0=正常）
        cfg        : 配置字典
        max_frames : 最多处理多少帧（调试用，防止视频太长）
        
    返回：
        TrackSet 对象，如果视频读取失败或未检测到人则返回 None
    """
    if cfg is None:
        cfg = get_config()

    clip_id = os.path.splitext(os.path.basename(video_path))[0]
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频文件：{video_path}")
        return None

    # 获取视频元信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps: # 处理 NaN
        fps = 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    model = get_yolo_model()
    
    # 临时字典，用于按人员 ID 收集轨迹
    # 由于 YOLOv8 默认不带强力追踪（Tracker），这里采用简单策略：
    # 每一帧检测到的人，按其在画面中的顺序（如从左到右）分配临时 ID。
    # 复杂追踪（如 DeepSORT）可作为后续优化点。
    tracks_dict: Dict[int, SkeletonTrack] = {}

    frame_idx = 0
    
    while cap.isOpened():
        if max_frames is not None and frame_idx >= max_frames:
            break
            
        ret, frame = cap.read()
        if not ret:
            break

        # 【核心优化：切换为 ByteTrack 追踪器】
        # ByteTrack 对低分检测框更鲁棒，适合两人重叠打斗场景
        # conf=0.2 降低检测阈值，让 ByteTrack 发挥关联低分框的优势
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.2, verbose=False)
        
        if len(results) > 0 and results[0].keypoints is not None:
            track_ids = results[0].boxes.id
            if track_ids is None:
                frame_idx += 1
                continue
                
            track_ids = track_ids.int().cpu().tolist()
            
            # 提取坐标 (x, y) 和 置信度 (conf)
            kpts_xyn = results[0].keypoints.xyn.cpu().numpy()
            kpts_conf = results[0].keypoints.conf.cpu().numpy() if results[0].keypoints.conf is not None else None

            for person_idx, person_kpts in enumerate(kpts_xyn):
                if person_idx >= len(track_ids):
                    break
                    
                track_id = track_ids[person_idx]
                keypoints_dict: Keypoints = {}
                
                for i, name in enumerate(COCO17_KEYPOINT_NAMES):
                    if i < len(person_kpts):
                        x, y = person_kpts[i]
                        # 安全提取置信度
                        conf = float(kpts_conf[person_idx][i]) if kpts_conf is not None else 1.0
                        
                        if x == 0.0 and y == 0.0:
                            keypoints_dict[name] = [0.0, 0.0, 0.0]
                        else:
                            # 组装为 [x, y, conf]
                            keypoints_dict[name] = [float(x), float(y), conf]
                    else:
                        keypoints_dict[name] = [0.0, 0.0, 0.0]
                
                if track_id not in tracks_dict:
                    tracks_dict[track_id] = SkeletonTrack(track_id=track_id, role="child")
                
                tracks_dict[track_id].frames.append(frame_idx)
                tracks_dict[track_id].keypoints.append(keypoints_dict)
                
        frame_idx += 1

    cap.release()

    if not tracks_dict:
        print(f"[WARNING] 视频中未检测到任何人：{video_path}")
        return None

    # 【实战优化：时空绝对对齐】
    # YOLO 追踪的轨迹是碎片的。必须将所有轨迹用空数据填充到相同的总帧数，
    # 保证 track.keypoints[i] 永远严格对应视频的物理第 i 帧！
    empty_kp = {name: [0.0, 0.0, 0.0] for name in COCO17_KEYPOINT_NAMES}
    for track in tracks_dict.values():
        aligned_kpts = []
        for i in range(frame_idx):
            if i in track.frames:
                idx = track.frames.index(i)
                aligned_kpts.append(track.keypoints[idx])
            else:
                aligned_kpts.append(empty_kp)
        track.keypoints = aligned_kpts
        track.frames = list(range(frame_idx)) # 重写 frames 列表




    track_set = TrackSet(
        clip_id=clip_id,
        label=label,
        tracks=list(tracks_dict.values()),
        fps=fps,  # 恢复原始 FPS
        total_frames=frame_idx
    )
    return track_set
    