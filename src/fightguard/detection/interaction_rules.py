"""
interaction_rules.py — 交互规则与状态机模块
======================================================
严格落实队长《建模公式（引入状态机和角色交互版本）》。
核心实现：
1. 肩宽尺度归一化（物理标尺）
2. 四段式状态机的同步因果律
3. 相对接近速度强制约束
4. 骨盆速度特征
5. 置信度抑制机制
"""

import math
from typing import Dict, List, Optional, Tuple, Set

from fightguard.contracts import InteractionEvent, SkeletonTrack, TrackSet
from fightguard.config import get_config
from fightguard.detection.math_utils import (
    euclidean_distance,
    get_neck_approx,
    get_pelvis_approx,
    get_shoulder_scale,
    normalize_feature,
)

# ============================================================
# 第一部分：基础几何与物理标尺计算
# ============================================================









def get_normalization_scale(track_a: SkeletonTrack, track_b: SkeletonTrack, frame_idx: int) -> float:
    """获取归一化尺度
    公式：(S_A + S_B)/2 + epsilon
    """
    if frame_idx >= len(track_a.keypoints) or frame_idx >= len(track_b.keypoints):
        return 1.0
    
    kp_a = track_a.keypoints[frame_idx]
    kp_b = track_b.keypoints[frame_idx]
    
    s_a = get_shoulder_scale(kp_a)
    s_b = get_shoulder_scale(kp_b)
    epsilon = 1e-6
    return (s_a + s_b) / 2.0 + epsilon

# ============================================================
# 第二部分：物理运动学特征提取
# ============================================================

def compute_limb_acceleration(
    track: SkeletonTrack, frame_idx: int, dt: float, part_type: str = "wrist"
) -> float:
    """计算肢体末端（手腕或脚踝）的线加速度"""
    if frame_idx < 2 or frame_idx >= len(track.keypoints):
        return 0.0
    
    kp_t = track.keypoints[frame_idx]
    kp_t1 = track.keypoints[frame_idx - 1]
    kp_t2 = track.keypoints[frame_idx - 2]
    
    max_accel = 0.0
    for side in ["left", "right"]:
        key = f"{side}_{part_type}"
        p_t = kp_t.get(key, [0.0, 0.0, 0.0])
        p_t1 = kp_t1.get(key, [0.0, 0.0, 0.0])
        p_t2 = kp_t2.get(key, [0.0, 0.0, 0.0])
        
        if p_t[:2] == [0, 0] or p_t1[:2] == [0, 0] or p_t2[:2] == [0, 0]:
            continue
        
        v_t = euclidean_distance(p_t, p_t1) / dt
        v_t1 = euclidean_distance(p_t1, p_t2) / dt
        max_accel = max(max_accel, abs(v_t - v_t1) / dt)
        
    return max_accel

def compute_joint_angular_acceleration(
    track: SkeletonTrack, frame_idx: int, dt: float, joint_type: str = "elbow"
) -> float:
    """计算关节（肘部或膝部）的角加速度"""
    if frame_idx < 2 or frame_idx >= len(track.keypoints):
        return 0.0
    
    def _get_angle(kp: dict, side: str) -> Optional[float]:
        if joint_type == "elbow":
            p1 = kp.get(f"{side}_shoulder", [0.0, 0.0, 0.0])
            p2 = kp.get(f"{side}_elbow", [0.0, 0.0, 0.0])
            p3 = kp.get(f"{side}_wrist", [0.0, 0.0, 0.0])
        else:
            p1 = kp.get(f"{side}_hip", [0.0, 0.0, 0.0])
            p2 = kp.get(f"{side}_knee", [0.0, 0.0, 0.0])
            p3 = kp.get(f"{side}_ankle", [0.0, 0.0, 0.0])
            
        if p1[:2] == [0, 0] or p2[:2] == [0, 0] or p3[:2] == [0, 0]:
            return None
        
        v1 = [p1[0] - p2[0], p1[1] - p2[1]]
        v2 = [p3[0] - p2[0], p3[1] - p2[1]]
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        norm = math.sqrt(v1[0]**2 + v1[1]**2) * math.sqrt(v2[0]**2 + v2[1]**2)
        if norm < 1e-6:
            return None
        return math.acos(max(-1.0, min(1.0, dot / norm)))

    max_alpha = 0.0
    for side in ["left", "right"]:
        theta_t = _get_angle(track.keypoints[frame_idx], side)
        theta_t1 = _get_angle(track.keypoints[frame_idx - 1], side)
        theta_t2 = _get_angle(track.keypoints[frame_idx - 2], side)
        
        if theta_t is not None and theta_t1 is not None and theta_t2 is not None:
            omega_t = (theta_t - theta_t1) / dt
            omega_t1 = (theta_t1 - theta_t2) / dt
            max_alpha = max(max_alpha, abs(omega_t - omega_t1) / dt)
            
    return max_alpha

def compute_torso_tilt_change(track: SkeletonTrack, frame_idx: int, n_interval: int = 3) -> float:
    """计算受力侧躯干倾角短时变化量 Δφ_B"""
    if frame_idx < n_interval or frame_idx >= len(track.keypoints):
        return 0.0
    
    def _get_tilt(kp: dict) -> Optional[float]:
        neck = get_neck_approx(kp)
        pelvis = get_pelvis_approx(kp)
        if neck[:2] == [0, 0] or pelvis[:2] == [0, 0]:
            return None
        return math.atan2(neck[0] - pelvis[0], neck[1] - pelvis[1])

    phi_t = _get_tilt(track.keypoints[frame_idx])
    phi_tn = _get_tilt(track.keypoints[frame_idx - n_interval])
    
    if phi_t is None or phi_tn is None:
        return 0.0
    return abs(phi_t - phi_tn)

def compute_pelvis_velocity(track: SkeletonTrack, frame_idx: int, dt: float) -> float:
    """计算骨盆速度"""
    if frame_idx < 1 or frame_idx >= len(track.keypoints):
        return 0.0
    
    pelvis_t = get_pelvis_approx(track.keypoints[frame_idx])
    pelvis_t1 = get_pelvis_approx(track.keypoints[frame_idx - 1])
    
    if pelvis_t[:2] == [0, 0] or pelvis_t1[:2] == [0, 0]:
        return 0.0
    
    return euclidean_distance(pelvis_t, pelvis_t1) / dt

def compute_attack_distance(track_a: SkeletonTrack, track_b: SkeletonTrack, frame_idx: int) -> float:
    """计算施力侧末端（手/脚）到受力侧部位的最小距离"""
    if frame_idx >= len(track_a.keypoints) or frame_idx >= len(track_b.keypoints):
        return float("inf")
    
    kp_a = track_a.keypoints[frame_idx]
    kp_b = track_b.keypoints[frame_idx]
    
    ends_a = [
        kp_a.get("left_wrist", [0.0, 0.0, 0.0]),
        kp_a.get("right_wrist", [0.0, 0.0, 0.0]),
        kp_a.get("left_ankle", [0.0, 0.0, 0.0]),
        kp_a.get("right_ankle", [0.0, 0.0, 0.0])
    ]
    
    targets_b = [
        get_neck_approx(kp_b),
        get_pelvis_approx(kp_b)
    ]
    
    min_dist = float("inf")
    for p_a in ends_a:
        if p_a[:2] == [0, 0]:
            continue
        for p_b in targets_b:
            if p_b[:2] == [0, 0]:
                continue
            min_dist = min(min_dist, euclidean_distance(p_a, p_b))
    
    return min_dist

def compute_relative_approach_speed(track_a: SkeletonTrack, track_b: SkeletonTrack, frame_idx: int, dt: float) -> float:
    """计算相对接近速度"""
    if frame_idx < 1:
        return 0.0
    
    d_t = compute_attack_distance(track_a, track_b, frame_idx)
    d_t1 = compute_attack_distance(track_a, track_b, frame_idx - 1)
    
    if d_t == float("inf") or d_t1 == float("inf"):
        return 0.0
    
    # 只关心接近（正值）
    return max(0.0, (d_t1 - d_t) / dt)

# ============================================================
# 第三部分：置信度抑制机制
# ============================================================

def compute_confidence_suppression(
    track_a: SkeletonTrack, 
    track_b: SkeletonTrack, 
    frame_idx: int,
    tau_c: float = 0.5
) -> float:
    """计算置信度抑制系数 γ
    公式：
    如果当前帧两人的平均关键点置信度 conf >= tau_c，则 gamma = 1.0；
    如果 conf < tau_c，则 gamma = conf / (tau_c + epsilon)。
    """
    if frame_idx >= len(track_a.keypoints) or frame_idx >= len(track_b.keypoints):
        return 0.0
        
    kp_a = track_a.keypoints[frame_idx]
    kp_b = track_b.keypoints[frame_idx]
    
    # 核心点 Q = {A的肩肘腕髋膝踝, B的颈骨盆头}
    a_parts = [
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    confs = []
    for part in a_parts:
        confs.append(kp_a.get(part, [0, 0, 0.0])[2])
        
    # B 的头部近似 (nose)
    confs.append(kp_b.get("nose", [0, 0, 0.0])[2])
    # B 的骨盆置信度近似
    pelvis_conf = (kp_b.get("left_hip", [0, 0, 0])[2] + kp_b.get("right_hip", [0, 0, 0])[2]) / 2.0
    confs.append(pelvis_conf)
    
    avg_c = sum(confs) / len(confs) if confs else 0.0
    epsilon = 1e-6
    
    if avg_c >= tau_c:
        return 1.0
    else:
        return avg_c / (tau_c + epsilon)

# ============================================================
# 第四部分：特征归一化
# ============================================================



# ============================================================
# 第五部分：四段式状态机
# ============================================================

class CaptainStateMachine:
    """基于队长公式的四段式状态机"""
    
    def __init__(self, cfg: dict):
        rules = cfg.get("rules", {})
        # 使用队长 PDF 中默认的物理阈值
        self.tau_dist = rules.get("tau_dist", 2.0)
        self.W = rules.get("W", 2)
        self.R = rules.get("R", 15)
        self.tau_v = rules.get("tau_v", 0.3)
        self.tau_a = rules.get("tau_a", 0.4)
        self.tau_alpha = rules.get("tau_alpha", 0.3)
        self.tau_phi = rules.get("tau_phi", 0.1)
        self.tau_p = rules.get("tau_p", 0.2)
        self.alert_threshold = rules.get("alert_threshold", 0.2)
        self.M = rules.get("M", 10)
        
        # 状态机状态
        self.state = 0  # 0: 初始状态, 1: 接近阶段, 2: 动作激活阶段, 3: 作用-响应阶段
        self.prox_buffer = 0
        self.sep_buffer = 0
        self.score_buffer = []
        self.in_event = False
    
    def update(self, dist: float, details_ab: dict, details_ba: dict, score_pair: float) -> Tuple[bool, float]:
        """状态机更新
        严格实现同步因果律，基于当前帧的瞬间同步物理量
        """
        # 0. 分离重置逻辑
        if dist > self.tau_dist:
            self.sep_buffer += 1
        else:
            self.sep_buffer = 0
            
        if self.sep_buffer >= self.R:
            self.state = 0
            self.prox_buffer = 0
            self.score_buffer.clear()
            self.in_event = False
            return False, 0.0

        # (1) 接近阶段
        if dist < self.tau_dist:
            self.prox_buffer += 1
        else:
            self.prox_buffer = 0
            
        if self.state == 0 and self.prox_buffer >= self.W:
            self.state = 1

        # (2) 动作激活阶段
        if self.state >= 1:
            # 动作激活条件：(r(v_rel) > tau_rel) OR (r(a_limb) > tau_a) OR (r(alpha_joint) > tau_alpha)
            action_ab = (
                details_ab.get("r_v", 0) > self.tau_v or
                details_ab.get("r_a", 0) > self.tau_a or
                details_ab.get("r_alpha", 0) > self.tau_alpha
            )
            
            action_ba = (
                details_ba.get("r_v", 0) > self.tau_v or
                details_ba.get("r_a", 0) > self.tau_a or
                details_ba.get("r_alpha", 0) > self.tau_alpha
            )
            
            if action_ab or action_ba:
                if self.state < 2:
                    self.state = 2

        # (3) 作用-响应阶段 —— 必须严格同步
        if self.state >= 2:
            # 作用-响应条件：(r(a_limb) > tau_a) AND (r(v_rel) > tau_rel) AND ((r(delta_phi) > tau_phi) OR (r(v_pelvis) > tau_p))
            response_ab = (
                details_ab.get("r_a", 0) > self.tau_a and
                details_ab.get("r_v", 0) > self.tau_v and
                (details_ab.get("r_phi", 0) > self.tau_phi or details_ab.get("r_p", 0) > self.tau_p)
            )
            
            response_ba = (
                details_ba.get("r_a", 0) > self.tau_a and
                details_ba.get("r_v", 0) > self.tau_v and
                (details_ba.get("r_phi", 0) > self.tau_phi or details_ba.get("r_p", 0) > self.tau_p)
            )
            
            if response_ab or response_ba:
                self.state = 3  # 物理链条闭环

        # (4) 事件确认阶段
        self.score_buffer.append(score_pair)
        if len(self.score_buffer) > self.M:
            self.score_buffer.pop(0)
            
        smoothed_score = sum(self.score_buffer) / len(self.score_buffer) if self.score_buffer else 0.0

        if self.state == 3 and smoothed_score > self.alert_threshold:
            self.in_event = True
        elif self.state < 3:
            self.in_event = False
        
        return self.in_event, smoothed_score

# ============================================================
# 第六部分：主入口 — 端到端规则流
# ============================================================

def compute_directional_score(
    track_a: SkeletonTrack, track_b: SkeletonTrack, frame_idx: int, cfg: dict, dt: float
) -> Tuple[float, dict]:
    """计算单向 (A -> B) 的综合得分与特征详情"""
    # 1. 计算归一化尺度
    norm_scale = get_normalization_scale(track_a, track_b, frame_idx)
    
    # 2. 提取物理特征 (涵盖上下肢)
    a_limb = max(
        compute_limb_acceleration(track_a, frame_idx, dt, "wrist"),
        compute_limb_acceleration(track_a, frame_idx, dt, "ankle")
    ) / norm_scale  # 归一化
    
    v_rel = compute_relative_approach_speed(track_a, track_b, frame_idx, dt) / norm_scale  # 归一化
    
    alpha_joint = max(
        compute_joint_angular_acceleration(track_a, frame_idx, dt, "elbow"),
        compute_joint_angular_acceleration(track_a, frame_idx, dt, "knee")
    )
    
    delta_phi_B = compute_torso_tilt_change(track_b, frame_idx)
    v_pelvis_B = compute_pelvis_velocity(track_b, frame_idx, dt) / norm_scale  # 归一化
    
    # 3. 归一化特征
    r_a = normalize_feature(a_limb, 0.0, 5.0)  # 使用合理的物理初始值
    r_v = normalize_feature(v_rel, 0.0, 1.0)
    r_alpha = normalize_feature(alpha_joint, 0.0, 10.0)
    r_phi = normalize_feature(delta_phi_B, 0.0, 0.5)
    r_p = normalize_feature(v_pelvis_B, 0.0, 1.0)
    
    # 4. 计算基础得分
    # 使用等权重作为默认配置
    score_base = (r_a + r_v + r_alpha + r_phi + r_p) / 5.0
    
    # 5. 置信度抑制系数（从配置读取 tau_c，支持 Optuna 调参）
    tau_c = cfg.get("rules", {}).get("tau_c", 0.5)
    gamma_t = compute_confidence_suppression(track_a, track_b, frame_idx, tau_c=tau_c)
    
    # 6. 最终单向得分
    score_final = gamma_t * score_base
    
    details = {
        "r_a": r_a, "r_v": r_v, "r_alpha": r_alpha, "r_phi": r_phi, "r_p": r_p,
        "gamma": gamma_t
    }
    return score_final, details

def run_rules_on_clip(
    track_set: TrackSet, cfg: Optional[dict] = None
) -> List[InteractionEvent]:
    from fightguard.detection.pairing import get_interaction_pairs, compute_pair_distance_at_frame
    
    if cfg is None:
        cfg = get_config()
    
    rules = cfg["rules"]
    tau_dist = rules.get("tau_dist", 2.0)
    alert_threshold = rules.get("alert_threshold", 0.2)
    dt = 1.0 / track_set.fps
    
    events: List[InteractionEvent] = []
    pairs = get_interaction_pairs(track_set, cfg)
    if not pairs:
        return events
    
    for track_a, track_b in pairs:
        n_frames = min(len(track_a.keypoints), len(track_b.keypoints))
        
        # 实例化严格遵循队长 PDF 的状态机
        fsm = CaptainStateMachine(cfg)

        event_start = 0
        max_event_score = 0.0
        triggered_rules = set()
        
        for fi in range(n_frames):
            # 1. 计算距离
            dist = compute_pair_distance_at_frame(track_a, track_b, fi)
            if dist is None:
                dist = float("inf")
            
            # 2. 双向评分计算
            score_ab, det_ab = compute_directional_score(track_a, track_b, fi, cfg, dt)
            score_ba, det_ba = compute_directional_score(track_b, track_a, fi, cfg, dt)
            
            # 3. 取主导分数
            score_pair = max(score_ab, score_ba)
            
            # 4. 状态机更新
            was_in_event = fsm.in_event
            is_in_event, smoothed_score = fsm.update(dist, det_ab, det_ba, score_pair)

            # 5. 事件生成逻辑
            if is_in_event and not was_in_event:
                event_start = fi
                max_event_score = smoothed_score
                triggered_rules.clear()
            
            elif is_in_event:
                max_event_score = max(max_event_score, smoothed_score)
                # 收集触发规则
                dominant_det = det_ab if score_ab > score_ba else det_ba
                if dominant_det["r_a"] > 0.3:
                    triggered_rules.add("high_limb_accel")
                if dominant_det["r_alpha"] > 0.3:
                    triggered_rules.add("high_joint_angular_accel")
                if dominant_det["r_phi"] > 0.3:
                    triggered_rules.add("torso_tilt_change")
                if dominant_det["r_p"] > 0.3:
                    triggered_rules.add("pelvis_velocity_change")
                if dominant_det["gamma"] < 0.5:
                    triggered_rules.add("low_confidence_suppressed")
            
            elif not is_in_event and was_in_event:
                events.append(InteractionEvent(
                    clip_id=track_set.clip_id,
                    event_type="child_conflict",
                    start_frame=event_start,
                    end_frame=max(0, fi - fsm.R), # 剔除分开的重置帧
                    track_ids=[track_a.track_id, track_b.track_id],
                    score=round(max_event_score, 3),
                    triggered_rules=list(triggered_rules),
                    teacher_present=False,
                    region="unknown"
                ))
                
        # 处理结尾仍在冲突中的情况
        if fsm.in_event:
            events.append(InteractionEvent(
                clip_id=track_set.clip_id,
                event_type="child_conflict",
                start_frame=event_start,
                end_frame=n_frames - 1,
                track_ids=[track_a.track_id, track_b.track_id],
                score=round(max_event_score, 3),
                triggered_rules=list(triggered_rules),
                teacher_present=False,
                region="unknown"
            ))
            
    return events

# ============================================================
# 兼容旧接口
# ============================================================
def run_rules_symmetric(track_set: TrackSet, cfg: Optional[dict] = None) -> List[InteractionEvent]:
    """兼容旧接口"""
    return run_rules_on_clip(track_set, cfg)

def _merge_events(events: List[InteractionEvent]) -> List[InteractionEvent]:
    """兼容旧接口"""
    return events

def compute_frame_score(track_a: SkeletonTrack, track_b: SkeletonTrack, frame_idx: int, cfg: dict, dt: float) -> Tuple[float, dict]:
    """兼容旧接口：计算单帧的得分与特征详情"""
    # 调用现有的compute_directional_score函数
    score, details = compute_directional_score(track_a, track_b, frame_idx, cfg, dt)
    
    # 转换返回的字典键，以兼容旧接口
    old_style_details = {
        "a_A": details.get("r_a", 0.0),
        "v_rel": details.get("r_v", 0.0),
        "alpha_A": details.get("r_alpha", 0.0),
        "delta_phi": details.get("r_phi", 0.0),
        "v_pelvis": details.get("r_p", 0.0)
    }
    
    return score, old_style_details
