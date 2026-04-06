"""
interaction_rules.py — 交互规则与状态机模块 (进阶版)
======================================================
严格落实队长《建模公式（引入状态机和角色交互版本）》。
核心升级：
1. 置信度抑制 (Confidence Suppression)：解决 YOLO 遮挡噪点
2. 综合末端特征：涵盖上肢(手腕)与下肢(脚踝)的攻击
3. 双向角色评分：不预设施力方，计算 A->B 和 B->A
4. 四段式状态机：接近 -> 动作激活 -> 作用响应 -> 确认
"""

import math
from typing import Dict, List, Optional, Tuple, Set

from fightguard.contracts import InteractionEvent, SkeletonTrack, TrackSet
from fightguard.config import get_config

# ============================================================
# 第一部分：基础几何与近似工具 (适配 [x, y, conf])
# ============================================================

def euclidean_distance(p1: List[float], p2: List[float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def get_neck_approx(kp: dict) -> List[float]:
    return kp.get("nose", [0.0, 0.0, 0.0])

def get_pelvis_approx(kp: dict) -> List[float]:
    lh = kp.get("left_hip", [0.0, 0.0, 0.0])
    rh = kp.get("right_hip", [0.0, 0.0, 0.0])
    if lh[:2] == [0.0, 0.0] and rh[:2] == [0.0, 0.0]:
        return [0.0, 0.0, 0.0]
    return [(lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0, (lh[2] + rh[2]) / 2.0]

def get_shoulder_scale(kp: dict) -> float:
    # 2D 视频中彻底废除肩宽缩放，避免透视畸变导致数学爆炸
    return 1.0

def get_body_center_formula(kp: dict) -> Optional[List[float]]:
    neck = get_neck_approx(kp)
    pelvis = get_pelvis_approx(kp)
    if neck[:2] == [0.0, 0.0] or pelvis[:2] == [0.0, 0.0]:
        return None
    return [(neck[0] + pelvis[0]) / 2.0, (neck[1] + pelvis[1]) / 2.0]

# ============================================================
# 第二部分：进阶版核心 - 置信度抑制 (Confidence Suppression)
# ============================================================

def compute_confidence_suppression(
    track_a: SkeletonTrack, 
    track_b: SkeletonTrack, 
    frame_idx: int,
    tau_c: float = 0.5
) -> float:
    """
    计算 A -> B 方向的置信度抑制系数 γ_t。
    核心点 Q = {A的肩肘腕髋膝踝, B的颈骨盆头}
    当 YOLO 发生严重遮挡或跟丢时，此函数将直接把危险得分压制为 0！
    """
    if frame_idx >= len(track_a.keypoints) or frame_idx >= len(track_b.keypoints):
        return 0.0
        
    kp_a = track_a.keypoints[frame_idx]
    kp_b = track_b.keypoints[frame_idx]
    
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
    pelvis_conf = (kp_b.get("left_hip", [0,0,0])[2] + kp_b.get("right_hip", [0,0,0])[2]) / 2.0
    confs.append(pelvis_conf)
    
    avg_c = sum(confs) / len(confs) if confs else 0.0
    
    if avg_c >= tau_c:
        return 1.0
    else:
        return avg_c / (tau_c + 1e-6)

# ============================================================
# 第三部分：物理运动学特征提取 (涵盖上下肢)
# ============================================================

def compute_limb_acceleration(
    track: SkeletonTrack, frame_idx: int, dt: float, part_type: str = "wrist"
) -> float:
    """
    计算肢体末端（手腕或脚踝）的线加速度。
    """
    if frame_idx < 2 or frame_idx >= len(track.keypoints): return 0.0
    
    kp_t   = track.keypoints[frame_idx]
    kp_t1  = track.keypoints[frame_idx - 1]
    kp_t2  = track.keypoints[frame_idx - 2]
    
    max_accel = 0.0
    for side in ["left", "right"]:
        key = f"{side}_{part_type}"
        p_t, p_t1, p_t2 = kp_t.get(key), kp_t1.get(key), kp_t2.get(key)
        
        if not (p_t and p_t1 and p_t2): continue
        if p_t[:2] == [0,0] or p_t1[:2] == [0,0] or p_t2[:2] == [0,0]: continue
        
        v_t  = euclidean_distance(p_t, p_t1) / dt
        v_t1 = euclidean_distance(p_t1, p_t2) / dt
        max_accel = max(max_accel, abs(v_t - v_t1) / dt)
        
    return max_accel

def compute_joint_angular_acceleration(
    track: SkeletonTrack, frame_idx: int, dt: float, joint_type: str = "elbow"
) -> float:
    """
    计算关节（肘部或膝部）的角加速度。
    """
    if frame_idx < 2 or frame_idx >= len(track.keypoints): return 0.0
    
    def _get_angle(kp: dict, side: str) -> Optional[float]:
        if joint_type == "elbow":
            p1, p2, p3 = kp.get(f"{side}_shoulder"), kp.get(f"{side}_elbow"), kp.get(f"{side}_wrist")
        else:
            p1, p2, p3 = kp.get(f"{side}_hip"), kp.get(f"{side}_knee"), kp.get(f"{side}_ankle")
            
        if not (p1 and p2 and p3): return None
        if p1[:2] == [0,0] or p2[:2] == [0,0] or p3[:2] == [0,0]: return None
        
        v1 = [p1[0] - p2[0], p1[1] - p2[1]]
        v2 = [p3[0] - p2[0], p3[1] - p2[1]]
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        norm = math.sqrt(v1[0]**2 + v1[1]**2) * math.sqrt(v2[0]**2 + v2[1]**2)
        if norm < 1e-6: return None
        return math.acos(max(-1.0, min(1.0, dot / norm)))

    max_alpha = 0.0
    for side in ["left", "right"]:
        theta_t  = _get_angle(track.keypoints[frame_idx], side)
        theta_t1 = _get_angle(track.keypoints[frame_idx - 1], side)
        theta_t2 = _get_angle(track.keypoints[frame_idx - 2], side)
        
        if theta_t is not None and theta_t1 is not None and theta_t2 is not None:
            omega_t  = (theta_t - theta_t1) / dt
            omega_t1 = (theta_t1 - theta_t2) / dt
            max_alpha = max(max_alpha, abs(omega_t - omega_t1) / dt)
            
    return max_alpha

def compute_torso_tilt_change(track: SkeletonTrack, frame_idx: int, n_interval: int = 3) -> float:
    """计算受力侧躯干倾角短时变化量 Δφ_B"""
    if frame_idx < n_interval or frame_idx >= len(track.keypoints): return 0.0
    
    def _get_tilt(kp: dict) -> Optional[float]:
        neck, pelvis = get_neck_approx(kp), get_pelvis_approx(kp)
        if neck[:2] == [0,0] or pelvis[:2] == [0,0]: return None
        return math.atan2(neck[0] - pelvis[0], neck[1] - pelvis[1])

    phi_t  = _get_tilt(track.keypoints[frame_idx])
    phi_tn = _get_tilt(track.keypoints[frame_idx - n_interval])
    
    if phi_t is None or phi_tn is None: return 0.0
    return abs(phi_t - phi_tn)

def compute_attack_distance(track_a: SkeletonTrack, track_b: SkeletonTrack, frame_idx: int) -> float:
    """计算施力侧末端（手/脚）到受力侧部位的最小距离"""
    if frame_idx >= len(track_a.keypoints) or frame_idx >= len(track_b.keypoints): return float("inf")
    kp_a, kp_b = track_a.keypoints[frame_idx], track_b.keypoints[frame_idx]
    
    ends_a = [kp_a.get("left_wrist"), kp_a.get("right_wrist"), kp_a.get("left_ankle"), kp_a.get("right_ankle")]
    targets_b = [get_neck_approx(kp_b), get_pelvis_approx(kp_b)]
    
    min_dist = float("inf")
    for p_a in ends_a:
        if not p_a or p_a[:2] == [0,0]: continue
        for p_b in targets_b:
            if not p_b or p_b[:2] == [0,0]: continue
            min_dist = min(min_dist, euclidean_distance(p_a, p_b))
    return min_dist

def compute_relative_approach_speed(track_a: SkeletonTrack, track_b: SkeletonTrack, frame_idx: int, dt: float) -> float:
    """计算相对接近速度"""
    if frame_idx < 1: return 0.0
    d_t  = compute_attack_distance(track_a, track_b, frame_idx)
    d_t1 = compute_attack_distance(track_a, track_b, frame_idx - 1)
    if d_t == float("inf") or d_t1 == float("inf"): return 0.0
    return max(0.0, (d_t1 - d_t) / dt) # 只关心接近（正值）



# ============================================================
# 第四部分：特征归一化与双向评分模型
# ============================================================

def normalize_feature(value: float, min_val: float, max_val: float) -> float:
    if max_val <= min_val: return 0.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

def compute_directional_score(
    track_a: SkeletonTrack, track_b: SkeletonTrack, frame_idx: int, cfg: dict, dt: float
) -> Tuple[float, dict]:
    """
    计算单向 (A -> B) 的综合得分与特征详情。
    严格落实置信度抑制和熵权法客观权重。
    """
    # 1. 提取物理特征 (涵盖上下肢)
    a_limb = max(
        compute_limb_acceleration(track_a, frame_idx, dt, "wrist"),
        compute_limb_acceleration(track_a, frame_idx, dt, "ankle")
    )
    v_rel = compute_relative_approach_speed(track_a, track_b, frame_idx, dt)
    alpha_joint = max(
        compute_joint_angular_acceleration(track_a, frame_idx, dt, "elbow"),
        compute_joint_angular_acceleration(track_a, frame_idx, dt, "knee")
    )
    delta_phi_B = compute_torso_tilt_change(track_b, frame_idx)
    
    # 2. 归一化 (区间上限可根据 EDA 结果微调)
    r_a = normalize_feature(a_limb, 0.0, 50.0)
    r_v = normalize_feature(v_rel, 0.0, 10.0)
    r_alpha = normalize_feature(alpha_joint, 0.0, 100.0)
    r_phi = normalize_feature(delta_phi_B, 0.0, 1.0)
    
    # 3. 熵权法客观权重 (来自阶段A数据驱动结果)
    w1, w2, w3, w4 = 0.4926, 0.2130, 0.1218, 0.1726
    score_base = w1 * r_a + w2 * r_v + w3 * r_alpha + w4 * r_phi
    
    # 4. 攻击部位惩罚因子 (简化近似：手腕到头部的距离)
    kp_a, kp_b = track_a.keypoints[frame_idx], track_b.keypoints[frame_idx]
    w_part = 1.0
    if frame_idx < len(track_a.keypoints) and frame_idx < len(track_b.keypoints):
        head_b = get_neck_approx(kp_b)
        wrist_a = kp_a.get("right_wrist", [0.0, 0.0, 0.0])
        if head_b[:2] != [0.0, 0.0] and wrist_a[:2] != [0.0, 0.0]:
            dist = euclidean_distance(wrist_a, head_b)
            if dist < 0.2: w_part = 1.5
            elif dist < 0.4: w_part = 1.2
            else: w_part = 0.8
            
    beta = 0.5
    P_t = 1.0 + beta * r_a * w_part
    
    # 5. 置信度抑制系数 (核心抗噪点机制)
    gamma_t = compute_confidence_suppression(track_a, track_b, frame_idx, tau_c=0.5)
    
    # 6. 最终单向得分
    score_final = gamma_t * score_base * P_t
    
    details = {
        "r_a": r_a, "r_v": r_v, "r_alpha": r_alpha, "r_phi": r_phi,
        "gamma": gamma_t, "P": P_t
    }
    return score_final, details

# ============================================================
# 第五部分：四段式状态机与事件确认 (FSM)
# ============================================================

# ============================================================
# 第五部分：实战优化版四段式状态机 (FSM) - 异步响应记忆
# ============================================================

class CaptainStateMachine:
    """
    基于队长公式的实战优化版。
    核心改进：引入“异步记忆缓存”，解决真实物理世界中“施力”与“受力”的时间差问题。
    """
    
    def __init__(self, cfg: dict):
        rules = cfg.get("rules", {})
        # 【实战优化】：将所有硬编码阈值暴露给 rules 字典，供 Optuna 动态调参
        self.tau_dist = rules.get("tau_dist", 2.0)       
        self.W = rules.get("W", 2)                
        self.R = rules.get("R", 15)               
        self.tau_v = rules.get("tau_v", 0.31)          
        self.tau_a = rules.get("tau_a", 0.40)          
        self.tau_alpha = rules.get("tau_alpha", 0.3)      
        self.tau_phi = rules.get("tau_phi", 0.11)        
        self.alert_threshold = rules.get("alert_threshold", 0.24)
        self.M = rules.get("M", 10)                
        
        self.state = 0            
        self.prox_buffer = 0      
        self.sep_buffer = 0       
        self.score_buffer = []    
        self.in_event = False     
        
        self.action_memory_ab = 0 
        self.action_memory_ba = 0 
        self.memory_window = rules.get("memory_window", 19) # 暴露异步记忆窗口

    

    def update(self, dist: float, details_ab: dict, details_ba: dict, score_pair: float) -> Tuple[bool, float]:
        # 0. 异步记忆衰减
        if self.action_memory_ab > 0: self.action_memory_ab -= 1
        if self.action_memory_ba > 0: self.action_memory_ba -= 1

        # (4) 冻结/重置逻辑
        if dist > self.tau_dist:
            self.sep_buffer += 1
        else:
            self.sep_buffer = 0
            
        if self.sep_buffer >= self.R:
            self.state = 0
            self.prox_buffer = 0
            self.score_buffer.clear()
            self.in_event = False
            self.action_memory_ab = 0
            self.action_memory_ba = 0
            return False, 0.0

        # (1) 接近阶段
        if dist < self.tau_dist:
            self.prox_buffer += 1
        else:
            self.prox_buffer = 0
            
        if self.state == 0 and self.prox_buffer >= self.W:
            self.state = 1

        # (2) 动作激活阶段 & 记录攻击动作
        if self.state >= 1:
            # 如果 A 有高爆发，记住 A 的攻击
            if details_ab.get("r_v", 0) > self.tau_v or details_ab.get("r_a", 0) > self.tau_a or details_ab.get("r_alpha", 0) > self.tau_alpha:
                self.action_memory_ab = self.memory_window
                if self.state < 2: self.state = 2
                
            # 如果 B 有高爆发，记住 B 的攻击
            if details_ba.get("r_v", 0) > self.tau_v or details_ba.get("r_a", 0) > self.tau_a or details_ba.get("r_alpha", 0) > self.tau_alpha:
                self.action_memory_ba = self.memory_window
                if self.state < 2: self.state = 2

        # (3) 作用-响应阶段 (实战版：异步链条)
        if self.state >= 2:
            response_formed = False
            # A 之前攻击过 (记忆仍在)，且现在 B 身体发生倾斜
            if self.action_memory_ab > 0 and details_ab.get("r_phi", 0) > self.tau_phi:
                response_formed = True
            # B 之前攻击过 (记忆仍在)，且现在 A 身体发生倾斜
            if self.action_memory_ba > 0 and details_ba.get("r_phi", 0) > self.tau_phi:
                response_formed = True
                
            if response_formed:
                self.state = 3  # 物理链条彻底闭环！

        # (5) 事件确认阶段
        self.score_buffer.append(score_pair)
        if len(self.score_buffer) > self.M:
            self.score_buffer.pop(0)
            
        smoothed_score = sum(self.score_buffer) / len(self.score_buffer) if self.score_buffer else 0.0

        if self.state == 3 and smoothed_score > self.alert_threshold:
            self.in_event = True
        
        return self.in_event, smoothed_score



# ============================================================
# 第六部分：主入口 — 端到端规则流
# ============================================================

def run_rules_on_clip(
    track_set: TrackSet, cfg: Optional[dict] = None
) -> List[InteractionEvent]:
    from fightguard.detection.pairing import get_interaction_pairs, compute_pair_distance_at_frame
    
    if cfg is None: cfg = get_config()
    
    rules = cfg["rules"]
    tau_dist = 2.0  # 2D 视频中放宽空间封印，由状态机主导
    alert_threshold = rules.get("alert_threshold", 0.20)
    dt = 1.0 / track_set.fps
    
    events: List[InteractionEvent] = []
    pairs = get_interaction_pairs(track_set, cfg)
    if not pairs: return events
    
    for track_a, track_b in pairs:
        n_frames = min(len(track_a.keypoints), len(track_b.keypoints))
        
        # 实例化严格遵循队长 PDF 的状态机
        fsm = CaptainStateMachine(cfg)

        event_start = 0
        max_event_score = 0.0
        triggered_rules = set()
        
        for fi in range(n_frames):
            # 1. 计算归一化距离
            dist = compute_pair_distance_at_frame(track_a, track_b, fi)
            if dist is None: dist = float("inf")
            
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
                if dominant_det["r_a"] > 0.3: triggered_rules.add("high_limb_accel")
                if dominant_det["r_alpha"] > 0.3: triggered_rules.add("high_joint_angular_accel")
                if dominant_det["r_phi"] > 0.3: triggered_rules.add("torso_tilt_change")
                if dominant_det["gamma"] < 0.5: triggered_rules.add("low_confidence_suppressed") # 记录被抑制的情况
                
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
    # 进阶版已内置双向检测，无需再反转轨迹调用
    return run_rules_on_clip(track_set, cfg)

def _merge_events(events: List[InteractionEvent]) -> List[InteractionEvent]:
    # 进阶版内置状态机，天然防碎片化，此函数保留作接口兼容
    return events
