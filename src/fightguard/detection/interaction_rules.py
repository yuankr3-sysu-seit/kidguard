"""
interaction_rules.py — 交互规则与评分模块
==========================================
严格按照队长《建模公式.pdf》实现基础版规则流。

规则流程：
  1. 触发先决条件：连续W帧内两人距离 < τ_dist
  2. 动作特征提取：腕部加速度、相对接近速度、肘部角加速度、躯干倾角变化
  3. 特征归一化 → 加权评分 → 惩罚因子 → 滑窗平均 → 报警判决

关键点近似说明（记录在技术文档）：
  - Neck  → 用 nose 近似（COCO-17无颈部点）
  - Pelvis→ 用 (left_hip + right_hip) / 2 近似
"""

import math
from typing import Dict, List, Optional, Tuple

from fightguard.contracts import (
    InteractionEvent,
    SkeletonTrack,
    TrackSet,
)
from fightguard.config import get_config
from fightguard.detection.pairing import (
    euclidean_distance,
    get_proximity_frames,
    get_torso_center,
)


# ============================================================
# 第一部分：关键点近似工具
# ============================================================

def get_neck_approx(kp: dict) -> Optional[List[float]]:
    """
    获取颈部近似坐标。
    COCO-17 无颈部点，用 nose 近似。
    """
    return kp.get("nose")


def get_pelvis_approx(kp: dict) -> Optional[List[float]]:
    """
    获取骨盆近似坐标。
    COCO-17 无骨盆点，用左右髋关节中点近似。
    """
    lh = kp.get("left_hip")
    rh = kp.get("right_hip")
    if lh is None or rh is None:
        return None
    if lh == [0.0, 0.0] and rh == [0.0, 0.0]:
        return None
    return [(lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0]

def get_shoulder_scale(kp: dict) -> float:
    """
    计算肩距作为个体尺度基准 S_i。
    【阶段二重大修复】：由于 2D 视频坐标已经归一化到 [0, 1]，
    如果再除以极小的肩宽（如 0.05），会导致距离和加速度发生 20 倍以上的数学爆炸！
    因此在 2D 归一化坐标系下，强制返回 1.0，废除肩宽缩放。
    """
    return 1.0

#def get_shoulder_scale(kp: dict) -> float:
    """
    计算肩距作为个体尺度基准 S_i。
    公式：S_i = ||K_i,L_Shoulder - K_i,R_Shoulder||
    用于归一化距离，消除拍摄距离差异。
    若肩距为零（数据缺失），返回1.0避免除零。
    """
    ls = kp.get("left_shoulder")
    rs = kp.get("right_shoulder")
    if ls is None or rs is None:
        return 1.0
    dist = euclidean_distance(ls, rs)
    return dist if dist > 1e-6 else 1.0


def get_body_center_formula(kp: dict) -> Optional[List[float]]:
    """
    按公式计算人体中心点：颈部与骨盆的中点。
    公式：C_i = (K_i,Neck + K_i,Pelvis) / 2
    """
    neck   = get_neck_approx(kp)
    pelvis = get_pelvis_approx(kp)
    if neck is None or pelvis is None:
        return get_torso_center(kp)  # 降级到髋关节中点
    return [(neck[0] + pelvis[0]) / 2.0, (neck[1] + pelvis[1]) / 2.0]


# ============================================================
# 第二部分：触发先决条件
# ============================================================

def check_proximity_condition(
    track_a: SkeletonTrack,
    track_b: SkeletonTrack,
    frame_idx: int,
    tau_dist: float,
    scale_a: float,
    scale_b: float,
) -> Tuple[bool, float]:
    """
    检查单帧的归一化中心距离是否满足近身条件。

    公式：D_AB = ||C_A - C_B|| / ((S_A + S_B) / 2)
    触发条件：D_AB < τ_dist

    返回：(是否满足条件, 归一化距离值)
    """
    if frame_idx >= len(track_a.keypoints) or frame_idx >= len(track_b.keypoints):
        return False, float("inf")

    kp_a = track_a.keypoints[frame_idx]
    kp_b = track_b.keypoints[frame_idx]

    center_a = get_body_center_formula(kp_a)
    center_b = get_body_center_formula(kp_b)

    if center_a is None or center_b is None:
        return False, float("inf")

    raw_dist  = euclidean_distance(center_a, center_b)
    avg_scale = (scale_a + scale_b) / 2.0
    norm_dist = raw_dist / avg_scale if avg_scale > 1e-6 else raw_dist

    return norm_dist < tau_dist, norm_dist


def find_proximity_windows(
    track_a: SkeletonTrack,
    track_b: SkeletonTrack,
    tau_dist: float,
    window_w: int,
) -> List[Tuple[int, int]]:
    """
    找出所有满足"连续W帧内距离均小于τ_dist"的时间窗。

    公式：∀τ ∈ [t-W+1, t], D_AB^τ < τ_dist

    返回：[(start_frame, end_frame), ...] 的列表
    """
    n_frames = min(len(track_a.keypoints), len(track_b.keypoints))

    # 预计算每帧的肩距尺度
    scales_a = [get_shoulder_scale(kp) for kp in track_a.keypoints[:n_frames]]
    scales_b = [get_shoulder_scale(kp) for kp in track_b.keypoints[:n_frames]]

    # 计算每帧是否满足近身条件
    prox_flags = []
    for i in range(n_frames):
        ok, _ = check_proximity_condition(
            track_a, track_b, i, tau_dist, scales_a[i], scales_b[i]
        )
        prox_flags.append(ok)

    # 滑窗检测：连续W帧都满足才算触发
    windows = []
    i = 0
    while i <= n_frames - window_w:
        if all(prox_flags[i: i + window_w]):
            # 找到一个触发窗，继续延伸直到条件不满足
            start = i
            end   = i + window_w
            while end < n_frames and prox_flags[end]:
                end += 1
            windows.append((start, end - 1))
            i = end
        else:
            i += 1

    return windows


# ============================================================
# 第三部分：动作特征提取
# ============================================================

def compute_wrist_acceleration(
    track: SkeletonTrack,
    frame_idx: int,
    dt: float,
    side: str = "right",
) -> float:
    """
    计算施力方手腕的线加速度 a_A。

    公式：
      v_A^t = ||K_A,wrist^t - K_A,wrist^(t-1)|| / (S_A * Δt)
      a_A^t = (v_A^t - v_A^(t-1)) / Δt

    参数：
        side: "right" 或 "left"，选择哪只手腕
    """
    wrist_key = f"{side}_wrist"

    if frame_idx < 2 or frame_idx >= len(track.keypoints):
        return 0.0

    kp_t   = track.keypoints[frame_idx]
    kp_t1  = track.keypoints[frame_idx - 1]
    kp_t2  = track.keypoints[frame_idx - 2]

    w_t  = kp_t.get(wrist_key)
    w_t1 = kp_t1.get(wrist_key)
    w_t2 = kp_t2.get(wrist_key)

    if w_t is None or w_t1 is None or w_t2 is None:
            return 0.0
        
    # 修复 YOLO 视觉丢失导致的 [0.0, 0.0] 瞬移 Bug
    if w_t == [0.0, 0.0] or w_t1 == [0.0, 0.0] or w_t2 == [0.0, 0.0]:
        return 0.0


    scale = get_shoulder_scale(kp_t)

    v_t  = euclidean_distance(w_t,  w_t1) / (scale * dt)
    v_t1 = euclidean_distance(w_t1, w_t2) / (scale * dt)

    return abs(v_t - v_t1) / dt


def compute_attack_distance(
    track_a: SkeletonTrack,
    track_b: SkeletonTrack,
    frame_idx: int,
    side: str = "right",
) -> float:
    """
    计算攻击距离：施力方手腕到受力方头部的归一化距离。

    公式：d_attack = ||K_A,wrist - K_B,head|| / ((S_A + S_B) / 2)
    """
    if frame_idx >= len(track_a.keypoints) or frame_idx >= len(track_b.keypoints):
        return float("inf")

    kp_a = track_a.keypoints[frame_idx]
    kp_b = track_b.keypoints[frame_idx]

    wrist_key = f"{side}_wrist"
    wrist = kp_a.get(wrist_key)
    head  = kp_b.get("nose")  # COCO-17用nose近似head

    if wrist is None or head is None:
            return float("inf")
        
    # 修复 YOLO 视觉丢失 Bug
    if wrist == [0.0, 0.0] or head == [0.0, 0.0]:
        return float("inf")


    raw_dist  = euclidean_distance(wrist, head)
    avg_scale = (get_shoulder_scale(kp_a) + get_shoulder_scale(kp_b)) / 2.0

    return raw_dist / avg_scale if avg_scale > 1e-6 else raw_dist


def compute_relative_approach_speed(
    track_a: SkeletonTrack,
    track_b: SkeletonTrack,
    frame_idx: int,
    dt: float,
    side: str = "right",
) -> float:
    """
    计算相对接近速度 v_rel。

    公式：v_rel^t = (d_attack^(t-1) - d_attack^t) / Δt
    正值表示正在接近，负值表示远离。
    """
    if frame_idx < 1:
        return 0.0

    d_t  = compute_attack_distance(track_a, track_b, frame_idx,     side)
    d_t1 = compute_attack_distance(track_a, track_b, frame_idx - 1, side)

    if d_t == float("inf") or d_t1 == float("inf"):
        return 0.0

    return (d_t1 - d_t) / dt


def compute_elbow_angle(kp: dict, side: str = "right") -> Optional[float]:
    """
    计算肘关节角度（余弦定理）。

    公式：θ = arccos( (p1-p2)·(p3-p2) / (||p1-p2|| * ||p3-p2||) )
    其中 p1=肩, p2=肘, p3=腕
    """
    shoulder_key = f"{side}_shoulder"
    elbow_key    = f"{side}_elbow"
    wrist_key    = f"{side}_wrist"

    p1 = kp.get(shoulder_key)
    p2 = kp.get(elbow_key)
    p3 = kp.get(wrist_key)

    if p1 is None or p2 is None or p3 is None:
        return None

    # 向量 p1→p2 和 p3→p2
    v1 = [p1[0] - p2[0], p1[1] - p2[1]]
    v2 = [p3[0] - p2[0], p3[1] - p2[1]]

    dot   = v1[0] * v2[0] + v1[1] * v2[1]
    norm1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    norm2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

    if norm1 < 1e-6 or norm2 < 1e-6:
        return None

    cos_val = max(-1.0, min(1.0, dot / (norm1 * norm2)))  # 防止浮点误差越界
    return math.acos(cos_val)


def compute_elbow_angular_acceleration(
    track: SkeletonTrack,
    frame_idx: int,
    dt: float,
    side: str = "right",
) -> float:
    """
    计算肘关节角加速度 α_A。

    公式：
      ω_A^t = (θ_A^t - θ_A^(t-1)) / Δt
      α_A^t = (ω_A^t - ω_A^(t-1)) / Δt
    """
    if frame_idx < 2 or frame_idx >= len(track.keypoints):
        return 0.0

    theta_t  = compute_elbow_angle(track.keypoints[frame_idx],     side)
    theta_t1 = compute_elbow_angle(track.keypoints[frame_idx - 1], side)
    theta_t2 = compute_elbow_angle(track.keypoints[frame_idx - 2], side)

    if theta_t is None or theta_t1 is None or theta_t2 is None:
        return 0.0

    omega_t  = (theta_t  - theta_t1) / dt
    omega_t1 = (theta_t1 - theta_t2) / dt

    return abs(omega_t - omega_t1) / dt


def compute_torso_tilt_change(
    track: SkeletonTrack,
    frame_idx: int,
    n_interval: int = 3,
) -> float:
    """
    计算受力方躯干倾角短时变化量 Δφ_B。

    公式：
      V_torso = K_Neck - K_Pelvis  （颈部到骨盆的向量）
      φ = arctan2(V_torso_x, V_torso_y)  （与竖直方向夹角）
      Δφ = |φ^t - φ^(t-n)|
    """
    if frame_idx < n_interval or frame_idx >= len(track.keypoints):
        return 0.0

    def _get_tilt(kp: dict) -> Optional[float]:
        neck   = get_neck_approx(kp)
        pelvis = get_pelvis_approx(kp)
        if neck is None or pelvis is None:
            return None
        vx = neck[0] - pelvis[0]
        vy = neck[1] - pelvis[1]
        return math.atan2(vx, vy)  # 与竖直方向（y轴）的夹角

    phi_t  = _get_tilt(track.keypoints[frame_idx])
    phi_tn = _get_tilt(track.keypoints[frame_idx - n_interval])

    if phi_t is None or phi_tn is None:
        return 0.0

    return abs(phi_t - phi_tn)


# ============================================================
# 第四部分：攻击部位权重判定
# ============================================================

def get_attack_part_weight(attack_dist_to_parts: dict) -> Tuple[float, str]:
    """
    根据攻击距离判定受力部位，返回对应权重。

    权重配置（来自建模公式.pdf）：
      头部/颈部  → W_part = 1.5
      胸腹/躯干  → W_part = 1.2
      四肢       → W_part = 0.8

    参数：
        attack_dist_to_parts: {"head": dist, "torso": dist, "limbs": dist}

    返回：(权重值, 受攻击部位名称)
    """
    min_part = min(attack_dist_to_parts, key=attack_dist_to_parts.get)
    weights  = {"head": 1.5, "torso": 1.2, "limbs": 0.8}
    return weights[min_part], min_part


# ============================================================
# 第五部分：特征归一化与评分
# ============================================================

def normalize_feature(value: float, min_val: float, max_val: float) -> float:
    """
    将特征值归一化到 [0, 1]。

    公式：r_i = max(0, min(1, (x_i - min_i) / (max_i - min_i)))
    """
    if max_val <= min_val:
        return 0.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def compute_frame_score(
    track_a: SkeletonTrack,
    track_b: SkeletonTrack,
    frame_idx: int,
    cfg: dict,
    dt: float = 1.0 / 30.0,
) -> Tuple[float, dict]:
    """
    计算单帧的冲突综合得分。

    公式：
      Score_base = w1*r(a_A) + w2*r(v_rel) + w3*r(α_A) + w4*r(Δφ_B)
      P          = 1 + β * r(a_A) * W_part
      Score_final= Score_base * P

    返回：(最终得分, 各特征详情字典)
    """
    rules = cfg["rules"]

    # ── 提取四个核心特征（取左右手腕的最大值）──
    a_A     = max(
        compute_wrist_acceleration(track_a, frame_idx, dt, "right"),
        compute_wrist_acceleration(track_a, frame_idx, dt, "left"),
    )
    v_rel   = max(
        compute_relative_approach_speed(track_a, track_b, frame_idx, dt, "right"),
        compute_relative_approach_speed(track_a, track_b, frame_idx, dt, "left"),
        0.0,  # 只关心接近，远离取0
    )
    alpha_A = max(
        compute_elbow_angular_acceleration(track_a, frame_idx, dt, "right"),
        compute_elbow_angular_acceleration(track_a, frame_idx, dt, "left"),
    )
    delta_phi_B = compute_torso_tilt_change(track_b, frame_idx)

    # ── 特征归一化 ──
    # 归一化区间参考建模公式，可在 default.yaml 中扩展
    r_aA        = normalize_feature(a_A,         0.0, 50.0)
    r_vrel      = normalize_feature(v_rel,        0.0, 10.0)
    r_alphaA    = normalize_feature(alpha_A,      0.0, 100.0)
    r_deltaPhiB = normalize_feature(delta_phi_B,  0.0, 1.0)

    # ── 权重（基于 451 个样本的熵权法客观赋权）──
    # w1: 腕部线加速度 (0.4926)  - 区分度最高，体现爆发力
    # w2: 相对接近速度 (0.2130)
    # w3: 肘部角加速度 (0.1218)
    # w4: 躯干倾角变化 (0.1726)
    w1, w2, w3, w4 = 0.4926, 0.2130, 0.1218, 0.1726


    # ── 基础得分 ──
    score_base = w1 * r_aA + w2 * r_vrel + w3 * r_alphaA + w4 * r_deltaPhiB

    # ── 攻击部位权重 ──
    d_head  = compute_attack_distance(track_a, track_b, frame_idx, "right")
    d_torso = euclidean_distance(
        track_a.keypoints[frame_idx].get("right_wrist", [0, 0]),
        get_pelvis_approx(track_b.keypoints[frame_idx]) or [0, 0],
    ) if frame_idx < len(track_a.keypoints) and frame_idx < len(track_b.keypoints) else float("inf")
    d_limbs = euclidean_distance(
        track_a.keypoints[frame_idx].get("right_wrist", [0, 0]),
        track_b.keypoints[frame_idx].get("right_knee", [0, 0]),
    ) if frame_idx < len(track_a.keypoints) and frame_idx < len(track_b.keypoints) else float("inf")

    w_part, hit_part = get_attack_part_weight({
        "head": d_head, "torso": d_torso, "limbs": d_limbs
    })

    # ── 惩罚因子 ──
    beta  = 0.5  # 调节系数，可后续移入 yaml
    P     = 1.0 + beta * r_aA * w_part

    # ── 最终得分 ──
    score_final = score_base * P

    details = {
        "a_A":       round(a_A,         4),
        "v_rel":     round(v_rel,        4),
        "alpha_A":   round(alpha_A,      4),
        "delta_phi": round(delta_phi_B,  4),
        "r_aA":      round(r_aA,         4),
        "r_vrel":    round(r_vrel,        4),
        "r_alphaA":  round(r_alphaA,     4),
        "r_dphi":    round(r_deltaPhiB,  4),
        "score_base":  round(score_base,  4),
        "w_part":      w_part,
        "hit_part":    hit_part,
        "P":           round(P,           4),
        "score_final": round(score_final, 4),
    }
    return score_final, details


# ============================================================
# 第六部分：滑窗平均与报警判决
# ============================================================

def sliding_window_avg(scores: List[float], window_m: int) -> List[float]:
    """
    对得分序列做滑窗平均。

    公式：Score_final_avg^t = (1/M) * Σ Score_final^(t-m), m=0..M-1
    """
    result = []
    for i in range(len(scores)):
        start    = max(0, i - window_m + 1)
        window   = scores[start: i + 1]
        result.append(sum(window) / len(window))
    return result


# ============================================================
# 第七部分：主入口 — 对一个clip执行完整规则流
# ============================================================

def run_rules_on_clip(
    track_set: TrackSet,
    cfg: Optional[dict] = None,
) -> List[InteractionEvent]:
    """
    对一个 TrackSet（clip）执行完整的规则流，返回检测到的事件列表。

    流程：
      1. 配对 → 2. 近身窗口检测 → 3. 逐帧评分 → 4. 滑窗平均 → 5. 报警判决

    返回：InteractionEvent 列表（可能为空）
    """
    from fightguard.detection.pairing import get_interaction_pairs

    if cfg is None:
        cfg = get_config()

    rules  = cfg["rules"]
    tau_dist  = rules["proximity_threshold"]
    window_w  = rules.get("proximity_window_frames", 5)   # 连续近身帧数
    window_m  = rules.get("smoothing_window_frames",  5)   # 滑窗平均帧数
    threshold = rules.get("alert_threshold",           0.3) # 报警阈值
    dt        = 1.0 / track_set.fps

    events: List[InteractionEvent] = []

    # Step1: 配对
    pairs = get_interaction_pairs(track_set, cfg)
    if not pairs:
        # 单人clip降级处理：只用施力方自身特征评分
        # 当clip只有1人时，无法做双人配对，
        # 改为检测该人自身的爆发性动作（高腕部加速度+高肘部角加速度）
        if len(track_set.tracks) == 1:
            track_a = track_set.tracks[0]
            solo_scores = []
            for fi in range(len(track_a.keypoints)):
                if fi < 2:
                    solo_scores.append(0.0)
                    continue
                from fightguard.detection.interaction_rules import (
                    compute_wrist_acceleration,
                    compute_elbow_angular_acceleration,
                    normalize_feature,
                )
                dt = 1.0 / track_set.fps
                a_A     = max(
                    compute_wrist_acceleration(track_a, fi, dt, "right"),
                    compute_wrist_acceleration(track_a, fi, dt, "left"),
                )
                alpha_A = max(
                    compute_elbow_angular_acceleration(track_a, fi, dt, "right"),
                    compute_elbow_angular_acceleration(track_a, fi, dt, "left"),
                )
                score = 0.6 * normalize_feature(a_A, 0.0, 50.0) + \
                        0.4 * normalize_feature(alpha_A, 0.0, 100.0)
                solo_scores.append(score)

            smoothed  = sliding_window_avg(solo_scores,
                            cfg["rules"].get("smoothing_window_frames", 5))
            threshold = cfg["rules"].get("alert_threshold", 0.3)
            in_event  = False
            event_start = 0
            for idx, avg_score in enumerate(smoothed):
                if avg_score > threshold and not in_event:
                    in_event    = True
                    event_start = idx
                elif avg_score <= threshold and in_event:
                    in_event = False
                    events.append(InteractionEvent(
                        clip_id        = track_set.clip_id,
                        event_type     = "child_conflict_solo",
                        start_frame    = event_start,
                        end_frame      = idx,
                        track_ids      = [track_a.track_id],
                        score          = max(smoothed[event_start:idx]),
                        triggered_rules= ["solo_high_accel"],
                        teacher_present= False,
                        region         = "unknown",
                    ))
            if in_event:
                events.append(InteractionEvent(
                    clip_id        = track_set.clip_id,
                    event_type     = "child_conflict_solo",
                    start_frame    = event_start,
                    end_frame      = len(smoothed) - 1,
                    track_ids      = [track_a.track_id],
                    score          = max(smoothed[event_start:]),
                    triggered_rules= ["solo_high_accel"],
                    teacher_present= False,
                    region         = "unknown",
                ))
        return events



    for track_a, track_b in pairs:
        # Step2: 找近身时间窗
        prox_windows = find_proximity_windows(track_a, track_b, tau_dist, window_w)
        if not prox_windows:
            continue

        for win_start, win_end in prox_windows:
            # Step3: 逐帧计算得分（在近身窗口内）
            frame_scores = []
            frame_details = []
            for fi in range(win_start, win_end + 1):
                score, details = compute_frame_score(track_a, track_b, fi, cfg, dt)
                frame_scores.append(score)
                frame_details.append(details)

            # Step4: 滑窗平均
            smoothed = sliding_window_avg(frame_scores, window_m)

            # Step5: 报警判决 — 找连续超阈值的片段
            in_event   = False
            event_start = 0
            triggered_rules = []

            for idx, avg_score in enumerate(smoothed):
                abs_frame = win_start + idx
                if avg_score > threshold and not in_event:
                    in_event    = True
                    event_start = abs_frame
                    triggered_rules = []
                elif avg_score > threshold and in_event:
                    # 收集触发的规则描述
                    d = frame_details[idx]
                    if d["r_aA"] > 0.3:
                        triggered_rules.append("high_wrist_accel")
                    if d["r_vrel"] > 0.3:
                        triggered_rules.append("high_approach_speed")
                    if d["r_alphaA"] > 0.3:
                        triggered_rules.append("high_elbow_angular_accel")
                    if d["r_dphi"] > 0.3:
                        triggered_rules.append("torso_tilt_change")
                elif avg_score <= threshold and in_event:
                    in_event = False
                    # 生成事件
                    event = InteractionEvent(
                        clip_id        = track_set.clip_id,
                        event_type     = "child_conflict",
                        start_frame    = event_start,
                        end_frame      = abs_frame,
                        track_ids      = [track_a.track_id, track_b.track_id],
                        score          = max(smoothed[event_start - win_start:idx]),
                        triggered_rules= list(set(triggered_rules)),
                        teacher_present= False,
                        region         = "unknown",
                    )
                    events.append(event)

            
            # 处理窗口末尾仍在事件中的情况
            if in_event:
                event = InteractionEvent(
                    clip_id         = track_set.clip_id,
                    event_type      = "child_conflict",
                    start_frame     = event_start,
                    end_frame       = win_end,
                    track_ids       = [track_a.track_id, track_b.track_id],
                    score           = max(smoothed[event_start - win_start:]),
                    triggered_rules = list(set(triggered_rules)),
                    teacher_present = False,
                    region          = "unknown",
                )
                events.append(event)

    return events


# ============================================================
# 第八部分：双人角色互换 — 对称检测
# ============================================================

def run_rules_symmetric(
    track_set: TrackSet,
    cfg: Optional[dict] = None,
) -> List[InteractionEvent]:
    """
    对称地运行规则流：同时以 A攻B 和 B攻A 两种视角检测，
    取两次结果的并集，避免漏报。

    在 NTU 数据集中，"施力方"和"受力方"的角色并不总是固定的，
    对称检测能提升召回率。
    """
    from fightguard.detection.pairing import get_interaction_pairs

    if cfg is None:
        cfg = get_config()

    pairs = get_interaction_pairs(track_set, cfg)
    if not pairs:
        return []

    all_events: List[InteractionEvent] = []

    for track_a, track_b in pairs:
        # 视角1：A为施力方，B为受力方
        ts_ab = TrackSet(
            clip_id      = track_set.clip_id,
            label        = track_set.label,
            tracks       = [track_a, track_b],
            fps          = track_set.fps,
            total_frames = track_set.total_frames,
        )
        events_ab = run_rules_on_clip(ts_ab, cfg)

        # 视角2：B为施力方，A为受力方
        ts_ba = TrackSet(
            clip_id      = track_set.clip_id,
            label        = track_set.label,
            tracks       = [track_b, track_a],
            fps          = track_set.fps,
            total_frames = track_set.total_frames,
        )
        events_ba = run_rules_on_clip(ts_ba, cfg)

        # 合并去重：同一帧范围内不重复计入
        merged = _merge_events(events_ab + events_ba)
        all_events.extend(merged)

    return all_events


def _merge_events(events: List[InteractionEvent]) -> List[InteractionEvent]:
    """
    合并时间上重叠的事件，避免对称检测产生重复报警。
    策略：按 start_frame 排序，相邻事件若重叠则合并为一个。
    """
    if not events:
        return []

    events_sorted = sorted(events, key=lambda e: e.start_frame)
    merged = [events_sorted[0]]

    for ev in events_sorted[1:]:
        last = merged[-1]
        if ev.start_frame <= last.end_frame:
            # 重叠，合并：取更大的 end_frame 和更高的 score
            last.end_frame      = max(last.end_frame, ev.end_frame)
            last.score          = max(last.score, ev.score)
            last.triggered_rules = list(set(last.triggered_rules + ev.triggered_rules))
        else:
            merged.append(ev)

    return merged
