"""
diagnose_a049.py — A049 样本逐帧解剖脚本
==========================================
任务 C：查明 A049（推搡）为什么全部得分为 0
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from fightguard.inputs.skeleton_source import load_skeleton_file
from fightguard.detection.interaction_rules import (
    compute_directional_score, 
    CaptainStateMachine,
    get_normalization_scale
)
from fightguard.detection.pairing import get_interaction_pairs, compute_pair_distance_at_frame
from fightguard.config import get_config

def diagnose_a049(sample_path: str = None):
    print("="*80)
    print(" A049 样本逐帧解剖 (任务 C)")
    print("="*80)
    
    cfg = get_config()
    
    # 如果没有指定路径，使用第一个找到的 A049 样本
    if sample_path is None:
        sample_path = "D:/dataset_1/nturgbd_skeletons_s001_to_s017/S001C001P001R001A049.skeleton"
    
    print(f"\n样本路径: {sample_path}")
    if not os.path.exists(sample_path):
        print(f"[ERROR] 文件不存在: {sample_path}")
        print("\n请手动指定一个 A049 样本路径:")
        print("  python scripts/diagnose_a049.py <path_to_A049.skeleton>")
        return
    
    # 加载样本
    print("\n[1/6] 加载样本...")
    ts = load_skeleton_file(sample_path, cfg)
    if ts is None:
        print("[ERROR] 加载失败")
        return
    
    print(f"      Clip ID: {ts.clip_id}")
    print(f"      Label: {ts.label}")
    print(f"      总帧数: {ts.total_frames}")
    print(f"      追踪人数: {len(ts.tracks)}")
    
    if len(ts.tracks) < 2:
        print("[ERROR] 样本中人数不足 2 人，无法进行交互判定")
        return
    
    # 获取交互对
    print("\n[2/6] 获取交互对...")
    pairs = get_interaction_pairs(ts, cfg)
    if not pairs:
        print("[ERROR] 未找到交互对")
        return
    
    track_a, track_b = pairs[0]
    print(f"      Track A: ID={track_a.track_id}, 帧数={len(track_a.keypoints)}")
    print(f"      Track B: ID={track_b.track_id}, 帧数={len(track_b.keypoints)}")
    
    dt = 1.0 / ts.fps
    
    # 初始化状态机
    fsm = CaptainStateMachine(cfg)
    
    # ============================================================
    # 逐帧跟踪
    # ============================================================
    print("\n[3/6] 逐帧跟踪分析...")
    print("="*80)
    
    n_frames = min(len(track_a.keypoints), len(track_b.keypoints))
    
    # 统计信息
    state_changes = []
    teleport_frames = []
    response_frames = []
    max_score = 0.0
    max_score_frame = 0
    
    print(f"\n{'帧':>4s} | {'距离':>6s} | {'肩宽':>6s} | "
          f"{'r_a':>5s} {'r_v':>5s} {'r_phi':>6s} {'r_p':>5s} {'gamma':>5s} | "
          f"{'Score':>6s} | {'State':>5s} | {'Event':>5s} | 备注")
    print("-" * 110)
    
    for fi in range(n_frames):
        # 1. 计算距离
        dist = compute_pair_distance_at_frame(track_a, track_b, fi)
        if dist is None:
            dist = float("inf")
        
        # 2. 计算肩宽
        shoulder_scale = get_normalization_scale(track_a, track_b, fi)
        
        # 3. 双向评分
        score_ab, det_ab = compute_directional_score(track_a, track_b, fi, cfg, dt)
        score_ba, det_ba = compute_directional_score(track_b, track_a, fi, cfg, dt)
        score_pair = max(score_ab, score_ba)
        
        dominant_det = det_ab if score_ab > score_ba else det_ba
        
        # 4. 状态机更新
        was_in_event = fsm.in_event
        was_state = fsm.state
        is_in_event, smoothed_score = fsm.update(dist, det_ab, det_ba, score_pair)
        
        # 记录最大得分
        if score_pair > max_score:
            max_score = score_pair
            max_score_frame = fi
        
        # 记录状态变化
        if fsm.state != was_state:
            state_changes.append((fi, was_state, fsm.state))
        
        # 检查是否被反瞬移过滤
        is_teleport = (det_ab["r_a"] == 0 and det_ab["r_v"] == 0 and 
                      det_ab.get("gamma", 1) > 0)  # gamma 正常但 r_a/r_v 被清零
        if is_teleport:
            teleport_frames.append(fi)
        
        # 检查是否满足作用-响应
        response_ab = (
            det_ab.get("r_a", 0) > fsm.tau_a and
            det_ab.get("r_v", 0) > fsm.tau_v and
            (det_ab.get("r_phi", 0) > fsm.tau_phi or det_ab.get("r_p", 0) > fsm.tau_p)
        )
        if response_ab:
            response_frames.append(fi)
        
        # 打印关键帧（每 10 帧或状态变化帧）
        if fi % 10 == 0 or fsm.state != was_state or fi < 5:
            note = ""
            if fsm.state != was_state:
                note = f"状态变化 {was_state}→{fsm.state}"
            elif fi in teleport_frames:
                note = "⚠️ 反瞬移过滤"
            
            print(f"{fi:4d} | {dist:6.3f} | {shoulder_scale:6.4f} | "
                  f"{dominant_det['r_a']:5.3f} {dominant_det['r_v']:5.3f} "
                  f"{dominant_det['r_phi']:6.2f} {dominant_det['r_p']:5.3f} "
                  f"{dominant_det['gamma']:5.3f} | "
                  f"{score_pair:6.3f} | {fsm.state:5d} | {'Y' if is_in_event else 'N':5s} | {note}")
    
    # ============================================================
    # 诊断结论
    # ============================================================
    print("\n" + "="*80)
    print(" [4/6] 诊断结论")
    print("="*80)
    
    print(f"\n  最大得分: {max_score:.4f} (出现在第 {max_score_frame} 帧)")
    print(f"  最终事件数: {'有' if fsm.in_event else '无'}")
    
    print(f"\n  状态变化历史:")
    if state_changes:
        for fi, old_state, new_state in state_changes:
            state_names = ["初始", "接近", "动作激活", "作用-响应"]
            print(f"    帧 {fi:4d}: {state_names[old_state]} → {state_names[new_state]}")
    else:
        print(f"    ⚠️  全程无状态变化，始终停留在状态 0")
    
    print(f"\n  反瞬移过滤:")
    if teleport_frames:
        print(f"    ⚠️  共 {len(teleport_frames)} 帧被反瞬移过滤")
        print(f"    帧号: {teleport_frames[:10]}{'...' if len(teleport_frames) > 10 else ''}")
    else:
        print(f"    ✅ 无反瞬移过滤")
    
    print(f"\n  作用-响应帧:")
    if response_frames:
        print(f"    ✅ 共 {len(response_frames)} 帧满足作用-响应条件")
        print(f"    帧号: {response_frames[:10]}{'...' if len(response_frames) > 10 else ''}")
    else:
        print(f"    ⚠️  无任何帧满足作用-响应条件")
    
    # ============================================================
    # 根因判断
    # ============================================================
    print(f"\n[5/6] 根因分析...")
    print("-" * 80)
    
    root_causes = []
    
    # 检查 1：是否从未进入接近状态
    if not any(sc[2] >= 1 for sc in state_changes):
        root_causes.append(("一级", "从未进入接近状态", 
                           f"两人距离始终 > tau_dist ({fsm.tau_dist:.2f})"))
    
    # 检查 2：是否从未进入动作激活状态
    if not any(sc[2] >= 2 for sc in state_changes):
        root_causes.append(("一级", "从未进入动作激活状态",
                           "r_a、r_v、r_alpha 均未超过阈值"))
    
    # 检查 3：是否从未进入作用-响应状态
    if not any(sc[2] >= 3 for sc in state_changes):
        root_causes.append(("一级", "从未进入作用-响应状态",
                           "同步 AND 条件未满足 或 时间窗确认未通过"))
    
    # 检查 4：是否被反瞬移过滤
    if teleport_frames:
        root_causes.append(("二级", "反瞬移过滤可能过严",
                           f"{len(teleport_frames)} 帧被过滤，可能误杀了真实动作"))
    
    # 检查 5：作用-响应帧数不足
    if response_frames and len(response_frames) < fsm.min_confirm_frames:
        root_causes.append(("二级", "作用-响应帧数不足",
                           f"仅 {len(response_frames)} 帧满足，需要 {fsm.min_confirm_frames} 帧"))
    
    # 检查 6：得分过低
    if max_score < cfg["rules"]["alert_threshold"]:
        root_causes.append(("一级", "最高得分低于报警阈值",
                           f"max_score={max_score:.4f} < alert_threshold={cfg['rules']['alert_threshold']:.4f}"))
    
    if root_causes:
        for level, cause, detail in root_causes:
            print(f"  [{level}根因] {cause}")
            print(f"    详情: {detail}")
    else:
        print(f"  ✅ 未发现明显异常，需进一步调查")
    
    # ============================================================
    # 最终判断
    # ============================================================
    print(f"\n[6/6] 最终判断")
    print("="*80)
    
    if max_score == 0.0:
        print(f"\n  ❌ A049 得分为 0 的直接原因:")
        if not response_frames:
            print(f"     所有帧的特征值 (r_a, r_v, r_phi, r_p) 均未超过阈值")
            print(f"     可能原因:")
            print(f"       1. NTU 归一化后的特征尺度与阈值不匹配")
            print(f"       2. 肩宽归一化导致特征值被过度压缩")
            print(f"       3. 帧内 min-max 归一化破坏了绝对尺度")
        elif teleport_frames:
            print(f"     作用-响应帧存在，但被反瞬移过滤清零")
        else:
            print(f"     未知原因，需检查特征计算过程")
    else:
        print(f"\n  ⚠️  A049 最高得分 {max_score:.4f}，但未触发事件")
        print(f"     可能原因:")
        if not any(sc[2] >= 3 for sc in state_changes):
            print(f"       - 状态机未进入作用-响应阶段 (State 3)")
        if len(response_frames) < fsm.min_confirm_frames:
            print(f"       - 时间窗确认未通过 ({len(response_frames)} < {fsm.min_confirm_frames})")
        if max_score < cfg["rules"]["alert_threshold"]:
            print(f"       - 得分低于阈值 ({max_score:.4f} < {cfg['rules']['alert_threshold']:.4f})")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        diagnose_a049(sys.argv[1])
    else:
        diagnose_a049()
