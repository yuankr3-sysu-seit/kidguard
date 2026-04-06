"""
debug_single_video.py — 单点爆破诊断器
========================================
递归走迷宫的“探路针”。
专门针对得分为 0.000 的漏报视频，逐帧打印底层变量，
揪出到底是哪一层逻辑（追踪、配对、置信度、状态机）杀死了真实的报警。
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from fightguard.inputs.video_source import process_video_to_trackset
from fightguard.detection.pairing import get_interaction_pairs, compute_pair_distance_at_frame
from fightguard.detection.interaction_rules import compute_directional_score, CaptainStateMachine
from fightguard.config import get_config

def run_debug():
    print("="*60)
    print(" 迷宫探路：单点爆破诊断 V_583.mp4")
    print("="*60)
    
    cfg = get_config()
    # 强行覆盖参数，保持与批量评测一致
    cfg["rules"]["proximity_window_frames"] = 2  
    cfg["rules"]["smoothing_window_frames"] = 2  
    cfg["rules"]["alert_threshold"] = 0.20
    
    # 锁定那个得分为 0.000 的漏报视频
    video_path = "D:/dataset_1/five_dataset/fight/V_583.mp4" 
    
    if not os.path.exists(video_path):
        print(f"[ERROR] 找不到视频: {video_path}")
        return

    print("[1] 正在提取骨骼 (YOLOv8 + BoT-SORT)...")
    track_set = process_video_to_trackset(video_path, label=1, cfg=cfg)
    
    if not track_set:
        print("[结论] 提取失败，YOLO 连人都没看到！")
        return
        
    print(f"[2] 提取成功！总帧数: {track_set.total_frames}, 发现轨迹数(ID): {len(track_set.tracks)}")
    for t in track_set.tracks:
        print(f"    - Track ID {t.track_id}: 存活 {len(t.frames)} 帧")
        
    pairs = get_interaction_pairs(track_set, cfg)
    if not pairs:
        print("[结论] 配对失败！画面里可能没有同时存活的两个人，或者距离太远。")
        return
        
    track_a, track_b = pairs[0]
    print(f"[3] 锁定主交互对: Track {track_a.track_id} 和 Track {track_b.track_id}")
    
    fsm = CaptainStateMachine(cfg)
    dt = 1.0 / track_set.fps
    n_frames = min(len(track_a.keypoints), len(track_b.keypoints))
    
    print("[4] 逐帧状态机流转回放:")
    print("帧号 | 距离 | 置信度抑制(γ) | A爆发(r_a) | B后退(r_phi) | 状态机阶段 | 平滑得分")
    print("-" * 75)
    
    for fi in range(n_frames):
        dist = compute_pair_distance_at_frame(track_a, track_b, fi)
        if dist is None: dist = float("inf")
        
        score_ab, det_ab = compute_directional_score(track_a, track_b, fi, cfg, dt)
        score_ba, det_ba = compute_directional_score(track_b, track_a, fi, cfg, dt)
        score_pair = max(score_ab, score_ba)
        
        is_in_event, smoothed_score = fsm.update(dist, det_ab, det_ba, score_pair)
        
        # 只打印状态机有动静的帧，或者得分不为 0 的帧
        if fsm.state > 0 or score_pair > 0.01:
            gamma = min(det_ab.get("gamma", 0), det_ba.get("gamma", 0))
            r_a = max(det_ab.get("r_a", 0), det_ba.get("r_a", 0))
            r_phi = max(det_ab.get("r_phi", 0), det_ba.get("r_phi", 0))
            print(f" {fi:3d} | {dist:.2f} |    {gamma:.3f}    |   {r_a:.3f}  |   {r_phi:.3f}   |   Stage {fsm.state}  |  {smoothed_score:.3f}")

if __name__ == "__main__":
    run_debug()