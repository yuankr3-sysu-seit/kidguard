import sys
import os
sys.path.insert(0, "src")

from fightguard.inputs.video_source import process_video_to_trackset
from fightguard.detection.interaction_rules import run_rules_symmetric
from fightguard.config import get_config

def run_demo():
    print("="*60)
    print(" KidGuard 幼儿园冲突风险管理分析系统 - 演示程序")
    print("="*60)
    
    cfg = get_config()
    
    # ── 针对真实 2D 监控视频的专属参数覆盖 ──
    # 真实视频中掌掴动作极快，缩短时间窗口，提升系统敏锐度
    cfg["rules"]["proximity_window_frames"] = 2  # 只要连续2帧近身就触发
    cfg["rules"]["smoothing_window_frames"] = 2  # 滑窗缩短到2帧，不稀释瞬间爆发
    cfg["rules"]["alert_threshold"] = 0.20       # 2D 画面特征幅度变小，适当降低报警门槛

    video_path = "D:/dataset_1/five_dataset/fight/cam1_1.mp4" # 确保路径正确
    
    if not os.path.exists(video_path):
        print(f"[ERROR] 找不到演示视频: {video_path}")
        return

    print(f"\n[1/3] 正在加载视频并提取骨骼轨迹...")
    print(f"      输入视频: {os.path.basename(video_path)}")
    print(f"      底层模型: YOLOv8-Pose + BoT-SORT Tracker")
    
    track_set = process_video_to_trackset(video_path, label=1, cfg=cfg)
    
    if not track_set:
        print("[FAILED] 骨骼提取失败。")
        return
        
    print(f"\n[2/3] 骨骼提取完成！")
    print(f"      视频总帧数: {track_set.total_frames}")
    print(f"      追踪到的人数: {len(track_set.tracks)}")
    
    print(f"\n[3/3] 正在将轨迹数据送入规则引擎进行物理特征判定...")
    
    # ── 诊断代码：打印真实视频的特征极值 ──
    print("[诊断] 正在分析视频中的物理特征极值...")
    from fightguard.detection.pairing import get_interaction_pairs, compute_pair_distance_at_frame
    from fightguard.detection.interaction_rules import compute_frame_score
    
    pairs = get_interaction_pairs(track_set, cfg)
    if pairs:
        a, b = pairs[0]
        min_dist = float('inf')
        max_accel = 0.0
        dt = 1.0 / track_set.fps
        
        for fi in range(min(len(a.keypoints), len(b.keypoints))):
            # 诊断 1：最小近身距离
            dist = compute_pair_distance_at_frame(a, b, fi)
            if dist is not None:
                min_dist = min(min_dist, dist)
                
            # 诊断 2：最大腕部加速度
            if fi >= 2:
                _, details = compute_frame_score(a, b, fi, cfg, dt)
                max_accel = max(max_accel, details.get("a_A", 0.0))
                
        print(f"      - 两人最小归一化距离: {min_dist:.4f} (当前阈值: {cfg['rules']['proximity_threshold']})")
        print(f"      - 最大腕部线加速度  : {max_accel:.4f} (当前归一化上限: 50.0)")
    else:
        print("      - 未能成功配对双人交互。")
    print("------------------------------------------------------------")

    
    events = run_rules_symmetric(track_set, cfg)
    
    print("\n" + "="*60)
    print(" 判定结果")
    print("="*60)
    
    if events:
        for i, e in enumerate(events):
            print(f" [报警 {i+1}] 发现冲突行为！")
            print(f"   - 发生时间: 帧 {e.start_frame} ~ {e.end_frame}")
            print(f"   - 冲突对象: Track {e.track_ids[0]} 与 Track {e.track_ids[1]}")
            print(f"   - 危险得分: {e.score:.3f}")
            print(f"   - 触发规则: {', '.join(e.triggered_rules)}")
    else:
        print(" [正常] 规则引擎分析完毕，未达到冲突报警阈值。")
        print(" (注: 当前阈值基于 NTU 数据集标定，真实监控视频需重新标定阈值)")
    print("="*60)

if __name__ == "__main__":
    run_demo()
