import sys
import os
sys.path.insert(0, "src")

from fightguard.inputs.video_source import process_video_to_trackset
from fightguard.detection.interaction_rules import run_rules_symmetric
from fightguard.config import get_config

cfg = get_config()

# 找一个 fight 文件夹里的视频来测试
video_dir = "D:/dataset_1/five_dataset/fight"
# 获取目录下的第一个视频文件
video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mpg'))]

if not video_files:
    print("未找到视频文件，请检查路径。")
else:
    test_video = os.path.join(video_dir, video_files[0])
    print(f"[INFO] 准备处理视频：{test_video}")
    
    # 为了测试速度，我们只处理前 60 帧（约 2 秒）
    print("[INFO] 开始提取骨骼 (使用 YOLOv8-Pose CPU 模式)...")
    print("[INFO] 开始提取骨骼 (开启 BoT-SORT 追踪)...")
    track_set = process_video_to_trackset(test_video, label=1, cfg=cfg) # 去掉 max_frames

    if track_set:
        print(f"[SUCCESS] 骨骼提取成功！总帧数: {track_set.total_frames}, 检测到人数: {len(track_set.tracks)}")
        
        print("[INFO] 开始运行规则引擎...")
        events = run_rules_symmetric(track_set, cfg)
        
        if events:
            e = events[0]
            print(f"[ALARM] 发现冲突！帧 {e.start_frame}~{e.end_frame}, 得分: {e.score:.3f}")
            print(f"        触发规则: {e.triggered_rules}")
        else:
            print("[OK] 规则引擎未报告冲突。")
    else:
        print("[FAILED] 骨骼提取失败或未检测到人。")
