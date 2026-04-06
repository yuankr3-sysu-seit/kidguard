"""
eval_video_dataset.py — 真实视频数据集批量评测脚本 (带实时秒表)
================================================================
用于在真实的 2D 监控视频数据集上评估规则引擎的泛化能力。
新增了后台实时秒表线程，缓解 YOLO 推理慢导致的“终端假死”焦虑。
"""

import sys
import os
import random
import time
import threading
from tqdm import tqdm


# 确保能导入 src 下的包
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from fightguard.inputs.video_source import process_video_to_trackset
from fightguard.detection.interaction_rules import run_rules_on_clip
from fightguard.config import get_config
from fightguard.evaluation.clip_metrics import calculate_metrics

def evaluate_on_videos(num_samples_per_class: int = 5):
    print("="*60)
    print(f" 真实视频数据集 评测 (每类抽取 {num_samples_per_class} 个)")
    print("="*60)
    
    cfg = get_config()
    
    # ── 针对真实 2D 监控视频的专属参数覆盖 ──
    cfg["rules"]["proximity_window_frames"] = 2  
    cfg["rules"]["smoothing_window_frames"] = 2  
    cfg["rules"]["alert_threshold"] = 0.20       
    
    dataset_dir = "D:/dataset_1/five_dataset"
    fight_dir = os.path.join(dataset_dir, "fight")
    nofight_dir = os.path.join(dataset_dir, "nofight")
    
    if not os.path.exists(fight_dir) or not os.path.exists(nofight_dir):
        print(f"[ERROR] 找不到视频数据集目录: {dataset_dir}")
        return

    fight_files = [f for f in os.listdir(fight_dir) if f.endswith(('.mp4', '.avi', '.mpg'))]
    nofight_files = [f for f in os.listdir(nofight_dir) if f.endswith(('.mp4', '.avi', '.mpg'))]
    
    # random.seed(42) # 固定随机种子
    
    
    sampled_fights = random.sample(fight_files, min(num_samples_per_class, len(fight_files)))
    sampled_nofights = random.sample(nofight_files, min(num_samples_per_class, len(nofight_files)))
    
    test_cases = [(os.path.join(fight_dir, f), 1) for f in sampled_fights] + \
                 [(os.path.join(nofight_dir, f), 0) for f in sampled_nofights]
                 
    random.shuffle(test_cases)
    results = []
    
    print(f"[INFO] 开始处理 {len(test_cases)} 个视频 (YOLOv8-Pose CPU 推理较慢，请耐心等待)...")
    
    # =========================================================
    # 核心：多线程实时秒表
    # =========================================================
    start_time = time.time()
    # 自定义进度条格式，预留 postfix 位置显示时间
    pbar = tqdm(test_cases, desc="视频处理进度", 
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [总耗时: {postfix}]")
    
    stop_timer = False
    def update_timer():
        while not stop_timer:
            elapsed = int(time.time() - start_time)
            mins, secs = divmod(elapsed, 60)
            # 实时刷新进度条的后缀文本
            pbar.set_postfix_str(f"{mins:02d}分{secs:02d}秒", refresh=True)
            time.sleep(1)

    # 启动后台秒表线程 (daemon=True 保证主程序退出时线程自动销毁)
    timer_thread = threading.Thread(target=update_timer, daemon=True)
    timer_thread.start()
    # =========================================================
    
    try:
        for video_path, actual_label in pbar:
            track_set = process_video_to_trackset(video_path, label=actual_label, cfg=cfg)
            
            predicted_label = 0
            top_score = 0.0
            
            if track_set and len(track_set.tracks) >= 2:
                # 调用我们刚刚重写的、带有队长四段式状态机的核心函数
                events = run_rules_on_clip(track_set, cfg)
                if events:
                    predicted_label = 1
                    top_score = max(e.score for e in events)
                    
            results.append({
                "video": os.path.basename(video_path),
                "actual": actual_label,
                "predicted": predicted_label,
                "top_score": top_score
            })
    finally:
        # 无论程序正常结束还是被 Ctrl+C 中断，都安全关闭秒表线程
        stop_timer = True
        timer_thread.join(timeout=1.0)
        pbar.close()
        
    metrics = calculate_metrics(results)
    total_elapsed = int(time.time() - start_time)
    
    print("\n" + "="*60)
    print(f" 真实视频 2D 场景评测结果 (总耗时: {total_elapsed // 60}分{total_elapsed % 60}秒)")
    print("="*60)
    print(f"  测试样本数 : {metrics.get('total', 0)} (冲突:{metrics.get('tp',0)+metrics.get('fn',0)}, 正常:{metrics.get('tn',0)+metrics.get('fp',0)})")
    print(f"  TP={metrics.get('tp',0)}  FP={metrics.get('fp',0)}  TN={metrics.get('tn',0)}  FN={metrics.get('fn',0)}")
    print("-" * 60)
    print(f"  Accuracy  (准确率) : {metrics.get('accuracy', 0.0):.4f}")
    print(f"  Precision (精确率) : {metrics.get('precision', 0.0):.4f}")
    print(f"  Recall    (召回率) : {metrics.get('recall', 0.0):.4f}")
    print(f"  FPR       (误报率) : {metrics.get('fpr', 0.0):.4f}")
    print("="*60)
    
    print("\n[错判案例分析]")
    for r in results:
        if r["actual"] != r["predicted"]:
            status = "漏报 (FN)" if r["actual"] == 1 else "误报 (FP)"
            print(f"  - {status}: {r['video']} | 实际:{r['actual']} 预测:{r['predicted']} | 最高得分:{r['top_score']:.3f}")

if __name__ == "__main__":
    evaluate_on_videos(num_samples_per_class=10)
