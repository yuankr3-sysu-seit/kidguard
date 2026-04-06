"""
eval_ntu_dataset.py — NTU RGBD 骨骼数据集批量评测脚本
======================================================
用于在 NTU RGBD 3D 骨骼数据集上评估规则引擎的性能。
复用 skeleton_source.py 读取 .skeleton 文件，
复用 interaction_rules.py 进行物理特征判定，
复用 clip_metrics.py 计算评测指标。

与 eval_video_dataset.py 的区别：
  - 数据源：NTU .skeleton 文件（3D 骨骼）vs 2D 监控视频
  - 无需 YOLO 推理，直接读取骨骼坐标
  - 运行速度极快（无视觉推理瓶颈）
"""

import sys
import os
import random
import time
from tqdm import tqdm

# 确保能导入 src 下的包
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from fightguard.inputs.skeleton_source import load_dataset
from fightguard.detection.interaction_rules import run_rules_on_clip
from fightguard.config import get_config
from fightguard.evaluation.clip_metrics import calculate_metrics

def evaluate_on_ntu(max_clips: int = None):
    """
    在 NTU 数据集上批量评测规则引擎
    
    参数：
        max_clips: 限制评测样本数（用于快速验证），None=全量评测
    """
    print("="*60)
    print(" NTU RGBD 骨骼数据集 批量评测")
    print("="*60)
    
    cfg = get_config()
    
    # ── NTU 数据集路径配置 ──
    # 请根据实际情况修改以下路径
    data_dirs = [
        "D:/dataset_1/nturgbd_skeletons_s001_to_s017",
        "D:/dataset_1/nturgbd_skeletons_s018_to_s032",
    ]
    
    # 检查目录是否存在
    valid_dirs = [d for d in data_dirs if os.path.exists(d)]
    if not valid_dirs:
        print(f"[ERROR] NTU 数据集目录不存在：")
        for d in data_dirs:
            print(f"  - {d}")
        print("\n请修改脚本中的 data_dirs 变量，指向正确的 NTU .skeleton 文件目录")
        return
    
    print(f"[INFO] 找到 {len(valid_dirs)} 个有效数据目录")
    
    # ── 加载 NTU 骨骼数据 ──
    print(f"\n[1/3] 正在加载 NTU .skeleton 文件...")
    if max_clips is not None:
        print(f"      限制样本数：{max_clips}（快速验证模式）")
    else:
        print(f"      全量加载模式")
    
    start_time = time.time()
    track_sets = load_dataset(valid_dirs, cfg=cfg, max_clips=max_clips)
    load_time = int(time.time() - start_time)
    
    if not track_sets:
        print("[ERROR] 未加载到任何有效数据，请检查路径和文件格式")
        return
    
    # 统计样本分布
    n_conflict = sum(1 for t in track_sets if t.label == 1)
    n_normal = sum(1 for t in track_sets if t.label == 0)
    n_single_person = sum(1 for t in track_sets if len(t.tracks) < 2)
    n_valid = len(track_sets) - n_single_person
    
    print(f"      加载耗时：{load_time} 秒")
    print(f"      总样本数：{len(track_sets)} 个（冲突:{n_conflict}, 正常:{n_normal}）")
    print(f"      ⚠️  单人样本：{n_single_person} 个（将在评测时跳过）")
    print(f"      ✅ 有效双人样本：{n_valid} 个")
    
    # ── 批量评测 ──
    print(f"\n[2/3] 正在运行规则引擎进行物理特征判定...")
    results = []
    
    for ts in tqdm(track_sets, desc="NTU 评测进度", unit="clip"):
        # 检查是否有足够的人数进行交互判定
        if len(ts.tracks) < 2:
            # 【核心修复：单人样本是无效废数据，直接跳过，不计入统计】
            # 双人交互检测系统无法对单人样本进行判定，不应算作 FN 或 TN
            continue
        
        # 调用规则引擎（复用 interaction_rules.py）
        events = run_rules_on_clip(ts, cfg)
        
        # 判定逻辑：有事件 = 冲突，无事件 = 正常
        predicted_label = 1 if events else 0
        top_score = max(e.score for e in events) if events else 0.0
        
        results.append({
            "clip": ts.clip_id,
            "actual": ts.label,
            "predicted": predicted_label,
            "top_score": top_score,
            "reason": ""
        })
    
    # ── 计算评测指标 ──
    print(f"\n[3/3] 正在计算评测指标...")
    metrics = calculate_metrics(results)
    total_elapsed = int(time.time() - start_time)
    
    # ── 输出结果 ──
    print("\n" + "="*60)
    print(f" NTU 3D 骨骼数据集评测结果 (总耗时: {total_elapsed} 秒)")
    print("="*60)
    print(f"  测试样本数 : {metrics.get('total', 0)} (冲突:{metrics.get('tp',0)+metrics.get('fn',0)}, 正常:{metrics.get('tn',0)+metrics.get('fp',0)})")
    print(f"  TP={metrics.get('tp',0)}  FP={metrics.get('fp',0)}  TN={metrics.get('tn',0)}  FN={metrics.get('fn',0)}")
    print("-" * 60)
    print(f"  Accuracy  (准确率) : {metrics.get('accuracy', 0.0):.4f}")
    print(f"  Precision (精确率) : {metrics.get('precision', 0.0):.4f}")
    print(f"  Recall    (召回率) : {metrics.get('recall', 0.0):.4f}")
    print(f"  FPR       (误报率) : {metrics.get('fpr', 0.0):.4f}")
    print(f"  FNR       (漏报率) : {metrics.get('fnr', 0.0):.4f}")
    print("="*60)
    
    # ── 错判案例分析 ──
    print("\n[错判案例分析]")
    fp_count = 0
    fn_count = 0
    
    for r in results:
        if r["actual"] != r["predicted"]:
            if r["actual"] == 1:
                fn_count += 1
                status = "漏报 (FN)"
            else:
                fp_count += 1
                status = "误报 (FP)"
            
            print(f"  - {status}: {r['clip']} | 实际:{r['actual']} 预测:{r['predicted']} | 最高得分:{r['top_score']:.3f}")
    
    if fp_count == 0 and fn_count == 0:
        print("  ✅ 完美！所有样本均正确分类！")
    else:
        print(f"\n  汇总：误报 {fp_count} 个，漏报 {fn_count} 个")
    
    print("="*60)

if __name__ == "__main__":
    # ── 配置区 ──
    # max_clips=None 表示全量评测
    # max_clips=100 表示只评测前 100 个样本（快速验证）
    MAX_CLIPS = 100
    
    evaluate_on_ntu(max_clips=MAX_CLIPS)
