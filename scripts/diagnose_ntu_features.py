"""
diagnose_ntu_features.py — NTU 特征分布诊断脚本
=================================================
任务 B：诊断 NTU 数据的特征分布，找出归一化与阈值不匹配的根因
"""

import sys
import os
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from fightguard.inputs.skeleton_source import load_dataset
from fightguard.detection.interaction_rules import compute_directional_score
from fightguard.detection.math_utils import get_shoulder_scale
from fightguard.detection.pairing import get_interaction_pairs
from fightguard.config import get_config

def diagnose_features(max_clips: int = 200):
    print("="*70)
    print(" NTU 特征分布诊断 (任务 B)")
    print("="*70)
    
    cfg = get_config()
    
    data_dirs = [
        "D:/dataset_1/nturgbd_skeletons_s001_to_s017",
        "D:/dataset_1/nturgbd_skeletons_s018_to_s032",
    ]
    
    # 加载数据
    print("\n[1/4] 加载 NTU 数据...")
    track_sets = load_dataset(data_dirs, cfg=cfg, max_clips=max_clips)
    if not track_sets:
        print("[ERROR] 加载失败")
        return
    
    # 分类统计
    conflict_samples = [ts for ts in track_sets if ts.label == 1]
    normal_samples = [ts for ts in track_sets if ts.label == 0]
    
    print(f"      冲突样本: {len(conflict_samples)}")
    print(f"      正常样本: {len(normal_samples)}")
    
    # ============================================================
    # 诊断 1：肩宽尺度分布
    # ============================================================
    print("\n[2/4] 诊断肩宽尺度 (shoulder_scale)...")
    
    shoulder_scales_conflict = []
    shoulder_scales_normal = []
    
    for ts in tqdm(conflict_samples + normal_samples, desc="提取肩宽", unit="clip"):
        is_conflict = (ts.label == 1)
        for track in ts.tracks:
            for kp in track.keypoints:
                if kp.get("nose", [0,0,0])[:2] != [0.0, 0.0]:  # 跳过空帧
                    scale = get_shoulder_scale(kp)
                    if is_conflict:
                        shoulder_scales_conflict.append(scale)
                    else:
                        shoulder_scales_normal.append(scale)
    
    print("\n  冲突样本肩宽分布:")
    if shoulder_scales_conflict:
        print(f"    最小值: {min(shoulder_scales_conflict):.6f}")
        print(f"    最大值: {max(shoulder_scales_conflict):.6f}")
        print(f"    均值:   {np.mean(shoulder_scales_conflict):.6f}")
        print(f"    中位数: {np.median(shoulder_scales_conflict):.6f}")
        print(f"    标准差: {np.std(shoulder_scales_conflict):.6f}")
    
    print("\n  正常样本肩宽分布:")
    if shoulder_scales_normal:
        print(f"    最小值: {min(shoulder_scales_normal):.6f}")
        print(f"    最大值: {max(shoulder_scales_normal):.6f}")
        print(f"    均值:   {np.mean(shoulder_scales_normal):.6f}")
        print(f"    中位数: {np.median(shoulder_scales_normal):.6f}")
        print(f"    标准差: {np.std(shoulder_scales_normal):.6f}")
    
    # ============================================================
    # 诊断 2：特征值分布（r_a, r_v, r_phi, r_p, gamma）
    # ============================================================
    print("\n[3/4] 诊断特征值分布...")
    
    features_conflict = {"r_a": [], "r_v": [], "r_phi": [], "r_p": [], "gamma": [], "score": []}
    features_normal = {"r_a": [], "r_v": [], "r_phi": [], "r_p": [], "gamma": [], "score": []}
    
    dt = 1.0 / 30.0  # NTU 固定 30fps
    
    for ts in tqdm(track_sets, desc="提取特征", unit="clip"):
        is_conflict = (ts.label == 1)
        target = features_conflict if is_conflict else features_normal
        
        pairs = get_interaction_pairs(ts, cfg)
        if not pairs:
            continue
        
        track_a, track_b = pairs[0]
        n_frames = min(len(track_a.keypoints), len(track_b.keypoints))
        
        for fi in range(n_frames):
            score_ab, det_ab = compute_directional_score(track_a, track_b, fi, cfg, dt)
            score_ba, det_ba = compute_directional_score(track_b, track_a, fi, cfg, dt)
            
            # 取主导方向
            if score_ab > score_ba:
                det = det_ab
                score = score_ab
            else:
                det = det_ba
                score = score_ba
            
            target["r_a"].append(det["r_a"])
            target["r_v"].append(det["r_v"])
            target["r_phi"].append(det["r_phi"])
            target["r_p"].append(det["r_p"])
            target["gamma"].append(det["gamma"])
            target["score"].append(score)
    
    # 输出统计
    for label_name, features in [("冲突样本", features_conflict), ("正常样本", features_normal)]:
        print(f"\n  {label_name}特征分布:")
        for feat_name, values in features.items():
            if values:
                arr = np.array(values)
                print(f"    {feat_name:8s}: min={arr.min():.4f}, max={arr.max():.4f}, "
                      f"mean={arr.mean():.4f}, median={np.median(arr):.4f}, "
                      f"std={arr.std():.4f}")
                
                # 检查异常
                zeros = np.sum(arr == 0.0)
                ones = np.sum(arr >= 0.99)
                if zeros > len(arr) * 0.5:
                    print(f"      ⚠️  警告：{zeros}/{len(arr)} ({zeros/len(arr)*100:.1f}%) 的值为 0")
                if ones > len(arr) * 0.5:
                    print(f"      ⚠️  警告：{ones}/{len(arr)} ({ones/len(arr)*100:.1f}%) 的值 >= 0.99")
    
    # ============================================================
    # 诊断 3：关键对比
    # ============================================================
    print("\n[4/4] 关键对比分析...")
    
    if features_conflict["score"] and features_normal["score"]:
        conflict_scores = np.array(features_conflict["score"])
        normal_scores = np.array(features_normal["score"])
        
        print(f"\n  得分对比:")
        print(f"    冲突样本: mean={conflict_scores.mean():.4f}, max={conflict_scores.max():.4f}")
        print(f"    正常样本: mean={normal_scores.mean():.4f}, max={normal_scores.max():.4f}")
        print(f"    重叠度: 正常样本>0.3 的比例 = {np.sum(normal_scores > 0.3) / len(normal_scores) * 100:.1f}%")
        print(f"    重叠度: 冲突样本>0.3 的比例 = {np.sum(conflict_scores > 0.3) / len(conflict_scores) * 100:.1f}%")
    
    if shoulder_scales_conflict and shoulder_scales_normal:
        print(f"\n  肩宽对比:")
        print(f"    冲突样本: mean={np.mean(shoulder_scales_conflict):.6f}")
        print(f"    正常样本: mean={np.mean(shoulder_scales_normal):.6f}")
        diff = abs(np.mean(shoulder_scales_conflict) - np.mean(shoulder_scales_normal))
        if diff < 0.01:
            print(f"    ⚠️  警告：两类样本肩宽几乎相同（差异 {diff:.6f}），归一化可能失去了区分度")
    
    print("\n" + "="*70)
    print(" 诊断完成")
    print("="*70)

if __name__ == "__main__":
    diagnose_features(max_clips=200)
