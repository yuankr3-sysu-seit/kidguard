"""
extract_features_eda.py — 探索性数据分析(EDA)特征提取脚本
==========================================================
阶段 A 核心脚本：为熵权法（Entropy Weight Method）准备原始数据。
遍历数据集，提取每个 clip 中交互双人的四个核心物理特征的峰值，
并保存为 CSV 文件，用于后续的数学统计与科学赋权。
"""

import sys
import os
import csv

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, **kwargs):
        return iterable

# 确保能导入 src 下的包
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from fightguard.inputs.skeleton_source import load_dataset
from fightguard.detection.pairing import get_interaction_pairs
from fightguard.detection.interaction_rules import compute_frame_score
from fightguard.config import get_config

def run_feature_extraction():
    print("="*60)
    print(" 阶段 A：数据驱动的特征提取 (EDA) 开始运行")
    print("="*60)
    
    cfg = get_config()
    
    # 数据集路径
    data_dirs = [
        "D:/dataset_1/nturgbd_skeletons_s001_to_s017",
        "D:/dataset_1/nturgbd_skeletons_s018_to_s032",
    ]
    
    # 加载样本 (使用 500 个样本以保证统计学意义)
    print("[1/3] 正在扫描并加载数据集...")
    track_sets = load_dataset(data_dirs, max_clips=500)
    if not track_sets:
        print("[ERROR] 未加载到数据，请检查路径。")
        return

    extracted_data = []
    skipped = 0

    print(f"[2/3] 正在提取物理特征峰值 (共 {len(track_sets)} 个有效 Clip)...")
    # 使用 tqdm 显示进度条
    for ts in tqdm(track_sets, desc="提取进度", unit="clip"):
        # 1. 寻找交互对
        pairs = get_interaction_pairs(ts, cfg)
        if not pairs:
            skipped += 1
            continue
            
        track_a, track_b = pairs[0]
        n_frames = min(len(track_a.keypoints), len(track_b.keypoints))
        dt = 1.0 / ts.fps
        
        # 2. 遍历所有帧，寻找四个特征的全局峰值
        max_a_A = 0.0
        max_v_rel = 0.0
        max_alpha_A = 0.0
        max_delta_phi = 0.0
        
        for fi in range(n_frames):
            # 复用 interaction_rules 中的计算逻辑，获取单帧的 details 字典
            _, details = compute_frame_score(track_a, track_b, fi, cfg, dt)
            
            max_a_A       = max(max_a_A,       details.get("a_A", 0.0))
            max_v_rel     = max(max_v_rel,     details.get("v_rel", 0.0))
            max_alpha_A   = max(max_alpha_A,   details.get("alpha_A", 0.0))
            max_delta_phi = max(max_delta_phi, details.get("delta_phi", 0.0))
            
        # 3. 记录该 clip 的特征画像
        extracted_data.append({
            "clip_id": ts.clip_id,
            "label": ts.label,  # 1 为冲突，0 为正常
            "peak_a_A": max_a_A,
            "peak_v_rel": max_v_rel,
            "peak_alpha_A": max_alpha_A,
            "peak_delta_phi": max_delta_phi
        })

    print(f"[3/3] 提取完成！成功提取 {len(extracted_data)} 个样本，因人数不足跳过 {skipped} 个。")
    
    # 4. 写入 CSV 文件
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'metrics'))
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "eda_raw_features.csv")
    
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "label", "peak_a_A", "peak_v_rel", "peak_alpha_A", "peak_delta_phi"])
        writer.writeheader()
        writer.writerows(extracted_data)
        
    print(f"✅ 特征数据已成功保存至: {csv_path}")
    print("下一步：我们将使用此数据矩阵，进行熵权法计算！")

if __name__ == "__main__":
    run_feature_extraction()
