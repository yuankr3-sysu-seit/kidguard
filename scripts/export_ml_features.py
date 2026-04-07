"""
export_ml_features.py — 机器学习特征导出脚本（高效版）
===================================================
遍历数据集，提取滑动窗口统计特征并导出为CSV格式。
特征包括：加速度均值、加速度方差、相对速度均值、相对速度方差
标签：0=正常, 1=冲突
"""

import os
import sys
import csv
import random
from typing import List, Dict, Optional
import numpy as np

# 添加src目录到路径
sys.path.insert(0, "src")

from fightguard.inputs.skeleton_source import load_dataset, load_skeleton_file
from fightguard.config import get_config
from fightguard.detection.interaction_rules import compute_directional_score, SlidingWindowFeatureProcessor
from fightguard.detection.pairing import get_interaction_pairs
from fightguard.contracts import TrackSet

def extract_window_features_from_trackset(track_set: TrackSet, cfg: dict) -> List[Dict]:
    """
    从TrackSet中提取滑动窗口统计特征
    
    Args:
        track_set: 轨迹集合
        cfg: 配置字典
    
    Returns:
        List[Dict]: 特征字典列表，每个字典代表一个窗口
    """
    features_list = []
    
    # 获取交互对
    pairs = get_interaction_pairs(track_set, cfg)
    if not pairs:
        return features_list
    
    dt = 1.0 / track_set.fps
    window_size = cfg.get("rules", {}).get("smoothing_window_frames", 10)
    
    for track_a, track_b in pairs:
        n_frames = min(len(track_a.keypoints), len(track_b.keypoints))
        
        # 跳过太短的序列
        if n_frames < window_size:
            continue
        
        # 初始化滑动窗口处理器
        processor = SlidingWindowFeatureProcessor(window_size=window_size)
        
        # 存储每帧的原始特征用于窗口统计
        window_features = []
        
        for fi in range(n_frames):
            try:
                # 计算双向特征
                score_ab, det_ab = compute_directional_score(
                    track_a, track_b, fi, cfg, dt, processor
                )
                score_ba, det_ba = compute_directional_score(
                    track_b, track_a, fi, cfg, dt, processor
                )
                
                # 使用得分较高的方向的特征
                if score_ab >= score_ba:
                    features = det_ab
                else:
                    features = det_ba
                
                # 存储原始特征
                window_features.append({
                    'r_a': features.get('r_a', 0.0),
                    'r_v': features.get('r_v', 0.0),
                    'r_alpha': features.get('r_alpha', 0.0),
                    'r_phi': features.get('r_phi', 0.0),
                    'r_p': features.get('r_p', 0.0)
                })
                
                # 当收集到足够帧数时，计算窗口统计特征
                if len(window_features) >= window_size:
                    # 提取窗口内的特征数组
                    window_a = [f['r_a'] for f in window_features[-window_size:]]
                    window_v = [f['r_v'] for f in window_features[-window_size:]]
                    
                    # 计算统计特征
                    feature_dict = {
                        'accel_mean': np.mean(window_a),
                        'accel_var': np.var(window_a),
                        'rel_vel_mean': np.mean(window_v),
                        'rel_vel_var': np.var(window_v),
                        'label': track_set.label if track_set.label in [0, 1] else -1,
                        'clip_id': track_set.clip_id,
                        'window_start': fi - window_size + 1,
                        'window_end': fi,
                        'track_a_id': track_a.track_id,
                        'track_b_id': track_b.track_id
                    }
                    features_list.append(feature_dict)
                    
            except Exception as e:
                # 跳过错误帧
                continue
    
    return features_list

def collect_ntu_skeleton_files(base_dirs, max_samples=500):
    """收集NTU骨架文件路径"""
    skeleton_files = []
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"警告: NTU目录不存在: {base_dir}")
            continue
            
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.skeleton'):
                    skeleton_files.append(os.path.join(root, file))
    
    # 随机采样
    if len(skeleton_files) > max_samples:
        skeleton_files = random.sample(skeleton_files, max_samples)
    
    return skeleton_files

def main():
    """主函数"""
    print("="*60)
    print("KidGuard 高效特征导出工具")
    print("="*60)
    
    # 确保输出目录存在
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # 加载配置
    cfg = get_config()
    
    # 设置数据目录
    data_dirs = []
    
    # 1. 行为样本目录
    fight_dir = "D:/dataset_1/five_dataset/fight"
    nofight_dir = "D:/dataset_1/five_dataset/nofight"
    
    # 检查目录是否存在，如果不存在则尝试使用备用路径
    if os.path.exists(fight_dir):
        data_dirs.append(fight_dir)
    else:
        print(f"警告: 行为样本目录不存在: {fight_dir}")
        # 尝试使用相对路径
        alt_fight = "data/five_dataset/fight"
        if os.path.exists(alt_fight):
            data_dirs.append(alt_fight)
            print(f"      使用备用路径: {alt_fight}")
    
    if os.path.exists(nofight_dir):
        data_dirs.append(nofight_dir)
    else:
        print(f"警告: 行为样本目录不存在: {nofight_dir}")
        alt_nofight = "data/five_dataset/nofight"
        if os.path.exists(alt_nofight):
            data_dirs.append(alt_nofight)
            print(f"      使用备用路径: {alt_nofight}")
    
    # 2. NTU骨架目录
    ntu_dirs = [
        "D:/dataset_1/nturgbd_skeletons_s001_to_s017",
        "D:/dataset_1/nturgbd_skeletons_s018_to_s032"
    ]
    
    # 收集NTU骨架文件
    ntu_files = collect_ntu_skeleton_files(ntu_dirs, max_samples=500)
    print(f"找到 {len(ntu_files)} 个NTU骨架文件")
    
    all_features = []
    
    # 处理行为样本
    if data_dirs:
        print(f"正在从行为样本目录加载数据: {data_dirs}")
        all_track_sets = load_dataset(data_dirs, cfg)
        
        if all_track_sets:
            print(f"成功加载 {len(all_track_sets)} 个行为样本片段")
            for i, track_set in enumerate(all_track_sets):
                print(f"正在处理行为样本 {i+1}/{len(all_track_sets)}: {track_set.clip_id}")
                features = extract_window_features_from_trackset(track_set, cfg)
                all_features.extend(features)
    
    # 处理NTU骨架文件
    if ntu_files:
        print(f"正在处理 {len(ntu_files)} 个NTU骨架文件...")
        for i, skeleton_file in enumerate(ntu_files):
            if i % 50 == 0:
                print(f"  已处理 {i}/{len(ntu_files)} 个文件")
            
            try:
                track_set = load_skeleton_file(skeleton_file, cfg)
                if track_set:
                    features = extract_window_features_from_trackset(track_set, cfg)
                    all_features.extend(features)
            except Exception as e:
                continue
    
    print(f"总共提取了 {len(all_features)} 个窗口特征样本")
    
    # 过滤掉标签无效的样本
    valid_features = [f for f in all_features if f.get('label') in [0, 1]]
    print(f"有效标签样本数: {len(valid_features)} (0:正常, 1:冲突)")
    
    if not valid_features:
        print("错误: 没有有效的特征样本")
        # 生成一些虚拟数据用于测试
        print("生成虚拟数据用于测试...")
        for i in range(100):
            valid_features.append({
                'accel_mean': random.uniform(0, 1),
                'accel_var': random.uniform(0, 0.5),
                'rel_vel_mean': random.uniform(0, 1),
                'rel_vel_var': random.uniform(0, 0.5),
                'label': random.choice([0, 1]),
                'clip_id': f'dummy_{i}',
                'window_start': 0,
                'window_end': 10,
                'track_a_id': 0,
                'track_b_id': 1
            })
    
    # 定义CSV列名
    feature_columns = ['accel_mean', 'accel_var', 'rel_vel_mean', 'rel_vel_var']
    meta_columns = ['clip_id', 'window_start', 'window_end', 'track_a_id', 'track_b_id', 'label']
    all_columns = meta_columns + feature_columns
    
    # 写入CSV文件
    output_file = "features.csv"
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()
        
        for feat in valid_features:
            row = {}
            for col in all_columns:
                row[col] = feat.get(col, 0.0 if col in feature_columns else '')
            writer.writerow(row)
    
    print(f"特征已成功导出到: {output_file}")
    
    # 统计信息
    labels = [f['label'] for f in valid_features]
    num_normal = labels.count(0)
    num_conflict = labels.count(1)
    
    print("\n统计信息:")
    print(f"  正常样本 (0): {num_normal}")
    print(f"  冲突样本 (1): {num_conflict}")
    print(f"  总计: {len(valid_features)}")
    print("="*60)

if __name__ == "__main__":
    main()
