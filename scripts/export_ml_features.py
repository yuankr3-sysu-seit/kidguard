"""
export_ml_features.py — 机器学习特征导出脚本
===========================================
遍历数据集，提取特征向量并导出为CSV格式。
特征包括：r_a, r_v, r_alpha, r_phi, r_p, gamma, volatility_factor, smoothing_factor
标签：0=正常, 1=冲突
"""

import os
import sys
import csv
from typing import List, Dict, Optional
import numpy as np

# 添加src目录到路径
sys.path.insert(0, "src")

from fightguard.inputs.skeleton_source import load_dataset
from fightguard.config import get_config
from fightguard.detection.interaction_rules import compute_directional_score
from fightguard.detection.pairing import get_interaction_pairs
from fightguard.contracts import TrackSet

def extract_features_from_trackset(track_set: TrackSet, cfg: dict) -> List[Dict]:
    """
    从TrackSet中提取特征向量
    
    Args:
        track_set: 轨迹集合
        cfg: 配置字典
    
    Returns:
        List[Dict]: 特征字典列表
    """
    features_list = []
    
    # 获取交互对
    pairs = get_interaction_pairs(track_set, cfg)
    if not pairs:
        return features_list
    
    dt = 1.0 / track_set.fps
    
    for track_a, track_b in pairs:
        n_frames = min(len(track_a.keypoints), len(track_b.keypoints))
        
        # 跳过太短的序列
        if n_frames < 10:
            continue
            
        # 提取每一帧的特征
        for fi in range(n_frames):
            try:
                # 计算双向特征
                score_ab, det_ab = compute_directional_score(
                    track_a, track_b, fi, cfg, dt, None, None
                )
                score_ba, det_ba = compute_directional_score(
                    track_b, track_a, fi, cfg, dt, None, None
                )
                
                # 使用得分较高的方向的特征
                if score_ab >= score_ba:
                    features = det_ab
                else:
                    features = det_ba
                
                # 添加标签
                features['label'] = track_set.label if track_set.label in [0, 1] else -1
                # 添加clip_id和帧信息用于调试
                features['clip_id'] = track_set.clip_id
                features['frame_idx'] = fi
                features['track_a_id'] = track_a.track_id
                features['track_b_id'] = track_b.track_id
                
                features_list.append(features)
                
            except Exception as e:
                print(f"警告: 在{track_set.clip_id}第{fi}帧提取特征时出错: {e}")
                continue
    
    return features_list

def main():
    """主函数"""
    print("="*60)
    print("KidGuard 机器学习特征导出工具")
    print("="*60)
    
    # 加载配置
    cfg = get_config()
    
    # 设置数据目录（这里需要根据实际情况修改）
    # 注意：这里需要用户提供实际的数据目录
    data_dirs = [
        "D:/dataset_1/five_dataset/fight",
        "D:/dataset_1/five_dataset/normal"
    ]
    
    # 检查数据目录是否存在
    valid_dirs = []
    for d in data_dirs:
        if os.path.exists(d):
            valid_dirs.append(d)
        else:
            print(f"警告: 数据目录不存在: {d}")
    
    if not valid_dirs:
        print("错误: 没有找到有效的数据目录")
        return
    
    print(f"正在从以下目录加载数据: {valid_dirs}")
    
    # 加载数据集
    print("正在加载骨骼数据...")
    all_track_sets = load_dataset(valid_dirs, cfg)
    
    if not all_track_sets:
        print("错误: 未能加载任何数据")
        return
    
    print(f"成功加载 {len(all_track_sets)} 个片段")
    
    # 提取特征
    all_features = []
    for i, track_set in enumerate(all_track_sets):
        print(f"正在处理片段 {i+1}/{len(all_track_sets)}: {track_set.clip_id}")
        features = extract_features_from_trackset(track_set, cfg)
        all_features.extend(features)
    
    print(f"总共提取了 {len(all_features)} 个特征样本")
    
    # 过滤掉标签无效的样本
    valid_features = [f for f in all_features if f.get('label') in [0, 1]]
    print(f"有效标签样本数: {len(valid_features)} (0:正常, 1:冲突)")
    
    if not valid_features:
        print("错误: 没有有效的特征样本")
        return
    
    # 定义CSV列名
    feature_columns = [
        'r_a', 'r_v', 'r_alpha', 'r_phi', 'r_p',
        'gamma', 'volatility_factor', 'smoothing_factor'
    ]
    meta_columns = ['clip_id', 'frame_idx', 'track_a_id', 'track_b_id', 'label']
    all_columns = meta_columns + feature_columns
    
    # 写入CSV文件
    output_file = "features.csv"
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()
        
        for feat in valid_features:
            # 确保所有特征列都存在
            row = {}
            for col in all_columns:
                if col in feat:
                    row[col] = feat[col]
                else:
                    # 为缺失的特征列提供默认值
                    if col in feature_columns:
                        row[col] = 0.0
                    else:
                        row[col] = ''
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
