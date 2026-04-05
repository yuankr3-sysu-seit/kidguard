"""
calculate_entropy_weights.py — 熵权法(EWM)权重计算器
======================================================
读取 EDA 提取的特征数据，使用信息熵理论客观推导四大物理特征的权重。
彻底废除“拍脑袋”的经验参数，实现数据驱动的科学赋权。
"""

import os
import pandas as pd
import numpy as np

def calculate_entropy_weights():
    print("="*60)
    print(" 阶段 A：熵权法 (Entropy Weight Method) 科学赋权")
    print("="*60)

    # 1. 读取数据
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'metrics', 'eda_raw_features.csv'))
    if not os.path.exists(csv_path):
        print(f"[ERROR] 找不到特征数据文件：{csv_path}")
        return

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print("[ERROR] 数据集为空，请先运行 extract_features_eda.py")
        return

    print(f"[INFO] 成功加载 {len(df)} 个双人交互样本（冲突:{len(df[df['label']==1])}，正常:{len(df[df['label']==0])}）")

    # 2. 构建特征矩阵 X
    # 我们有四个核心特征，都是“正向指标”（值越大，越危险）
    features = ["peak_a_A", "peak_v_rel", "peak_alpha_A", "peak_delta_phi"]
    X = df[features].values

    # 3. 数据标准化 (Min-Max)
    # 将所有特征映射到 [0, 1] 区间，消除量纲影响
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    ranges = X_max - X_min
    ranges[ranges == 0] = 1e-9  # 防止除以 0

    # 为了后续计算对数时不出现 log(0)，整体加上一个极小值 1e-6
    Y = (X - X_min) / ranges + 1e-6

    # 4. 计算特征比重矩阵 P
    P = Y / Y.sum(axis=0)

    # 5. 计算信息熵 E
    n = len(df)
    k = 1.0 / np.log(n)
    E = -k * (P * np.log(P)).sum(axis=0)

    # 6. 计算差异系数 D (冗余度)
    D = 1.0 - E

    # 7. 计算最终权重 W
    W = D / D.sum()

    print("[计算完成] 各特征的客观数学权重如下：")
    print("-" * 40)
    print(f" 1. 腕部线加速度 (a_A)      : {W[0]:.4f}")
    print(f" 2. 相对接近速度 (v_rel)    : {W[1]:.4f}")
    print(f" 3. 肘部角加速度 (alpha_A)  : {W[2]:.4f}")
    print(f" 4. 躯干倾角变化 (delta_phi): {W[3]:.4f}")
    print("-" * 40)
    
    print("[工程指令] 请将上述权重更新至 interaction_rules.py 中的 w1, w2, w3, w4 变量！")

if __name__ == "__main__":
    calculate_entropy_weights()
