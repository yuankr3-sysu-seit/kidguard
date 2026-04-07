"""
train_svm_baseline.py — SVM基线分类器训练脚本
===========================================
读取特征CSV文件，训练SVM分类器，并保存模型。
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

def main():
    """主函数"""
    print("="*60)
    print("KidGuard SVM基线分类器训练")
    print("="*60)
    
    # 检查特征文件是否存在
    feature_file = "features.csv"
    if not os.path.exists(feature_file):
        print(f"错误: 特征文件 '{feature_file}' 不存在")
        print("请先运行 export_ml_features.py 生成特征文件")
        return
    
    # 加载特征数据
    print(f"正在加载特征文件: {feature_file}")
    df = pd.read_csv(feature_file)
    
    # 检查必要的列
    required_feature_cols = ['r_a', 'r_v', 'r_alpha', 'r_phi', 'r_p', 
                            'gamma', 'volatility_factor', 'smoothing_factor']
    required_meta_cols = ['label']
    
    for col in required_feature_cols + required_meta_cols:
        if col not in df.columns:
            print(f"错误: 缺少必要的列 '{col}'")
            return
    
    # 准备特征和标签
    X = df[required_feature_cols].values
    y = df['label'].values
    
    print(f"数据集形状: X={X.shape}, y={y.shape}")
    print(f"类别分布: 正常={np.sum(y==0)}, 冲突={np.sum(y==1)}")
    
    # 划分训练集和测试集 (8:2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    # 数据标准化
    print("正在进行数据标准化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练SVM分类器
    print("正在训练SVM分类器 (RBF核)...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    # 在测试集上评估
    y_pred = svm.predict(X_test_scaled)
    
    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    
    print("\n测试集性能指标:")
    print(f"  准确率 (Accuracy): {accuracy:.4f}")
    print(f"  精确率 (Precision): {precision:.4f}")
    print(f"  召回率 (Recall): {recall:.4f}")
    
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred, target_names=['正常', '冲突']))
    
    # 保存模型和标准化器
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "svm_baseline.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(svm, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\n模型已保存到: {model_path}")
    print(f"标准化器已保存到: {scaler_path}")
    
    # 保存测试集预测结果用于后续分析
    test_results = pd.DataFrame({
        'true_label': y_test,
        'pred_label': y_pred
    })
    test_results_path = os.path.join(model_dir, "test_predictions.csv")
    test_results.to_csv(test_results_path, index=False)
    print(f"测试集预测结果已保存到: {test_results_path}")
    
    print("="*60)

if __name__ == "__main__":
    main()
