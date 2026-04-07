"""
plot_metrics.py — 评估指标可视化脚本
====================================
生成混淆矩阵和特征分布图。
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

def main():
    """主函数"""
    print("="*60)
    print("KidGuard 评估指标可视化")
    print("="*60)
    
    # 检查必要的文件
    model_path = "models/svm_baseline.pkl"
    scaler_path = "models/scaler.pkl"
    test_results_path = "models/test_predictions.csv"
    features_file = "features.csv"
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 '{model_path}' 不存在")
        print("请先运行 train_svm_baseline.py 训练模型")
        return
    
    if not os.path.exists(test_results_path):
        print(f"错误: 测试结果文件 '{test_results_path}' 不存在")
        print("请先运行 train_svm_baseline.py 生成测试结果")
        return
    
    if not os.path.exists(features_file):
        print(f"错误: 特征文件 '{features_file}' 不存在")
        print("请先运行 export_ml_features.py 生成特征文件")
        return
    
    # 加载测试结果
    print("正在加载测试结果...")
    test_results = pd.read_csv(test_results_path)
    y_true = test_results['true_label'].values
    y_pred = test_results['pred_label'].values
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 创建混淆矩阵可视化
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['正常', '冲突'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title('混淆矩阵 (Confusion Matrix)')
    
    # 确保日志目录存在
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 保存混淆矩阵图像
    cm_path = os.path.join(log_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    print(f"混淆矩阵已保存到: {cm_path}")
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=['正常', '冲突']))
    
    # 计算并打印其他指标
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n性能指标:")
    print(f"  准确率 (Accuracy): {accuracy:.4f}")
    print(f"  精确率 (Precision): {precision:.4f}")
    print(f"  召回率 (Recall): {recall:.4f}")
    print(f"  F1-score: {f1:.4f}")
    
    # 生成特征分布散点图
    print("\n正在生成特征分布散点图...")
    df_features = pd.read_csv(features_file)
    
    # 检查必要的列
    required_cols = ['accel_var', 'rel_vel_var', 'label']
    missing_cols = [col for col in required_cols if col not in df_features.columns]
    if missing_cols:
        print(f"警告: 特征文件中缺少以下列，无法生成特征分布图: {missing_cols}")
    else:
        plt.figure(figsize=(10, 8))
        
        # 按类别着色
        colors = ['blue', 'red']
        labels = ['正常', '冲突']
        
        for label_value, color, label_name in zip([0, 1], colors, labels):
            mask = df_features['label'] == label_value
            plt.scatter(
                df_features.loc[mask, 'accel_var'],
                df_features.loc[mask, 'rel_vel_var'],
                c=color, alpha=0.6, s=50, label=label_name, edgecolors='w', linewidth=0.5
            )
        
        plt.xlabel('Acceleration Variance (accel_var)', fontsize=12)
        plt.ylabel('Relative Velocity Variance (rel_vel_var)', fontsize=12)
        plt.title('Feature Distribution: Acceleration Variance vs Relative Velocity Variance', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存特征分布图
        feature_path = os.path.join(log_dir, "feature_distribution.png")
        plt.tight_layout()
        plt.savefig(feature_path, dpi=300)
        plt.close()
        
        print(f"特征分布图已保存到: {feature_path}")
    
    print("="*60)

if __name__ == "__main__":
    main()
