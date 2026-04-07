"""
plot_metrics.py — 评估指标可视化脚本
====================================
生成混淆矩阵并保存为图像文件。
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def main():
    """主函数"""
    print("="*60)
    print("KidGuard 评估指标可视化")
    print("="*60)
    
    # 检查必要的文件
    model_path = "models/svm_baseline.pkl"
    scaler_path = "models/scaler.pkl"
    test_results_path = "models/test_predictions.csv"
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 '{model_path}' 不存在")
        print("请先运行 train_svm_baseline.py 训练模型")
        return
    
    if not os.path.exists(test_results_path):
        print(f"错误: 测试结果文件 '{test_results_path}' 不存在")
        print("请先运行 train_svm_baseline.py 生成测试结果")
        return
    
    # 加载测试结果
    print("正在加载测试结果...")
    test_results = pd.read_csv(test_results_path)
    y_true = test_results['true_label'].values
    y_pred = test_results['pred_label'].values
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 创建可视化
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['正常', '冲突'],
                yticklabels=['正常', '冲突'])
    plt.title('混淆矩阵 (Confusion Matrix)')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    
    # 确保日志目录存在
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 保存图像
    output_path = os.path.join(log_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"混淆矩阵已保存到: {output_path}")
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=['正常', '冲突']))
    
    # 计算并打印其他指标
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    
    print("\n性能指标:")
    print(f"  准确率 (Accuracy): {accuracy:.4f}")
    print(f"  精确率 (Precision): {precision:.4f}")
    print(f"  召回率 (Recall): {recall:.4f}")
    
    print("="*60)

if __name__ == "__main__":
    main()
