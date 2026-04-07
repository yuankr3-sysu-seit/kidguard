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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns

# 尝试导入SMOTE
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("警告: imblearn 未安装，将使用手动过采样")

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
    
    # 检查必要的列（更新为增强特征）
    required_feature_cols = [
        'accel_mean', 'accel_var', 'accel_range', 'accel_energy',
        'rel_vel_mean', 'rel_vel_var', 'rel_vel_range', 'rel_vel_energy'
    ]
    # 确保列存在，如果某些列不存在，使用可用的列
    available_feature_cols = [col for col in required_feature_cols if col in df.columns]
    if len(available_feature_cols) < 4:
        print("警告: 特征列不足，尝试使用基本特征")
        basic_cols = ['accel_mean', 'accel_var', 'rel_vel_mean', 'rel_vel_var']
        available_feature_cols = [col for col in basic_cols if col in df.columns]
    
    required_meta_cols = ['label']
    
    for col in required_feature_cols + required_meta_cols:
        if col not in df.columns:
            print(f"错误: 缺少必要的列 '{col}'")
            return
    
    # 准备特征和标签
    X = df[available_feature_cols].values
    y = df['label'].values
    
    print(f"使用的特征列: {available_feature_cols}")
    
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
    
    # 处理类别不平衡
    print("\n处理类别不平衡...")
    conflict_indices = np.where(y_train == 1)[0]
    normal_indices = np.where(y_train == 0)[0]
    
    print(f"训练集中 - 正常样本: {len(normal_indices)}, 冲突样本: {len(conflict_indices)}")
    
    if len(conflict_indices) == 0:
        print("错误: 训练集中没有冲突样本!")
        return
    
    # 过采样冲突样本
    if SMOTE_AVAILABLE:
        print("使用SMOTE进行过采样...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    else:
        print("使用手动复制进行过采样...")
        # 计算需要复制的次数以达到平衡
        n_conflict = len(conflict_indices)
        n_normal = len(normal_indices)
        
        if n_conflict < n_normal:
            # 复制冲突样本
            repeat_times = n_normal // n_conflict
            X_conflict = X_train_scaled[conflict_indices]
            y_conflict = y_train[conflict_indices]
            
            X_conflict_repeated = np.repeat(X_conflict, repeat_times, axis=0)
            y_conflict_repeated = np.repeat(y_conflict, repeat_times, axis=0)
            
            # 组合
            X_train_resampled = np.vstack([X_train_scaled, X_conflict_repeated])
            y_train_resampled = np.hstack([y_train, y_conflict_repeated])
        else:
            X_train_resampled = X_train_scaled
            y_train_resampled = y_train
    
    print(f"过采样后训练集大小: {X_train_resampled.shape[0]}")
    print(f"过采样后类别分布: 正常={np.sum(y_train_resampled==0)}, 冲突={np.sum(y_train_resampled==1)}")
    
    # 训练SVM分类器（改进版）
    print("\n正在训练SVM分类器 (RBF核, class_weight='balanced')...")
    svm = SVC(kernel='rbf', class_weight='balanced', C=10.0, probability=True, random_state=42)
    svm.fit(X_train_resampled, y_train_resampled)
    
    # 在测试集上评估
    y_pred = svm.predict(X_test_scaled)
    
    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print("\n测试集性能指标:")
    print(f"  准确率 (Accuracy): {accuracy:.4f}")
    print(f"  精确率 (Precision): {precision:.4f}")
    print(f"  召回率 (Recall): {recall:.4f}")
    print(f"  F1-score: {f1:.4f}")
    
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred, target_names=['正常', '冲突']))
    
    # 创建模型目录
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['正常', '冲突'], 
                yticklabels=['正常', '冲突'])
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    cm_path = os.path.join(model_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"混淆矩阵已保存到: {cm_path}")
    
    # 如果召回率仍然为0，尝试随机森林
    if recall == 0:
        print("\n警告: 召回率为0，尝试随机森林分类器...")
        rf = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train_resampled, y_train_resampled)
        y_pred_rf = rf.predict(X_test_scaled)
        
        recall_rf = recall_score(y_test, y_pred_rf, zero_division=0)
        print(f"随机森林召回率: {recall_rf:.4f}")
        
        if recall_rf > recall:
            print("随机森林表现更好，保存随机森林模型")
            model_path_rf = os.path.join(model_dir, "random_forest.pkl")
            with open(model_path_rf, 'wb') as f:
                pickle.dump(rf, f)
            print(f"随机森林模型已保存到: {model_path_rf}")
            
            # 更新预测结果
            y_pred = y_pred_rf
            # 重新计算指标
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            print("\n更新后的测试集性能指标:")
            print(f"  准确率 (Accuracy): {accuracy:.4f}")
            print(f"  精确率 (Precision): {precision:.4f}")
            print(f"  召回率 (Recall): {recall:.4f}")
            print(f"  F1-score: {f1:.4f}")
    
    # 保存SVM模型和标准化器
    model_path = os.path.join(model_dir, "svm_baseline.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(svm, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\nSVM模型已保存到: {model_path}")
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
