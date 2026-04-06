"""
clip_metrics.py — 评测指标计算模块
===================================
计算系统在数据集上的 Accuracy, Precision, Recall, FPR, F1 等核心指标。
"""

from typing import Dict, List

def calculate_metrics(results: List[Dict]) -> Dict[str, float]:
    """
    根据预测结果列表计算核心评测指标。
    
    参数：
        results: 包含 actual 和 predicted 字段的字典列表
    返回：
        包含各项指标的字典
    """
    tp = fp = tn = fn = 0
    for r in results:
        actual = r.get("actual", -1)
        predicted = r.get("predicted", 0)
        
        if actual == 1 and predicted == 1: tp += 1
        elif actual == 0 and predicted == 1: fp += 1
        elif actual == 0 and predicted == 0: tn += 1
        elif actual == 1 and predicted == 0: fn += 1

    total = tp + fp + tn + fn
    if total == 0:
        return {}

    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    fpr       = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr       = fn / (tp + fn) if (tp + fn) > 0 else 0.0  # 漏报率
    acc       = (tp + tn) / total
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "total": total,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),  # 新增：漏报率
        "f1_score": round(f1, 4)
    }
