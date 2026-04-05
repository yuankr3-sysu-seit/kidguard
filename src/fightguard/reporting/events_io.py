"""
events_io.py — 事件日志持久化模块
==================================
负责将检测到的交互事件和评测结果写入 CSV 或 JSON 文件。
"""

import os
import csv
from typing import Dict, List
from fightguard.contracts import InteractionEvent

def save_eval_results_csv(results: List[Dict], filepath: str) -> None:
    """将评测明细写入 CSV 文件"""
    if not results:
        return
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
def save_events_csv(events: List[InteractionEvent], filepath: str) -> None:
    """将检测到的事件列表写入 CSV 文件"""
    if not events:
        return
        
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        # 使用 InteractionEvent 的 to_dict 方法获取字段
        fieldnames = events[0].to_dict().keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow(event.to_dict())
