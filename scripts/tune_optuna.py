"""
tune_optuna.py — 极限压榨：基于贝叶斯优化的自动化调参脚本
===========================================================
将感知层(YOLO)与认知层(规则)解耦。先将骨骼轨迹提取至内存，
然后利用 Optuna 框架在认知层进行数百次超参数组合搜索，
寻找当前架构在真实数据集上的真正物理天花板（最高 F1-Score）。
"""

import sys
import os
import random
import optuna
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from fightguard.inputs.video_source import process_video_to_trackset
from fightguard.detection.interaction_rules import run_rules_on_clip
from fightguard.config import get_config

def run_tuning(num_samples: int = 15):
    print("="*60)
    print(" 启动极限压榨：Optuna 自动化超参数搜索")
    print("="*60)
    
    cfg = get_config()
    dataset_dir = "D:/dataset_1/five_dataset"
    fight_dir = os.path.join(dataset_dir, "fight")
    nofight_dir = os.path.join(dataset_dir, "nofight")
    
    if not os.path.exists(fight_dir) or not os.path.exists(nofight_dir):
        print(f"[ERROR] 找不到视频目录: {dataset_dir}")
        return

    fight_files = [os.path.join(fight_dir, f) for f in os.listdir(fight_dir) if f.endswith(('.mp4', '.avi'))]
    nofight_files = [os.path.join(nofight_dir, f) for f in os.listdir(nofight_dir) if f.endswith(('.mp4', '.avi'))]
    
    # 随机抽取样本构建调参验证集
    random.seed(42)
    sampled_fights = random.sample(fight_files, min(num_samples, len(fight_files)))
    sampled_nofights = random.sample(nofight_files, min(num_samples, len(nofight_files)))
    
    test_cases = [(f, 1) for f in sampled_fights] + [(f, 0) for f in sampled_nofights]
    random.shuffle(test_cases)
    
    print(f"[1/3] 感知层解耦：正在提取 {len(test_cases)} 个视频的骨骼轨迹至内存...")
    print("      (此过程较慢，但只需执行一次！)")
    
    cached_data = []
    for video_path, actual_label in tqdm(test_cases, desc="提取轨迹"):
        track_set = process_video_to_trackset(video_path, label=actual_label, cfg=cfg)
        if track_set and len(track_set.tracks) >= 2:
            cached_data.append({"track_set": track_set, "actual": actual_label})

    print(f"[2/3] 提取完毕！成功缓存 {len(cached_data)} 个有效交互样本。")
    print("      即将启动 Optuna 认知层参数轰炸...")

    # ==========================================
    # 定义 Optuna 目标函数
    # ==========================================
    def objective(trial):
        # 1. 让 Optuna 自由组合参数 (定义搜索空间)
        test_cfg = get_config()
        rules = test_cfg["rules"]
        
        # 【2D 场景搜索空间重构】
        # 置信度抑制阈值：YOLO 的置信度分布与 NTU 不同
        rules["tau_c"] = trial.suggest_float("tau_c", 0.3, 0.8)
        
        # 肢体加速度阈值：已除以肩宽归一化，2D 像素空间值较小
        rules["tau_a"] = trial.suggest_float("tau_a", 0.1, 2.0)
        
        # 相对接近速度阈值：已归一化，2D 像素空间值较小
        rules["tau_v"] = trial.suggest_float("tau_v", 0.1, 2.0)
        
        # 躯干倾角阈值：角度制，2D 像素空间变化范围
        rules["tau_phi"] = trial.suggest_float("tau_phi", 5.0, 45.0)
        
        # 骨盆速度阈值：已归一化
        rules["tau_p"] = trial.suggest_float("tau_p", 0.1, 2.0)
        
        # 状态机时序参数
        rules["memory_window"] = trial.suggest_int("memory_window", 5, 30)
        rules["M"] = trial.suggest_int("M", 3, 12)
        rules["alert_threshold"] = trial.suggest_float("alert_threshold", 0.10, 0.40)
        
        # 2. 在缓存数据上跑规则引擎
        tp = fp = tn = fn = 0
        for data in cached_data:
            events = run_rules_on_clip(data["track_set"], test_cfg)
            predicted = 1 if events else 0
            actual = data["actual"]
            
            if predicted == 1 and actual == 1: tp += 1
            elif predicted == 1 and actual == 0: fp += 1
            elif predicted == 0 and actual == 0: tn += 1
            elif predicted == 0 and actual == 1: fn += 1
            
        # 3. 计算 F1-Score (平衡精确率和召回率)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return f1_score

    # ==========================================
    # 启动 100 次迭代轰炸
    # ==========================================
    optuna.logging.set_verbosity(optuna.logging.WARNING) # 只显示警告，保持控制台干净
    study = optuna.create_study(direction="maximize")
    
    print("[3/3] 正在执行 100 次参数变异与测试...")
    # n_trials=100 表示尝试 100 组不同的参数组合
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    
    print("" + "="*60)
    print(" 极限压榨完成！当前架构的真实天花板参数已找到：")
    print("="*60)
    print(f"  最高 F1-Score: {study.best_value:.4f}")
    print("  最佳参数组合:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"    - {key}: {value:.4f}")
        else:
            print(f"    - {key}: {value}")
    print("="*60)
    print("请将上述最佳参数填入 default.yaml 或交互规则代码中！")

if __name__ == "__main__":
    # 抽取 30 个视频 (15正+15负) 作为调参验证集
    run_tuning(num_samples=15)
