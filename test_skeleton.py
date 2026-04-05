import sys, os, csv
sys.path.insert(0, "src")

from fightguard.inputs.skeleton_source import load_dataset
from fightguard.detection.interaction_rules import run_rules_symmetric
from fightguard.config import get_config

cfg      = get_config()
data_dirs = [
    "D:/dataset_1/nturgbd_skeletons_s001_to_s017",
    "D:/dataset_1/nturgbd_skeletons_s018_to_s032",
]

# 加载500个clip做更可靠的评估
track_sets = load_dataset(data_dirs, max_clips=500)

tp = fp = tn = fn = 0
rows = []  # 用于写CSV

for ts in track_sets:
    events    = run_rules_symmetric(ts, cfg)
    predicted = 1 if events else 0
    actual    = ts.label
    top_score = round(max((e.score for e in events), default=0.0), 4)
    top_rules = str(events[0].triggered_rules) if events else "[]"

    if actual == 1 and predicted == 1: tp += 1
    elif actual == 0 and predicted == 1: fp += 1
    elif actual == 0 and predicted == 0: tn += 1
    elif actual == 1 and predicted == 0: fn += 1

    rows.append({
        "clip_id":   ts.clip_id,
        "actual":    actual,
        "predicted": predicted,
        "result":    "TP" if actual==1 and predicted==1 else
                     "FP" if actual==0 and predicted==1 else
                     "TN" if actual==0 and predicted==0 else "FN",
        "top_score": top_score,
        "rules":     top_rules,
    })

# ── 写入CSV ──────────────────────────────────────────────────
os.makedirs("outputs/metrics", exist_ok=True)
csv_path = "outputs/metrics/eval_results.csv"
with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# ── 输出最终指标 ──────────────────────────────────────────────
total     = tp + fp + tn + fn
recall    = tp / (tp + fn)    if (tp + fn) > 0 else 0.0
precision = tp / (tp + fp)    if (tp + fp) > 0 else 1.0
fpr       = fp / (tn + fp)    if (tn + fp) > 0 else 0.0
acc       = (tp + tn) / total if total > 0       else 0.0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

print(f"{'='*55}")
print(f"  KidGuard 基础版规则流 — 正式评测报告")
print(f"  数据集：NTU RGBD s001~s032  样本数：{total}")
print(f"  冲突样本：{tp+fn}   正常样本：{tn+fp}")
print(f"{'='*55}")
print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
print(f"  Accuracy  (准确率) : {acc:.4f}")
print(f"  Precision (精确率) : {precision:.4f}")
print(f"  Recall    (召回率) : {recall:.4f}")
print(f"  F1 Score           : {f1:.4f}")
print(f"  FPR       (误报率) : {fpr:.4f}")
print(f"{'='*55}")
print(f"  结果已保存至：{csv_path}")
