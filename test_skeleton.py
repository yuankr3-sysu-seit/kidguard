import sys
sys.path.insert(0, "src")

from fightguard.inputs.skeleton_source import load_dataset
from fightguard.detection.pairing import (
    get_interaction_pairs,
    get_proximity_frames,
    compute_pair_distance_at_frame,
)
from fightguard.config import get_config

cfg = get_config()
data_dirs = ["D:/dataset_1/nturgbd_skeletons_s001_to_s017"]
tracks = load_dataset(data_dirs, max_clips=20)

# 找第一个有2人的冲突clip
target = None
for t in tracks:
    if len(t.tracks) >= 2 and t.label == 1:
        target = t
        break

if target:
    a, b = target.tracks[0], target.tracks[1]
    print(f"Clip: {target.clip_id}, 总帧数: {target.total_frames}")

    # 验证归一化后的坐标范围
    kp = a.keypoints[0]
    print(f"归一化后左手腕: {kp['left_wrist']}")
    print(f"归一化后右肩:   {kp['right_shoulder']}")

    # 验证距离范围
    dists = []
    for i in range(min(len(a.keypoints), len(b.keypoints))):
        d = compute_pair_distance_at_frame(a, b, i)
        if d is not None:
            dists.append(d)
    print(f"归一化后距离 — 最小: {min(dists):.4f}, 最大: {max(dists):.4f}, 均值: {sum(dists)/len(dists):.4f}")

    # 验证近身帧
    pairs = get_interaction_pairs(target, cfg)
    if pairs:
        a2, b2 = pairs[0]
        prox = get_proximity_frames(a2, b2, cfg["rules"]["proximity_threshold"])
        print(f"近身帧数量（阈值={cfg['rules']['proximity_threshold']}）: {len(prox)} / {target.total_frames}")
        print(f"近身帧索引（前10个）: {prox[:10]}")
