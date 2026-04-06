import itertools
from typing import List, Tuple
from fightguard.contracts import TrackSet, SkeletonTrack
from fightguard.detection.math_utils import get_body_center_formula, euclidean_distance

def compute_pair_distance_at_frame(track_a: SkeletonTrack, track_b: SkeletonTrack, frame_idx: int) -> float:
    if frame_idx >= len(track_a.keypoints) or frame_idx >= len(track_b.keypoints):
        return None
    ca = get_body_center_formula(track_a.keypoints[frame_idx])
    cb = get_body_center_formula(track_b.keypoints[frame_idx])
    if ca is None or cb is None: return None
    return euclidean_distance(ca, cb)

def get_interaction_pairs(track_set: TrackSet, cfg: dict) -> List[Tuple[SkeletonTrack, SkeletonTrack]]:
    """
    筛选交互对。
    【实战优化】：增加存活时间过滤，剔除 YOLO 产生的碎片化幽灵 ID。
    """
    valid_tracks = []
    for t in track_set.tracks:
        # 统计真实存活的物理帧数（非全零填充帧）
        alive_count = sum(1 for kp in t.keypoints if kp.get("nose", [0,0,0])[:2] != [0.0, 0.0])
        if alive_count >= 15: # 必须在画面里活过 0.5 秒
            valid_tracks.append(t)
            
    # 如果过滤后没人了，降级兜底
    if len(valid_tracks) < 2:
        valid_tracks = track_set.tracks 
        
    pairs = []
    best_pair = None
    min_avg_dist = float('inf')

    for a, b in itertools.combinations(valid_tracks, 2):
        total_dist = 0.0
        valid_count = 0
        for fi in range(len(a.keypoints)):
            ca = get_body_center_formula(a.keypoints[fi])
            cb = get_body_center_formula(b.keypoints[fi])
            if ca and cb:
                total_dist += euclidean_distance(ca, cb)
                valid_count += 1
        
        if valid_count > 0:
            avg_dist = total_dist / valid_count
            if avg_dist < min_avg_dist:
                min_avg_dist = avg_dist
                best_pair = (a, b)
                
    if best_pair:
        pairs.append(best_pair)
        
    return pairs
