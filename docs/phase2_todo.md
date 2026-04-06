# Phase 2 推进清单（4.6 – 5.1）

## 已完成（冻结）
- [x] YOLOv8-Pose + BoT-SORT 端到端
- [x] 时空对齐（全帧补齐）
- [x] 幽灵 ID 剔除（alive_count）
- [x] 熵权法（EWM）权重确定
- [x] 四段式状态机（同步 / 异步尝试）
- [x] Optuna 自动化调参（F1 ≈ 0.75）

## 当前问题（已确认）
- 2D 视频中 ID Switch 频繁
- Limb Swapping 导致 FP
- 遮挡导致关键帧置信度低，γ 抑制过强
- BoT-SORT 在打架场景下不稳定

## 下一步只做三件事（严格限量）
- [ ] 替换追踪器：BoT-SORT → ByteTrack
- [ ] 固定参数，仅评估 tracker 变化带来的影响
- [ ] 对比三组：BoT-SORT / ByteTrack / 无追踪

## 明确不做（避免分心）
- [ ] 不再调规则阈值
- [ ] 不再引入新特征
- [ ] 不讨论 ST-GCN（除非 Phase 2 彻底失败）
