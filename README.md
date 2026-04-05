# KidGuard — 幼儿园冲突风险管理分析系统

中山大学大创项目 | 电子与信息工程学院

## 项目简介

基于计算机视觉，通过骨骼关键点空间几何关系构建规则库，
实现幼儿园场景下冲突行为的轻量化识别与风险管理分析。

## 环境配置

```bash
conda activate fightguard
pip install -r requirements.txt
```

## 运行方式
### 阶段一：骨骼数据规则验证
```bash
python scripts/run_skeleton_interaction.py
```
### 阶段二：视频端到端（待实现）
```bash
python scripts/run_video_end2end.py
```


## 项目架构
kidguard/
├── configs/          # 全局参数与规则阈值配置
├── scripts/          # 各阶段运行入口
├── src/fightguard/   # 核心包
│   ├── contracts.py      # 数据契约（COCO-17 字典标准）
│   ├── config.py         # 配置读取
│   ├── inputs/           # 数据读取模块
│   ├── detection/        # 配对与规则判定
│   ├── evaluation/       # 评测指标
│   └── reporting/        # 日志与可视化输出
├── data/             # 数据集（不上传 Git）
└── outputs/          # 运行结果（不上传 Git）
