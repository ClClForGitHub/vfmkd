<!-- 8c3ea5d9-5744-43ea-9934-c4abc0f3564c 4660c728-4837-4636-8c79-8fc404e910c9 -->
# VFMKD 项目计划 V2

## 阶段划分

- **阶段1（当前迭代）**：整理项目结构并补齐 YOLOv8 检测链路（Backbone→Neck→Head→COCO 训练脚本）。
- **阶段2（后续迭代）**：Prompt 辨析头实验 + 训练性能优化，待阶段1完成后细化实施细节。

## 阶段1 任务

1. **plan-setup**：将现有计划文档移动/复制到 `plan/` 目录，重命名为 `plan/vfmkd-project-plan-v2.md`，并在文档开头写明“tools/core 禁止移动”的警告与调整引用需同步检查的提示。
2. **structure-refresh**：梳理当前目录 → 输出新版结构图（在 plan 文档中以代码块展示），标记各模块状态：`[已完成]`、“主要功能说明”、`[未完成]`。清理方案：归档历史文件至 `tools/archives/`，核心逻辑仍留在 `tools/core/`。
3. **neck-head-design**：在计划中锁定 YOLOv8 Neck/Head 的实现策略与文件落点（例如 `tools/core/models/necks/yolov8_pafpn.py`、`tools/core/models/heads/yolov8_head.py`），说明多尺寸仅差通道、统一读取配置。
4. **train-coco-plan**：描述 COCO 训练脚本目标（如 `tools/train_coco_detection.py`），强调 `freeze_backbone`、`enable_distill` 开关、训练流程与配置文件需求。

## 阶段2 提前记录

- **prompt-head-roadmap**：列出待决策的框选择策略、实验流程、评估指标占位说明。
- **speed-optimization-roadmap**：总结现有瓶颈与潜在优化（数据加载、AMP、日志节流）供下一阶段细化。

## 注意事项

- 文档中包含显式警告：`tools/core` 为核心目录，严禁移动；迁移文件后必须检索并更新所有引用路径，避免导入错误。
- 计划内容仅覆盖本次对话确定的重点，初版 plan 中的其它 TODO 不再继承，除非在新文档中重新列出。

### To-dos

- [ ] 建立 plan 目录并准备 vfmkd-project-plan-v2.md 初稿
- [ ] 在计划文档中绘制新版目录图并标记完成度
- [ ] 写出 YOLOv8 Neck/Head 实现策略与目标文件
- [ ] 描述 COCO 训练脚本设计与配置需求
- [ ] 整理阶段2 prompt 头与性能优化的路线占位