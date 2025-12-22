# VFMKD项目长期记忆系统

## 概述

VFMKD项目长期记忆系统旨在让项目计划能够控制更多对话，通过记录项目状态、任务进度、技术债务和实验记录，确保每次对话都能基于完整的项目上下文进行。

## 文件结构

```
VFMKD/
├── vfmkd-project-setup.plan.md    # 项目计划文档
├── PROJECT_STATUS.md              # 项目状态跟踪
├── LONG_TERM_MEMORY.md           # 长期记忆配置
├── memory.json                    # 记忆数据文件（自动生成）
└── tools/
    └── memory_manager.py          # 记忆管理器
```

## 核心功能

### 1. 项目状态跟踪
- **任务管理**: 记录已完成、进行中、待完成的任务
- **进度统计**: 自动计算项目完成率
- **技术债务**: 跟踪待解决的问题
- **实验记录**: 记录所有实验和结果

### 2. 对话历史管理
- **对话总结**: 记录每次对话的关键内容
- **关键点提取**: 识别重要的技术决策
- **上下文保持**: 确保后续对话的连续性

### 3. 性能指标跟踪
- **特征提取速度**: 记录模型性能
- **内存使用**: 跟踪资源消耗
- **存储效率**: 监控存储优化

## 使用方法

### 基本使用

```python
from tools.memory_manager import VFMKDMemoryManager

# 创建记忆管理器
manager = VFMKDMemoryManager()

# 更新任务状态
manager.update_task_status("vit_backbone", "completed", "ViT backbone实现完成")

# 添加实验记录
manager.add_experiment("sam2_large_test", "success", "SAM2.1 Large模型测试成功")

# 添加技术债务
manager.add_technical_debt("特征对齐策略需要优化", "high")

# 更新性能指标
manager.update_performance_metrics({
    "feature_extraction_speed": "1.2s/image",
    "memory_usage": "450MB/image"
})

# 添加对话总结
manager.add_conversation_summary(
    "实现了ViT backbone，测试通过",
    ["ViT backbone完成", "测试覆盖率100%", "性能达标"]
)
```

### 命令行使用

```bash
# 运行记忆管理器
python tools/memory_manager.py

# 查看项目状态
cat PROJECT_STATUS.md

# 查看长期记忆配置
cat LONG_TERM_MEMORY.md
```

## 记忆数据格式

### memory.json结构

```json
{
  "project_info": {
    "name": "VFMKD",
    "description": "Vision Foundation Model Knowledge Distillation",
    "created": "2024-12-19T10:00:00",
    "last_updated": "2024-12-19T15:30:00"
  },
  "completed_tasks": [
    {
      "id": "repvit_backbone",
      "status": "completed",
      "updated": "2024-12-19T10:30:00",
      "details": "RepViT backbone从EdgeSAM迁移完成"
    }
  ],
  "in_progress_tasks": [
    {
      "id": "vit_backbone",
      "status": "in_progress",
      "updated": "2024-12-19T14:00:00",
      "details": "ViT backbone实现中"
    }
  ],
  "pending_tasks": [
    {
      "id": "mamba_backbone",
      "status": "pending",
      "updated": "2024-12-19T09:00:00",
      "details": "Mamba backbone待实现"
    }
  ],
  "technical_debt": [
    {
      "issue": "特征对齐策略需要优化",
      "priority": "high",
      "created": "2024-12-19T11:00:00",
      "status": "open"
    }
  ],
  "experiments": [
    {
      "name": "sam2_base_test",
      "result": "success",
      "timestamp": "2024-12-19T12:00:00",
      "details": "SAM2.1 Base+模型测试成功"
    }
  ],
  "performance_metrics": {
    "feature_extraction_speed": "1.2s/image",
    "memory_usage": "450MB/image",
    "last_updated": "2024-12-19T15:00:00"
  },
  "conversation_history": [
    {
      "summary": "实现了ViT backbone，测试通过",
      "key_points": ["ViT backbone完成", "测试覆盖率100%", "性能达标"],
      "timestamp": "2024-12-19T15:30:00"
    }
  ]
}
```

## 集成到对话流程

### 1. 对话开始
- 加载项目记忆
- 检查项目状态
- 识别当前任务优先级

### 2. 对话进行
- 记录关键决策
- 更新任务状态
- 跟踪技术债务

### 3. 对话结束
- 生成对话总结
- 更新项目状态
- 保存记忆数据

## 最佳实践

### 1. 定期更新
- 每次对话后更新记忆
- 每周生成状态报告
- 及时记录技术债务

### 2. 信息质量
- 使用清晰的任务描述
- 记录具体的性能指标
- 保持实验记录的完整性

### 3. 版本控制
- 将memory.json加入.gitignore
- 定期备份重要记忆
- 使用PROJECT_STATUS.md作为主要状态文档

## 故障排除

### 常见问题

1. **记忆文件损坏**
   - 删除memory.json，系统会重新创建
   - 检查JSON格式是否正确

2. **任务状态混乱**
   - 使用memory_manager.py重新整理
   - 检查任务ID是否唯一

3. **性能指标不准确**
   - 定期校准测量工具
   - 使用标准化的测试环境

## 扩展功能

### 1. 自动化集成
- 集成到CI/CD流程
- 自动生成状态报告
- 智能任务优先级调整

### 2. 可视化
- 生成项目进度图表
- 性能指标趋势分析
- 技术债务热力图

### 3. 协作功能
- 多用户记忆共享
- 任务分配和跟踪
- 评论和讨论记录

## 注意事项

1. **隐私保护**: 不要记录敏感信息
2. **数据安全**: 定期备份重要记忆
3. **性能影响**: 避免过度记录，保持系统响应速度
4. **兼容性**: 确保记忆格式向前兼容

---

**创建时间**: 2024-12-19
**版本**: v1.0
**维护者**: AI Assistant












