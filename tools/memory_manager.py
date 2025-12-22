#!/usr/bin/env python3
"""
VFMKD项目记忆更新脚本
用于在每次对话后更新项目状态和长期记忆
"""

import os
import json
import datetime
from pathlib import Path

class VFMKDMemoryManager:
    """VFMKD项目记忆管理器"""
    
    def __init__(self, project_root="VFMKD"):
        self.project_root = Path(project_root)
        self.memory_file = self.project_root / "memory.json"
        self.status_file = self.project_root / "PROJECT_STATUS.md"
        self.plan_file = self.project_root / "vfmkd-project-setup.plan.md"
        
    def load_memory(self):
        """加载现有记忆"""
        if self.memory_file.exists():
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "project_info": {
                "name": "VFMKD",
                "description": "Vision Foundation Model Knowledge Distillation",
                "created": datetime.datetime.now().isoformat(),
                "last_updated": datetime.datetime.now().isoformat()
            },
            "completed_tasks": [],
            "in_progress_tasks": [],
            "pending_tasks": [],
            "technical_debt": [],
            "experiments": [],
            "performance_metrics": {},
            "conversation_history": []
        }
    
    def save_memory(self, memory):
        """保存记忆"""
        memory["project_info"]["last_updated"] = datetime.datetime.now().isoformat()
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)
    
    def update_task_status(self, task_id, status, details=None):
        """更新任务状态"""
        memory = self.load_memory()
        
        # 从所有列表中移除任务
        for task_list in ["completed_tasks", "in_progress_tasks", "pending_tasks"]:
            memory[task_list] = [t for t in memory[task_list] if t.get("id") != task_id]
        
        # 添加到对应列表
        task_info = {
            "id": task_id,
            "status": status,
            "updated": datetime.datetime.now().isoformat(),
            "details": details or ""
        }
        
        if status == "completed":
            memory["completed_tasks"].append(task_info)
        elif status == "in_progress":
            memory["in_progress_tasks"].append(task_info)
        else:
            memory["pending_tasks"].append(task_info)
        
        self.save_memory(memory)
    
    def add_experiment(self, experiment_name, result, details=None):
        """添加实验记录"""
        memory = self.load_memory()
        experiment = {
            "name": experiment_name,
            "result": result,
            "timestamp": datetime.datetime.now().isoformat(),
            "details": details or ""
        }
        memory["experiments"].append(experiment)
        self.save_memory(memory)
    
    def add_technical_debt(self, issue, priority="medium"):
        """添加技术债务"""
        memory = self.load_memory()
        debt = {
            "issue": issue,
            "priority": priority,
            "created": datetime.datetime.now().isoformat(),
            "status": "open"
        }
        memory["technical_debt"].append(debt)
        self.save_memory(memory)
    
    def update_performance_metrics(self, metrics):
        """更新性能指标"""
        memory = self.load_memory()
        memory["performance_metrics"].update(metrics)
        memory["performance_metrics"]["last_updated"] = datetime.datetime.now().isoformat()
        self.save_memory(memory)
    
    def add_conversation_summary(self, summary, key_points=None):
        """添加对话总结"""
        memory = self.load_memory()
        conversation = {
            "summary": summary,
            "key_points": key_points or [],
            "timestamp": datetime.datetime.now().isoformat()
        }
        memory["conversation_history"].append(conversation)
        
        # 只保留最近10次对话
        memory["conversation_history"] = memory["conversation_history"][-10:]
        
        self.save_memory(memory)
    
    def get_project_summary(self):
        """获取项目总结"""
        memory = self.load_memory()
        return {
            "total_tasks": len(memory["completed_tasks"]) + len(memory["in_progress_tasks"]) + len(memory["pending_tasks"]),
            "completed": len(memory["completed_tasks"]),
            "in_progress": len(memory["in_progress_tasks"]),
            "pending": len(memory["pending_tasks"]),
            "completion_rate": len(memory["completed_tasks"]) / max(1, len(memory["completed_tasks"]) + len(memory["in_progress_tasks"]) + len(memory["pending_tasks"])),
            "technical_debt_count": len(memory["technical_debt"]),
            "experiments_count": len(memory["experiments"])
        }
    
    def generate_status_report(self):
        """生成状态报告"""
        memory = self.load_memory()
        summary = self.get_project_summary()
        
        report = f"""# VFMKD项目状态报告

## 项目概览
- **项目名称**: {memory['project_info']['name']}
- **描述**: {memory['project_info']['description']}
- **最后更新**: {memory['project_info']['last_updated']}

## 任务进度
- **总任务数**: {summary['total_tasks']}
- **已完成**: {summary['completed']}
- **进行中**: {summary['in_progress']}
- **待完成**: {summary['pending']}
- **完成率**: {summary['completion_rate']:.1%}

## 技术债务
- **待解决问题**: {summary['technical_debt_count']}

## 实验记录
- **实验总数**: {summary['experiments_count']}

## 最近对话
"""
        
        for conv in memory["conversation_history"][-3:]:
            report += f"- **{conv['timestamp']}**: {conv['summary']}\n"
        
        return report

def main():
    """主函数"""
    manager = VFMKDMemoryManager()
    
    # 示例使用
    print("VFMKD记忆管理器")
    print("=" * 50)
    
    # 获取项目总结
    summary = manager.get_project_summary()
    print(f"项目完成率: {summary['completion_rate']:.1%}")
    print(f"技术债务: {summary['technical_debt_count']}个")
    
    # 生成状态报告
    report = manager.generate_status_report()
    print("\n状态报告:")
    print(report)

if __name__ == "__main__":
    main()











