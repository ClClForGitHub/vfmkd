# 贡献指南

感谢您对 VFMKD 项目的关注！我们欢迎各种形式的贡献。

## 如何贡献

### 报告问题

如果您发现了bug或有功能建议：

1. 检查 [Issues](https://github.com/yourusername/vfmkd/issues) 页面，确认问题是否已被报告
2. 如果没有，创建一个新的 Issue，并提供：
   - 清晰的标题和描述
   - 重现步骤（如果是bug）
   - 预期行为和实际行为
   - 您的环境信息（Python版本、PyTorch版本等）

### 提交代码

1. **Fork 项目**
   ```bash
   # Fork 仓库到您的账户
   # 然后克隆您的fork
   git clone https://github.com/yourusername/vfmkd.git
   cd vfmkd
   ```

2. **创建分支**
   ```bash
   git checkout -b feature/your-feature-name
   # 或
   git checkout -b fix/your-bug-fix
   ```

3. **设置开发环境**
   ```bash
   pip install -r requirements.txt
   pip install -e .[dev]  # 安装开发依赖
   ```

4. **进行更改**
   - 编写代码
   - 添加测试
   - 更新文档

5. **运行测试**
   ```bash
   pytest tests/
   ```

6. **代码格式化**
   ```bash
   black vfmkd/ tools/ tests/
   flake8 vfmkd/ tools/ tests/
   ```

7. **提交更改**
   ```bash
   git add .
   git commit -m "简短描述您的更改"
   ```

8. **推送到GitHub**
   ```bash
   git push origin feature/your-feature-name
   ```

9. **创建 Pull Request**
   - 访问您的 fork 页面
   - 点击 "New Pull Request"
   - 填写 PR 描述，说明您的更改

## 代码规范

### Python 代码风格

- 遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 规范
- 使用 `black` 进行代码格式化
- 使用 `flake8` 进行代码检查
- 使用类型注解（Type Hints）

### 提交信息规范

使用清晰的提交信息：

```
类型: 简短描述（50字符以内）

详细描述（可选，72字符换行）

- 要点1
- 要点2
```

类型包括：
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式（不影响代码运行）
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

### 文档

- 为公开的函数、类添加 docstring
- 使用 Google 风格的 docstring
- 更新 README.md（如果需要）

示例：
```python
def extract_features(image: np.ndarray, model: torch.nn.Module) -> torch.Tensor:
    """从图像中提取特征。
    
    Args:
        image: 输入图像，shape为(H, W, C)
        model: 特征提取模型
        
    Returns:
        提取的特征，shape为(C, H', W')
        
    Raises:
        ValueError: 如果图像格式不正确
    """
    pass
```

## 开发建议

### 添加新的 Backbone

1. 在 `vfmkd/models/backbones/` 创建新文件
2. 继承 `BaseBackbone` 类
3. 实现必要的方法
4. 在 `configs/backbones/` 添加配置
5. 添加测试用例
6. 更新文档

### 添加新的损失函数

1. 在 `vfmkd/distillation/losses/` 创建新文件
2. 实现损失函数
3. 添加单元测试
4. 在配置文件中注册
5. 更新文档

### 添加新的教师模型

1. 在 `vfmkd/teachers/` 创建新文件
2. 继承 `BaseTeacher` 类
3. 实现特征提取方法
4. 在 `configs/teachers/` 添加配置
5. 添加测试用例
6. 更新文档

## 测试

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_backbones.py

# 查看覆盖率
pytest --cov=vfmkd tests/
```

### 编写测试

- 为新功能添加测试
- 确保测试覆盖率不低于80%
- 使用有意义的测试名称
- 包含正常情况和边界情况

## 审查过程

1. 提交 PR 后，维护者会进行审查
2. 可能会收到修改建议
3. 请及时响应并更新代码
4. 所有检查通过后，PR将被合并

## 行为准则

- 尊重他人
- 接受建设性批评
- 专注于对项目最有利的事情
- 对社区成员表示同理心

## 问题？

如有任何问题，欢迎：
- 创建 Issue
- 发送邮件至 vfmkd@example.com
- 加入我们的讨论组

再次感谢您的贡献！

