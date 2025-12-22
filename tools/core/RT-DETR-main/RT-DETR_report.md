# RT-DETR-main 调研摘要

## 1. 项目定位与代码结构
- `RT-DETR-main/` 同时包含原始 RT-DETR 与 RT-DETRv2（Paddle 与 PyTorch 两套实现）。PyTorch 版本分别位于 `rtdetr_pytorch/` 与 `rtdetrv2_pytorch/`，提供从训练、评估、导出到部署（Torch/ONNX/TensorRT/OpenVINO）的全流程脚本。
- `hubconf.py` 暴露了 Torch Hub 接口，可一键构建/加载官方发布的 RT-DETR 与 RT-DETRv2 权重，默认使用 `rtdetrv2_pytorch` 的配置体系。
- `benchmark/`、`references/` 等子目录覆盖部署示例与性能脚本；根目录还附带完整论文 PDF 便于查阅算法细节。

## 2. PyTorch 实现能力
1. **配置化建模**：通过 `configs/` + `src/core/YAMLConfig` 递归合并多个 YAML，实例化 `RTDETR` 主体（`backbone`、`HybridEncoder`、`RTDETRTransformer`）以及损失、后处理、优化器、数据加载器等组件。
2. **训练/微调/恢复**：`tools/train.py` 支持单/多卡训练（torchrun）、AMP、EMA、断点恢复（`-r`）与“tuning”热启动（`-t` 仅加载模型权重，自动忽略缺失/不匹配参数）。
3. **导出部署**：`tools/export_onnx.py`、`references/deploy/*.py` 覆盖 ONNX、TensorRT、ONNXRuntime、Torch 推理；另有 `tools/export_trt.py`、`run_profile.py` 等工具统计 FLOPs/吞吐。
4. **数据扩展**：`configs/dataset` 提供 COCO 默认配置；自定义数据只需关闭 `remap_mscoco_category` 并在 YAML 中切换路径，官方 README 也给出具体步骤。

## 3. 关键模块概览
- **RTDETR 主体**：`src/zoo/rtdetr/rtdetr.py` 定义模块注入点（`backbone`、`encoder`、`decoder`），forward 先做多尺度 resize，再串联 backbone→encoder→decoder。
- **Backbone 选项**：
  - `PResNet`（`src/nn/backbone/presnet.py`）覆盖 ResNet18/34/50/101-vd，支持 `freeze_at`、`freeze_norm`、`return_idx` 多特征输出，`pretrained=True` 时会自动从官方 URL 下载“vd”变种权重。
  - `regnet.py`、`dla.py` 则提供 RegNet、DLA34 等轻量备选，方便在 YAML 中替换 `RTDETR.backbone`。
- **训练引擎**：`src/solver/det_solver.py` + `det_engine.py` 负责循环、日志、COCO 评测；`load_tuning_state` 会从 checkpoint 中提取 `model` 或 `ema.module`，并按键名匹配写入当前模型。

## 4. 与我们蒸馏 Backbone 的衔接方案
> 目标：先在自研蒸馏脚本中训练 backbone，再将其注入 RT-DETR 进行 COCO 等下游训练；在“不要改动上游代码”的前提下，可以利用现有 `tuning`/checkpoint 机制完成权重移植。

1. **保持结构一致**：确保蒸馏模型的结构与 RT-DETR 选择的 backbone 类（如 `PResNet` 的 depth/variant/return_idx）完全一致，这样 state_dict 的键名和张量形状才能对齐。
2. **准备权重字典**：在蒸馏脚本结束后，把 `nn.Module.state_dict()` 映射到 RT-DETR 期望的命名空间（例：`backbone.conv1.*`、`backbone.res_layers.0.blocks.0.branch2a.*`）。最简单的方法是：
   - 载入官方 RT-DETR checkpoint；
   - 用蒸馏得到的张量覆盖其中 `state['model'][k]` 里所有 `k` 以 `backbone.` 开头的条目；
   - 保存为新的 `tuning` checkpoint。
3. **热启动加载**：训练 RT-DETR 时执行  
   `torchrun ... tools/train.py -c <config> -t path/to/new_backbone_ckpt.pth`  
   `load_tuning_state` 会自动将匹配到的 backbone 参数写入模型，其它模块保持随机初始化或沿用旧权重。
4. **如需只替换 backbone**：若未来想避免手动改 checkpoint，可在自研脚本里直接构造一个字典：`{'model': backbone_state, 'ema': None}`，只要键名符合 `backbone.*`，`load_tuning_state` 也能完成匹配，其余未提供的参数会被忽略。
5. **推理部署**：调好后的模型可以按官方流程导出 ONNX/TRT，或通过 `hubconf.py` 的自定义条目向团队分享新 backbone 对应的整包权重。

## 5. 后续建议
1. **梳理 Backbone 命名映射**：在我们现有蒸馏脚本里记录 RT-DETR `backbone` 键名（可直接打印官方 checkpoint 的 `backbone` 子字典）以免对齐错误。
2. **自动化打包脚本**：编写一个小工具，把蒸馏得到的权重注入官方 checkpoint 并保存为 `-t` 可用的文件，减少人工操作。
3. **验证流程**：注入后先运行 `--test-only` 验证 COCO 验证集精度，以确认加载成功，再启动正式训练。
4. **关注 RT-DETRv2**：如果未来要利用 Bag-of-Freebies 或更高 AP，可以同样方式替换 `rtdetrv2_pytorch` 中的 backbone；其配置和加载方式与 v1 保持一致。

> 备注：更多细节可随时参考 `README.md`、`README_cn.md` 以及附带的两份 PDF 论文。

