
import torch.nn as nn

# 导入我们项目自定义的模型组件
from .backbones.yolov8_backbone import YOLOv8Backbone
from .necks.yolov8_pafpn import YOLOv8PAFPN
from .heads.detection.yolov8_detect_head import YOLOv8DetectHead

class AssembledDetectionModel(nn.Module):
    """
    一个用于将Backbone, Neck, Head拼接在一起的容器模型。
    这个独立的类定义解决了在不同脚本中加载模型时遇到的 'AttributeError'。
    """
    def __init__(self, backbone: YOLOv8Backbone, neck: YOLOv8PAFPN, head: YOLOv8DetectHead, yaml: dict):
        super().__init__()
        # 为了与Ultralytics的内部结构保持最大兼容性，我们将模块列表命名为'model'
        self.model = nn.Sequential(backbone, neck, head)
        self.yaml = yaml
    
    def forward(self, x):
        # Ultralytics在处理模型时，会直接调用 model(x) 或 model.forward(x)
        return self.model(x)
