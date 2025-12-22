import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class FGDLoss(nn.Module):
    """
    官方FGD核心：矩形area前景/背景掩码 + S/C注意力重加权 + 前景/背景MSE + mask对齐 + 关系约束。
    不引入外部分割权重，后续可替换权重来源。
    """

    def __init__(
        self,
        alpha_fg: float = 1.0,
        beta_bg: float = 0.25,
        alpha_edge: float = 2.0,  # 边缘权重，默认是前景的2倍
        gamma_mask: float = 0.0,
        lambda_rela: float = 0.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha_fg = alpha_fg
        self.beta_bg = beta_bg
        self.alpha_edge = alpha_edge
        self.gamma_mask = gamma_mask
        self.lambda_rela = lambda_rela
        self.temperature = temperature

        self.align_conv: nn.Conv2d | None = None
        self.conv_mask_s: nn.Conv2d | None = None
        self.conv_mask_t: nn.Conv2d | None = None
        self.channel_add_conv_s = None
        self.channel_add_conv_t = None

    @staticmethod
    def _get_attention(x: torch.Tensor, temperature: float) -> Tuple[torch.Tensor, torch.Tensor]:
        n, c, h, w = x.shape
        value = x.abs()
        fea_map = value.mean(dim=1, keepdim=True)  # (N,1,H,W)
        s_attn = (h * w) * F.softmax((fea_map / temperature).view(n, -1), dim=1).view(n, h, w)
        channel_map = value.mean(dim=(2, 3), keepdim=False)  # (N,C)
        c_attn = c * F.softmax(channel_map / temperature, dim=1)  # (N,C)
        return s_attn, c_attn

    @staticmethod
    def _build_fg_bg_masks(
        bboxes_xyxy: List[torch.Tensor], img_size: Tuple[int, int], feat_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hf, wf = feat_size
        hi, wi = img_size
        b = len(bboxes_xyxy)
        device = bboxes_xyxy[0].device if bboxes_xyxy and bboxes_xyxy[0].numel() > 0 else torch.device("cpu")
        fg = torch.zeros((b, hf, wf), dtype=torch.float32, device=device)
        area_map = torch.zeros((b, hf, wf), dtype=torch.float32, device=device)
        for i, boxes in enumerate(bboxes_xyxy):
            if boxes.numel() == 0:
                continue
            scale_x = wf / float(wi)
            scale_y = hf / float(hi)
            xyxy = boxes.clone()
            xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] * scale_x).clamp(0, wf - 1)
            xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] * scale_y).clamp(0, hf - 1)
            xyxy = xyxy.long()
            for j in range(xyxy.size(0)):
                x1, y1, x2, y2 = xyxy[j]
                if x2 <= x1 or y2 <= y1:
                    continue
                hmin, hmax = y1, y2
                wmin, wmax = x1, x2
                area = 1.0 / max(1, (hmax + 1 - hmin)) / max(1, (wmax + 1 - wmin))
                cur = torch.full((hmax - hmin + 1, wmax - wmin + 1), area, device=device)
                # 取最大，保持与官方实现“矩形+area权重”的语义
                old = area_map[i, hmin : hmax + 1, wmin : wmax + 1]
                area_map[i, hmin : hmax + 1, wmin : wmax + 1] = torch.maximum(old, cur)
                fg[i, hmin : hmax + 1, wmin : wmax + 1] = 1.0
        # 背景mask基于前景二值
        bg = 1.0 - (fg > 0).float()
        # 归一化背景为概率分布（仅用于参与权重时稳定，FGD官方即做此归一）
        bg_sum = bg.sum(dim=(1, 2), keepdim=True)
        bg = torch.where(bg_sum > 0, bg / (bg_sum + 1e-6), bg)
        # 前景以area作为值（而非概率），与FGD一致；用于sqrt(Mask_fg)权重。
        return area_map, bg

    def _ensure_align_and_context(self, cs: int, ct: int, device: torch.device) -> None:
        if cs != ct and self.align_conv is None:
            self.align_conv = nn.Conv2d(cs, ct, kernel_size=1, bias=False).to(device)
        if self.conv_mask_s is None:
            self.conv_mask_s = nn.Conv2d(ct, 1, kernel_size=1).to(device)
        if self.conv_mask_t is None:
            self.conv_mask_t = nn.Conv2d(ct, 1, kernel_size=1).to(device)
        if self.channel_add_conv_s is None:
            self.channel_add_conv_s = nn.Sequential(
                nn.Conv2d(ct, ct // 2, kernel_size=1),
                nn.LayerNorm([ct // 2, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(ct // 2, ct, kernel_size=1),
            ).to(device)
        if self.channel_add_conv_t is None:
            self.channel_add_conv_t = nn.Sequential(
                nn.Conv2d(ct, ct // 2, kernel_size=1),
                nn.LayerNorm([ct // 2, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(ct // 2, ct, kernel_size=1),
            ).to(device)

    def _spatial_pool_with_conv(self, x: torch.Tensor, which: str) -> torch.Tensor:
        # 与官方一致：conv_mask -> softmax(HW) -> matmul，得到(N,C,1,1)
        n, c, h, w = x.size()
        input_x = x.view(n, c, h * w).unsqueeze(1)  # (N,1,C,HW)
        if which == 's':
            context_mask = self.conv_mask_s(x)
        else:
            context_mask = self.conv_mask_t(x)
        context_mask = context_mask.view(n, 1, h * w)
        context_mask = F.softmax(context_mask, dim=2).unsqueeze(-1)  # (N,1,HW,1)
        context = torch.matmul(input_x, context_mask).view(n, c, 1, 1)  # (N,C,1,1)
        return context

    def _get_rela_loss(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 官方：context注入 + MSE(reduction='sum')/len(out)
        context_s = self._spatial_pool_with_conv(s, 's')
        context_t = self._spatial_pool_with_conv(t, 't')
        out_s = s + self.channel_add_conv_s(context_s)
        out_t = t + self.channel_add_conv_t(context_t)
        loss_mse = nn.MSELoss(reduction='sum')
        return loss_mse(out_s, out_t) / len(out_s)

    def forward(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        bboxes_xyxy: List[torch.Tensor] | None = None,
        img_size: Tuple[int, int] | None = None,
        fg_map: torch.Tensor | None = None,
        bg_map: torch.Tensor | None = None,
        edge_map: torch.Tensor | None = None,  # 新增：边缘图
    ) -> torch.Tensor:
        # 空间/通道对齐
        if student_feat.shape[2:] != teacher_feat.shape[2:]:
            student_feat = F.interpolate(student_feat, size=teacher_feat.shape[2:], mode="bilinear", align_corners=False)
        cs, ct = student_feat.size(1), teacher_feat.size(1)
        self._ensure_align_and_context(cs, ct, student_feat.device)
        if cs != ct:
            student_feat = self.align_conv(student_feat)

        b, c, h, w = teacher_feat.shape

        # Mask构造（优先使用预计算fg/bg；否则从bbox构造）
        if fg_map is not None and bg_map is not None:
            mask_fg = fg_map
            mask_bg = bg_map
        else:
            assert bboxes_xyxy is not None and img_size is not None, "FGD需要提供bboxes+img_size或fg/bg"
            mask_fg, mask_bg = self._build_fg_bg_masks(bboxes_xyxy, img_size=img_size, feat_size=(h, w))
            mask_fg = mask_fg.unsqueeze(1)
            mask_bg = mask_bg.unsqueeze(1)

        # 注意力
        s_attn_t, c_attn_t = self._get_attention(teacher_feat, self.temperature)
        s_attn_s, c_attn_s = self._get_attention(student_feat, self.temperature)
        c_attn_t_ = c_attn_t.unsqueeze(-1).unsqueeze(-1)
        s_attn_t_ = s_attn_t.unsqueeze(1)

        # 重加权
        fea_t = teacher_feat * torch.sqrt(s_attn_t_) * torch.sqrt(c_attn_t_)
        fea_s = student_feat * torch.sqrt(s_attn_t_) * torch.sqrt(c_attn_t_)

        # 前景/背景加权
        fg_t = fea_t * torch.sqrt(torch.clamp(mask_fg, min=0))
        fg_s = fea_s * torch.sqrt(torch.clamp(mask_fg, min=0))
        bg_t = fea_t * torch.sqrt(torch.clamp(mask_bg, min=0))
        bg_s = fea_s * torch.sqrt(torch.clamp(mask_bg, min=0))

        # 前/背景MSE（官方缩放：/len(Mask)~batch size）
        loss_mse = nn.MSELoss(reduction='sum')
        loss_fg = loss_mse(fg_s, fg_t) / len(mask_fg)
        loss_bg = loss_mse(bg_s, bg_t) / len(mask_bg)

        # 边缘loss（如果提供了edge_map）
        loss_edge = 0.0
        if edge_map is not None and self.alpha_edge > 0:
            # 边缘加权：使用edge_map作为权重
            edge_t = fea_t * torch.sqrt(torch.clamp(edge_map, min=0))
            edge_s = fea_s * torch.sqrt(torch.clamp(edge_map, min=0))
            loss_edge = loss_mse(edge_s, edge_t) / len(edge_map)

        # 掩码对齐（官方：|C_s-C_t|/len(C_s) + |S_s-S_t|/len(S_s)）
        mask_loss = torch.sum(torch.abs(c_attn_s - c_attn_t)) / len(c_attn_s)
        mask_loss = mask_loss + torch.sum(torch.abs(s_attn_s - s_attn_t)) / len(s_attn_s)

        # 关系约束
        rela_loss = self._get_rela_loss(student_feat, teacher_feat)

        # 总loss：前景 + 背景 + 边缘
        loss = self.alpha_fg * loss_fg + self.beta_bg * loss_bg + self.alpha_edge * loss_edge
        if self.gamma_mask > 0:
            loss = loss + self.gamma_mask * mask_loss
        if self.lambda_rela > 0:
            loss = loss + self.lambda_rela * rela_loss
        return loss


