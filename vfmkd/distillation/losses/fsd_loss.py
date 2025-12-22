import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class FSDLikeLoss(nn.Module):
    """
    FSD核心：基于GT的高斯外扩+area构造平滑前景分布，前景/背景做KL分布对齐；
    可叠加S/C注意力与可选mask/关系正则；默认不引入外部分割权重来源。
    """

    def __init__(
        self,
        weight_fg: float = 1.0,
        weight_bg: float = 0.2,
        temperature: float = 1.0,
        gamma_mask: float = 0.0,
        lambda_rela: float = 0.0,
        gaussian_from_mask: bool = True,
        gaussian_mix: str = "max",  # "max" | "blend"
        gaussian_blend_lambda: float = 0.5,
    ) -> None:
        super().__init__()
        self.weight_fg = weight_fg
        self.weight_bg = weight_bg
        self.temperature = temperature
        self.gamma_mask = gamma_mask
        self.lambda_rela = lambda_rela
        self.gaussian_from_mask = gaussian_from_mask
        self.gaussian_mix = gaussian_mix
        self.gaussian_blend_lambda = gaussian_blend_lambda
        self.align_conv: nn.Conv2d | None = None

        self.channel_add_conv_s = None
        self.channel_add_conv_t = None

    @staticmethod
    def _feature_norm(x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        xp = x.permute(1, 0, 2, 3).reshape(c, -1)
        mean = xp.mean(dim=-1, keepdim=True)
        std = xp.std(dim=-1, keepdim=True)
        xp = (xp - mean) / (std + 1e-6)
        return xp.reshape(c, n, h, w).permute(1, 0, 2, 3)

    @staticmethod
    def _softmax_spatial(x: torch.Tensor, tau: float) -> torch.Tensor:
        n, c, h, w = x.shape
        x = x.reshape(n * c, h * w) / tau
        return F.softmax(x, dim=1)

    @staticmethod
    def _get_attention(x: torch.Tensor, temperature: float) -> Tuple[torch.Tensor, torch.Tensor]:
        n, c, h, w = x.shape
        value = x.abs()
        fea_map = value.mean(dim=1, keepdim=True)
        s_attn = (h * w) * F.softmax((fea_map / temperature).view(n, -1), dim=1).view(n, h, w)
        channel_map = value.mean(dim=(2, 3), keepdim=False)
        c_attn = c * F.softmax(channel_map / temperature, dim=1)
        return s_attn, c_attn

    @staticmethod
    def _gaussian_block(hmin, hmax, wmin, wmax, x_center, y_center, H, W, area, device):
        # 在bbox外扩一圈范围内生成二维高斯，并与area相乘
        d_h = (hmax - hmin) // 2
        start_h = max(0, hmin - d_h)
        end_h = min(hmax + d_h, H)
        d_w = (wmax - wmin) // 2
        start_w = max(0, wmin - d_w)
        end_w = min(wmax + d_w, W)
        if end_h <= start_h or end_w <= start_w:
            return None
        a = torch.linspace(start_h, end_h, end_h - start_h, device=device).reshape(-1, 1)
        b = torch.linspace(start_w, end_w, end_w - start_w, device=device).reshape(1, -1)
        # sigma使用中心归一化版本（与sfc实现一致的形状形式）
        d = (-0.5) * (((a - y_center) ** 2) / (y_center + 1e-8) ** 2 + ((b - x_center) ** 2) / (x_center + 1e-8) ** 2)
        return area * torch.exp(d)

    @staticmethod
    def _gaussian_from_mask(fg_bin: torch.Tensor) -> Tuple[torch.Tensor, float, float, float, float]:
        """
        从(H,W)的二值前景mask估计中心与尺度，并返回(μx, μy, σx, σy)。
        若前景为空，则返回居中且σ为H/4,W/4的默认值。
        """
        H, W = fg_bin.shape
        device = fg_bin.device
        inds = torch.nonzero(fg_bin > 0, as_tuple=False)
        if inds.numel() == 0:
            mu_y = (H - 1) / 2.0
            mu_x = (W - 1) / 2.0
            sig_y = max(1.0, H / 4.0)
            sig_x = max(1.0, W / 4.0)
            return fg_bin, mu_x, mu_y, sig_x, sig_y
        ys = inds[:, 0].float()
        xs = inds[:, 1].float()
        mu_y = ys.mean()
        mu_x = xs.mean()
        sig_y = torch.clamp(ys.std(unbiased=False), min=1.0)
        sig_x = torch.clamp(xs.std(unbiased=False), min=1.0)
        return fg_bin, float(mu_x.item()), float(mu_y.item()), float(sig_x.item()), float(sig_y.item())

    @staticmethod
    def _make_gaussian_grid(H: int, W: int, mu_x: float, mu_y: float, sig_x: float, sig_y: float, device: torch.device) -> torch.Tensor:
        yy = torch.arange(0, H, device=device).float().view(H, 1)
        xx = torch.arange(0, W, device=device).float().view(1, W)
        gx = (xx - mu_x) ** 2 / (sig_x + 1e-8) ** 2
        gy = (yy - mu_y) ** 2 / (sig_y + 1e-8) ** 2
        g = torch.exp(-0.5 * (gx + gy))
        s = g.sum()
        if s > 0:
            g = g / s
        return g

    @staticmethod
    def _build_gaussian_fg_bg(
        bboxes_xyxy: List[torch.Tensor], img_size: Tuple[int, int], feat_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hf, wf = feat_size
        hi, wi = img_size
        b = len(bboxes_xyxy)
        device = bboxes_xyxy[0].device if bboxes_xyxy and bboxes_xyxy[0].numel() > 0 else torch.device("cpu")
        fg = torch.zeros((b, hf, wf), dtype=torch.float32, device=device)
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
                area = 1.0 / max(1, (y2 + 1 - y1)) / max(1, (x2 + 1 - x1))
                # 高斯范围 + 取最大叠加
                gb = FSDLikeLoss._gaussian_block(y1, y2, x1, x2, x_center=(x1 + x2) / 2, y_center=(y1 + y2) / 2,
                                                 H=hf, W=wf, area=area, device=device)
                if gb is not None:
                    fg[i, max(0, y1 - (y2 - y1) // 2) : min(hf, y2 + (y2 - y1) // 2),
                          max(0, x1 - (x2 - x1) // 2) : min(wf, x2 + (x2 - x1) // 2)] = torch.maximum(
                        fg[i, max(0, y1 - (y2 - y1) // 2) : min(hf, y2 + (y2 - y1) // 2),
                              max(0, x1 - (x2 - x1) // 2) : min(wf, x2 + (x2 - x1) // 2)],
                        gb,
                    )
                # 矩形区域至少为area
                fg[i, y1 : y2 + 1, x1 : x2 + 1] = torch.maximum(
                    fg[i, y1 : y2 + 1, x1 : x2 + 1], torch.full((y2 - y1 + 1, x2 - x1 + 1), area, device=device)
                )
        bg = 1.0 - (fg > 0).float()
        # 归一化背景
        bg_sum = bg.sum(dim=(1, 2), keepdim=True)
        bg = torch.where(bg_sum > 0, bg / (bg_sum + 1e-6), bg)
        return fg, bg

    def _ensure_align_and_context(self, cs: int, ct: int, device: torch.device) -> None:
        if cs != ct and self.align_conv is None:
            self.align_conv = nn.Conv2d(cs, ct, kernel_size=1, bias=False).to(device)
        if self.channel_add_conv_s is None:
            self.channel_add_conv_s = nn.Sequential(
                nn.Conv2d(ct, ct // 2, kernel_size=1), nn.ReLU(inplace=True), nn.Conv2d(ct // 2, ct, kernel_size=1)
            ).to(device)
        if self.channel_add_conv_t is None:
            self.channel_add_conv_t = nn.Sequential(
                nn.Conv2d(ct, ct // 2, kernel_size=1), nn.ReLU(inplace=True), nn.Conv2d(ct // 2, ct, kernel_size=1)
            ).to(device)

    @staticmethod
    def _spatial_pool(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(2, 3), keepdim=True)

    def _get_rela_loss(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ct_s = self._spatial_pool(s)
        ct_t = self._spatial_pool(t)
        out_s = s + self.channel_add_conv_s(ct_s)
        out_t = t + self.channel_add_conv_t(ct_t)
        return F.mse_loss(out_s, out_t, reduction="sum") / max(1, out_s.numel() // out_s.shape[1])

    def forward(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        bboxes_xyxy: List[torch.Tensor] | None = None,
        img_size: Tuple[int, int] | None = None,
        fg_map: torch.Tensor | None = None,
        bg_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 对齐到教师尺度
        if student_feat.shape[2:] != teacher_feat.shape[2:]:
            student_feat = F.interpolate(student_feat, size=teacher_feat.shape[2:], mode="bilinear", align_corners=False)
        cs, ct = student_feat.size(1), teacher_feat.size(1)
        self._ensure_align_and_context(cs, ct, student_feat.device)
        if cs != ct:
            student_feat = self.align_conv(student_feat)

        b, c, h, w = teacher_feat.shape
        # 前景/背景权重：优先使用外部fg/bg（来自JSON分割）；否则使用GT框生成高斯
        if fg_map is None or bg_map is None:
            assert bboxes_xyxy is not None and img_size is not None, "FSD需要提供bboxes+img_size或fg/bg"
            fg, bg = self._build_gaussian_fg_bg(bboxes_xyxy, img_size=img_size, feat_size=(h, w))
            fg_map = fg.unsqueeze(1)
            bg_map = bg.unsqueeze(1)
        else:
            # 可选：基于掩码生成高斯并与掩码融合
            if self.gaussian_from_mask:
                # 期望fg_map为(B,1,H,W)的{0,1}或[0,1]，先二值化
                fg_bin = (fg_map.squeeze(1) > 0.5).float()
                new_fg_list = []
                for i in range(b):
                    bin_i, mu_x, mu_y, sig_x, sig_y = self._gaussian_from_mask(fg_bin[i])
                    g = self._make_gaussian_grid(h, w, mu_x, mu_y, sig_x, sig_y, device=fg_bin.device)
                    # 面积常数: 1/NumFgPixels（若0则退化为均匀极小值）
                    num_fg = torch.clamp(bin_i.sum(), min=1.0)
                    A = 1.0 / num_fg
                    gA = A * g
                    area_map = (bin_i > 0).float() / num_fg
                    if self.gaussian_mix == "blend":
                        lam = float(self.gaussian_blend_lambda)
                        fused = lam * area_map + (1.0 - lam) * gA
                    else:
                        fused = torch.maximum(area_map, gA)
                    new_fg_list.append(fused)
                fg_map = torch.stack(new_fg_list, dim=0).unsqueeze(1)
                # 背景随之更新
                bg = 1.0 - (fg_map.squeeze(1) > 0).float()
                s = bg.view(b, -1).sum(dim=1, keepdim=True).clamp(min=1.0)
                bg = (bg.view(b, -1) / s).view(b, h, w)
                bg_map = bg.unsqueeze(1)

        # S/C注意力（与FGD一致）
        s_attn_t, c_attn_t = self._get_attention(teacher_feat, self.temperature)
        c_attn_t_ = c_attn_t.unsqueeze(-1).unsqueeze(-1)
        s_attn_t_ = s_attn_t.unsqueeze(1)
        fea_t = teacher_feat * torch.sqrt(s_attn_t_) * torch.sqrt(c_attn_t_)
        fea_s = student_feat * torch.sqrt(s_attn_t_) * torch.sqrt(c_attn_t_)

        # 前景/背景特征
        fg_t = fea_t * torch.sqrt(torch.clamp(fg_map, min=0))
        fg_s = fea_s * torch.sqrt(torch.clamp(fg_map, min=0))
        bg_t = fea_t * torch.sqrt(torch.clamp(bg_map, min=0))
        bg_s = fea_s * torch.sqrt(torch.clamp(bg_map, min=0))

        # 归一化+softmax -> KL（分布对齐）
        fg_t = self._feature_norm(fg_t)
        fg_s = self._feature_norm(fg_s)
        bg_t = self._feature_norm(bg_t)
        bg_s = self._feature_norm(bg_s)

        fg_t_sm = self._softmax_spatial(fg_t, tau=1.0)
        fg_s_sm = self._softmax_spatial(fg_s, tau=1.0)
        bg_t_sm = self._softmax_spatial(bg_t, tau=1.0)
        bg_s_sm = self._softmax_spatial(bg_s, tau=1.0)

        eps = 1e-5
        loss_fg = torch.sum(-fg_t_sm * torch.log(eps + fg_s_sm) + fg_t_sm * torch.log(eps + fg_t_sm)) / max(1, b * c)
        loss_bg = torch.sum(-bg_t_sm * torch.log(eps + bg_s_sm) + bg_t_sm * torch.log(eps + bg_t_sm)) / max(1, b * c)

        loss = self.weight_fg * loss_fg + self.weight_bg * loss_bg

        # 可选mask/关系正则（默认0，不引入外部权重来源）
        if self.gamma_mask > 0:
            s_attn_s, c_attn_s = self._get_attention(student_feat, self.temperature)
            mask_loss = (torch.abs(c_attn_s - c_attn_t).sum() / max(1, c_attn_s.shape[0])) + (
                torch.abs(s_attn_s - s_attn_t).sum() / max(1, s_attn_s.shape[0] * s_attn_s.shape[1] * s_attn_s.shape[2])
            )
            loss = loss + self.gamma_mask * mask_loss
        if self.lambda_rela > 0:
            rela_loss = self._get_rela_loss(student_feat, teacher_feat)
            loss = loss + self.lambda_rela * rela_loss

        return loss


