import torch

# 这段代码定义了一个用于创建三角形因果掩码的类
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        # 计算掩码的形状，其中 B 表示批量大小，L 表示序列长度
        mask_shape = [B, 1, L, L]
        # 在这个上下文中，禁用梯度计算
        with torch.no_grad():
            # 创建一个上三角矩阵，并将其转换为布尔类型。torch.triu 函数用于获取输入矩阵的上三角部分，diagonal 参数指定了主对角线的偏移量。
            # 最后，将其移动到指定的设备上
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask
