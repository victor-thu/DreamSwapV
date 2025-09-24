import math
import torch
import torch.nn as nn
from utils.sam.aot.utils.learning import freeze_params


# ------------------------------- 内部小组件 -------------------------------

class ConvBNAct(nn.Sequential):
    """Conv2d -> BatchNorm -> ReLU 的小组件（可选激活）"""
    def __init__(self, in_c, out_c, k=3, s=1, p=0, d=1, use_act=True, BatchNorm=None, bias=False):
        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, dilation=d, bias=bias),
            BatchNorm(out_c)
        ]
        if use_act:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


def _os_config(output_stride: int):
    """根据 output_stride 返回每个 stage 的 stride / dilation 配置。"""
    if output_stride == 16:
        # 与原实现保持一致
        strides   = [1, 2, 2, 1]
        dilations = [1, 1, 1, 2]
    elif output_stride == 8:
        strides   = [1, 2, 1, 1]
        dilations = [1, 1, 2, 4]
    else:
        raise NotImplementedError(f"Unsupported output_stride: {output_stride}")
    return strides, dilations


# ------------------------------- 主体模块 -------------------------------

class Bottleneck(nn.Module):
    """ResNet Bottleneck（与原始接口保持一致）"""
    expansion = 4

    def __init__(self,
                 inplanes: int,
                 planes:   int,
                 stride:   int = 1,
                 dilation: int = 1,
                 downsample: nn.Module | None = None,
                 BatchNorm=None):
        super().__init__()
        # conv1: 1x1
        self.conv1 = ConvBNAct(inplanes, planes, k=1, s=1, p=0, d=1, use_act=True, BatchNorm=BatchNorm)
        # conv2: 3x3（可带 stride / dilation）
        self.conv2 = ConvBNAct(planes, planes, k=3, s=stride, p=dilation, d=dilation, use_act=True, BatchNorm=BatchNorm)
        # conv3: 1x1
        self.conv3 = ConvBNAct(planes, planes * self.expansion, k=1, s=1, p=0, d=1, use_act=False, BatchNorm=BatchNorm)

        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride     = stride
        self.dilation   = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    与原类保持一致的外部行为：
    - 构造签名相同
    - forward 返回 4 个尺度的特征列表 xs（最后两项同为 16x 下采样）
    - 冻结策略 freeze(freeze_at) 一致
    """
    def __init__(self, block, layers, output_stride, BatchNorm, freeze_at=0):
        super().__init__()
        self.inplanes = 64

        strides, dilations = _os_config(output_stride)

        # stem
        self.conv1   = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1     = BatchNorm(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # stages
        self.layer1 = self._make_layer(block,  64, layers[0],
                                       stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1],
                                       stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2],
                                       stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
        # 与原实现一致：不使用 layer4（保留 16x）
        # self.layer4 = self._make_layer(block, 512, layers[3],
        #                                stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)

        # 记录 stem/stages 以供 freeze
        self.stem   = [self.conv1, self.bn1]
        self.stages = [self.layer1, self.layer2, self.layer3]

        self._init_weight()
        self.freeze(freeze_at)

    # ---- 内部构建 ----
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None) -> nn.Sequential:
        """
        与原方法等价：当 stride 改变或通道数不匹配时，使用 1x1 downsample。
        第一块的 dilation 使用 max(dilation // 2, 1)（保持与原实现一致）。
        """
        downsample = None
        out_c = planes * block.expansion
        if stride != 1 or self.inplanes != out_c:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_c, kernel_size=1, stride=stride, bias=False),
                BatchNorm(out_c),
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                            stride=stride,
                            dilation=max(dilation // 2, 1),
                            downsample=downsample,
                            BatchNorm=BatchNorm))
        self.inplanes = out_c
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                stride=1,
                                dilation=dilation,
                                downsample=None,
                                BatchNorm=BatchNorm))
        return nn.Sequential(*layers)

    # ---- forward ----
    def forward(self, x: torch.Tensor):
        # stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        xs = []
        # 4×
        x = self.layer1(x); xs.append(x)
        # 8×
        x = self.layer2(x); xs.append(x)
        # 16×
        x = self.layer3(x); xs.append(x)
        # 与原实现一致：重复一份 16×（占位，兼容外部依赖）
        xs.append(x)

        return xs

    # ---- 初始化 & 冻结 ----
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def freeze(self, freeze_at: int):
        """
        freeze_at 语义保持一致：
        - >=1 冻结 stem
        - >=2 依次冻结 layer1
        - >=3 冻结 layer2
        - >=4 冻结 layer3
        """
        if freeze_at >= 1:
            for m in self.stem:
                freeze_params(m)
        for idx, stage in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                freeze_params(stage)


# ------------------------------- 构造函数 -------------------------------

def ResNet50(output_stride, BatchNorm, freeze_at=0):
    """构建 ResNet-50（与原函数名/签名一致）"""
    return ResNet(Bottleneck, [3, 4, 6, 3], output_stride, BatchNorm, freeze_at=freeze_at)


def ResNet101(output_stride, BatchNorm, freeze_at=0):
    """构建 ResNet-101（与原函数名/签名一致）"""
    return ResNet(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm, freeze_at=freeze_at)


# ------------------------------- 自测（不影响外部） -------------------------------
if __name__ == "__main__":
    model = ResNet101(BatchNorm=nn.BatchNorm2d, output_stride=8)
    x = torch.rand(1, 3, 512, 512)
    feats = model(x)  # 与原实现一致，返回 list[Tensor]，长度 4
    print([f.shape for f in feats])
