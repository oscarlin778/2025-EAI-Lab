import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# Bottleneck Block
# ----------------------
# ----------------------
# Bottleneck Block
# ----------------------
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, planes, out_channels, downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        assert len(out_channels) == 3, "Bottleneck requires out_channels list of length 3"
        
        ################################################
        #           以下是填空完成的部分               #            
        ################################################
        self.conv1 = nn.Conv2d(in_channels, out_channels[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels[0])

        self.conv2 = nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels[1])

        self.conv3 = nn.Conv2d(out_channels[1], out_channels[2], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels[2])
        ################################################
        #           以上是填空完成的部分               #            
        ################################################

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out

# ----------------------
# ResNet
# ----------------------
class ResNet(nn.Module):
    def __init__(self, block, layers, cfg, num_classes=1000, in_channels=3):
        super(ResNet, self).__init__()
        self.current_cfg_idx = 0

        # Conv1
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.current_cfg_idx += 1
        self.inplanes = 64

        # Layer1~Layer4
        self.layer1 = self._make_layer(block, 64, layers[0], cfg)
        self.layer2 = self._make_layer(block, 128, layers[1], cfg, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], cfg, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], cfg, stride=2)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.inplanes, num_classes)

   
    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        #############################################################################
        #                       以下是填空完成的部分                              #
        #############################################################################
        downsample = None
        # 檢查是否需要 downsample (Projection Shortcut)
        # 條件: stride 不是 1, 或是輸入維度 (self.inplanes) 和 期望的輸出維度 (planes * block.expansion) 不符
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Conv2d(self.inplanes, planes * block.expansion,
                                    kernel_size=1, stride=stride, bias=False)
        
        layers = []
        
        # 1. 處理 Layer 的第一個 Block
        # 從 cfg 獲取這個 block 對應的 3 個通道數 (要做成 list 才能修改)
        out_ch = list(cfg[self.current_cfg_idx : self.current_cfg_idx + 3])
        self.current_cfg_idx += 3 # 移動索引
        
        # 強制固定 bottleneck 的輸出通道數
        out_ch[2] = planes * block.expansion 
        
        layers.append(block(self.inplanes, planes, out_ch, downsample, stride))
        
        # 更新 self.inplanes 供這個 layer 的 "下一個" block 使用
        self.inplanes = planes * block.expansion 

        # 2. 處理剩下的 Blocks
        for _ in range(1, blocks):
            # 從 cfg 獲取 3 個通道數
            out_ch = list(cfg[self.current_cfg_idx : self.current_cfg_idx + 3])
            self.current_cfg_idx += 3
            
            # 強制固定 bottleneck 的輸出通道數
            out_ch[2] = planes * block.expansion
            
            layers.append(block(self.inplanes, planes, out_ch))

        return nn.Sequential(*layers)
        #############################################################################
        #                       以上是填空完成的部分                              #
        #############################################################################

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNet50(num_classes=10, in_channels=3, cfg=None):
    # cfg: number of channels for conv1 and the three conv layers in each Bottleneck
    if cfg is None:
        cfg = [64] + \
              [64, 64, 256]*3 + \
              [128, 128, 512]*4 + \
              [256, 256, 1024]*6 + \
              [512, 512, 2048]*3
    layers = [3, 4, 6, 3]
    return ResNet(Bottleneck, layers, cfg, num_classes=num_classes, in_channels=in_channels)

def ResNet101(num_classes=10, in_channels=3, cfg=None):
    # cfg: number of channels for conv1 and the three conv layers in each Bottleneck
    if cfg is None:
        cfg = [64] + \
              [64, 64, 256]*3 + \
              [128, 128, 512]*4 + \
              [256, 256, 1024]*23 + \
              [512, 512, 2048]*3
    layers = [3, 4, 23, 3]
    return ResNet(Bottleneck, layers, cfg, num_classes=num_classes, in_channels=in_channels)

def ResNet152(num_classes=10, in_channels=3, cfg=None):
    # cfg: number of channels for conv1 and the three conv layers in each Bottleneck
    if cfg is None:
        cfg = [64] + \
              [64, 64, 256]*3 + \
              [128, 128, 512]*8 + \
              [256, 256, 1024]*36 + \
              [512, 512, 2048]*3
    layers = [3, 8, 36, 3]
    return ResNet(Bottleneck, layers, cfg, num_classes=num_classes, in_channels=in_channels)