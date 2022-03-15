import torch.nn as nn
import warnings
from collections import namedtuple
from typing import Optional, Tuple, List, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class Network(nn.Module):
    def __init__(self, in_channels=3, num_classes=21, init_weights: Optional[bool] = False, transform_input: bool = False, aux_logits: bool = True, dropout: float = 0.2, dropout_aux: float = 0.7, blocks: Optional[List[Callable[..., nn.Module]]] = None) -> None:
        super(Network, self).__init__()
        if blocks == None:
            blocks = [BasicConv2d, Inception_A, Reduction_A, Inception_B, Reduction_B, Inception_C, InceptionAux]
        assert len(blocks) == 7
        # define the network layers
        conv_block = blocks[0]
        inception_A = blocks[1]
        reduction_A = blocks[2]
        inception_B = blocks[3]
        reduction_B = blocks[4]
        inception_C = blocks[5]
        inception_aux_block = blocks[6]

        self.aux_logits = aux_logits
        self.transform_input = transform_input
        
        self.Conv2d_1a_3x3 = conv_block(3, 32, kernel_size=3, stride=2, padding=1)
        self.Conv2d_2a_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.Conv2d_2b_3x3 = conv_block(64, 80, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.Conv2d_3b_1x1 = conv_block(80, 192, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(192, 384, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        self.inception3a = inception_A(384)
        self.inception3b = inception_A(384)
        self.inception3c = inception_A(384)
        self.inception3d = inception_A(384)
        self.reduction3 = reduction_A(384, k=192, l=224, m=256, n=384)
        
        self.inception4a = inception_B(1024)
        self.inception4b = inception_B(1024)
        self.inception4c = inception_B(1024)
        self.inception4d = inception_B(1024)
        self.inception4e = inception_B(1024)
        self.inception4f = inception_B(1024)
        self.inception4g = inception_B(1024)
        self.reduction4 = reduction_B(1024)
        
        self.inception5a = inception_C(1536)
        self.inception5b = inception_C(1536)
        self.inception5c = inception_C(1536)
        
        if aux_logits:
            self.aux1 = inception_aux_block(1024, num_classes, dropout=dropout_aux)
            self.aux2 = inception_aux_block(1536, num_classes, dropout=dropout_aux)
        else:
            self.aux1 = None  # type: ignore[assignment]
            self.aux2 = None  # type: ignore[assignment]
            
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1536, num_classes)
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        # define the network structure
         # N x 3 x 224 x 224
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 112 x 112
        x = self.Conv2d_2a_3x3(x)
        # N x 64 x 112 x 112
        x = self.Conv2d_2b_3x3(x)
        # N x 80 x 112 x 112
        x = self.maxpool1(x)
        # N x 80 x 55 x 55
        x = self.Conv2d_3b_1x1(x)
        # N x 192 x 55 x 55
        x = self.Conv2d_4a_3x3(x)
        # N x 384 x 55 x 55
        x = self.maxpool2(x)
        
        # inception block A
        # N x 384 x 27 x 27
        x = self.inception3a(x)
        # N x 384 x 27 x 27
        x = self.inception3b(x)
        # N x 384 x 27 x 27
        x = self.inception3c(x)
        # N x 384 x 27 x 27
        x = self.inception3d(x)
        # N x 384 x 27 x 27
        x = self.reduction3(x)
        # N x 1024 x 13 x 13
        aux1: Optional[Tensor] = None
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(x) # aux classifier 1
        
        # inception block B
        x = self.inception4a(x)
        # N x 1024 x 13 x 13
        x = self.inception4b(x)
        # N x 1024 x 13 x 13
        x = self.inception4c(x)
        # N x 1024 x 13 x 13
        x = self.inception4d(x)
        # N x 1024 x 13 x 13
        x = self.reduction4(x)
        # N x 1536 x 6 x 6
        aux2: Optional[Tensor] = None
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(x)# aux classifier 2

        # inception block C
        x = self.inception5a(x)
        # N x 1536 x 6 x 6
        x = self.inception5b(x)
        # N x 1536 x 6 x 6
        x = self.inception5c(x)
        # N x 1536 x 6 x 6

        # Output
        x = self.avgpool(x)
        # N x 1536 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1536
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux2, aux1
        
       
    

class Inception_A(nn.Module):
    def __init__(self, in_channels):
        super(Inception_A, self).__init__()
        self.branch_0 = Conv2d(in_channels, 96, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, 64, 1, stride=1, padding=0, bias=False),
            Conv2d(64, 96, 3, stride=1, padding=1, bias=False),
        )
        self.branch_2 = nn.Sequential(
            Conv2d(in_channels, 64, 1, stride=1, padding=0, bias=False),
            Conv2d(64, 96, 3, stride=1, padding=1, bias=False),
            Conv2d(96, 96, 3, stride=1, padding=1, bias=False),
        )
        self.brance_3 = nn.Sequential(
            nn.AvgPool2d(3, 1, padding=1, count_include_pad=False),
            Conv2d(384, 96, 1, stride=1, padding=0, bias=False)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.brance_3(x)
        return self.relu(torch.cat((x0, x1, x2, x3), dim=1) + x)
    
class Reduction_A(nn.Module):
    # 35 -> 17
    def __init__(self, in_channels, k, l, m, n):
        super(Reduction_A, self).__init__()
        self.branch_0 = Conv2d(in_channels, n, 3, stride=2, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, k, 1, stride=1, padding=0, bias=False),
            Conv2d(k, l, 3, stride=1, padding=1, bias=False),
            Conv2d(l, m, 3, stride=2, padding=0, bias=False),
        )
        self.branch_2 = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        return torch.cat((x0, x1, x2), dim=1) # 17 x 17 x 1024
    
class Inception_B(nn.Module):
    def __init__(self, in_channels):
        super(Inception_B, self).__init__()
        self.branch_0 = Conv2d(in_channels, 384, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, 192, 1, stride=1, padding=0, bias=False),
            Conv2d(192, 224, (1, 7), stride=1, padding=(0, 3), bias=False),
            Conv2d(224, 256, (7, 1), stride=1, padding=(3, 0), bias=False),
        )
        self.branch_2 = nn.Sequential(
            Conv2d(in_channels, 192, 1, stride=1, padding=0, bias=False),
            Conv2d(192, 192, (7, 1), stride=1, padding=(3, 0), bias=False),
            Conv2d(192, 224, (1, 7), stride=1, padding=(0, 3), bias=False),
            Conv2d(224, 224, (7, 1), stride=1, padding=(3, 0), bias=False),
            Conv2d(224, 256, (1, 7), stride=1, padding=(0, 3), bias=False)
        )
        self.branch_3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            Conv2d(in_channels, 128, 1, stride=1, padding=0, bias=False)
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(1024, 1024, 1, 1, 0, bias=False)
    def forward(self, x):
        x4 = self.conv(x)
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return self.relu(torch.cat((x0, x1, x2, x3), dim=1) + x4)
    
class Reduction_B(nn.Module):
    # 17 -> 8
    def __init__(self, in_channels):
        super(Reduction_B, self).__init__()
        self.branch_0 = nn.Sequential(
            Conv2d(in_channels, 192, 1, stride=1, padding=0, bias=False),
            Conv2d(192, 192, 3, stride=2, padding=0, bias=False),
        )
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, 256, 1, stride=1, padding=0, bias=False),
            Conv2d(256, 256, (1, 7), stride=1, padding=(0, 3), bias=False),
            Conv2d(256, 320, (7, 1), stride=1, padding=(3, 0), bias=False),
            Conv2d(320, 320, 3, stride=2, padding=0, bias=False)
        )
        self.branch_2 = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        return torch.cat((x0, x1, x2), dim=1)  # 8 x 8 x 1536
    
class Inception_C(nn.Module):
    def __init__(self, in_channels):
        super(Inception_C, self).__init__()
        self.branch_0 = Conv2d(in_channels, 256, 1, stride=1, padding=0, bias=False)

        self.branch_1 = Conv2d(in_channels, 384, 1, stride=1, padding=0, bias=False)
        self.branch_1_1 = Conv2d(384, 256, (1, 3), stride=1, padding=(0, 1), bias=False)
        self.branch_1_2 = Conv2d(384, 256, (3, 1), stride=1, padding=(1, 0), bias=False)

        self.branch_2 = nn.Sequential(
            Conv2d(in_channels, 384, 1, stride=1, padding=0, bias=False),
            Conv2d(384, 448, (3, 1), stride=1, padding=(1, 0), bias=False),
            Conv2d(448, 512, (1, 3), stride=1, padding=(0, 1), bias=False),
        )
        self.branch_2_1 = Conv2d(512, 256, (1, 3), stride=1, padding=(0, 1), bias=False)
        self.branch_2_2 = Conv2d(512, 256, (3, 1), stride=1, padding=(1, 0), bias=False)

        self.branch_3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            Conv2d(in_channels, 256, 1, stride=1, padding=0, bias=False)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x1_1 = self.branch_1_1(x1)
        x1_2 = self.branch_1_2(x1)
        x1 = torch.cat((x1_1, x1_2), 1)
        x2 = self.branch_2(x)
        x2_1 = self.branch_2_1(x2)
        x2_2 = self.branch_2_2(x2)
        x2 = torch.cat((x2_1, x2_2), dim=1)
        x3 = self.branch_3(x)
        return self.relu(torch.cat((x0, x1, x2, x3), dim=1) + x) # 8 x 8 x 1536
    
class InceptionAux(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_block: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.7,
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 2048, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        # aux1: N x 1024 x 13 x 13, aux2: N x 1536 x 6 x 6
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # aux1: N x 1024 x 1 x 1, aux2: N x 1536 x 1 x 1
        x = self.conv(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = self.dropout(x)
        # N x 1024
        x = self.fc2(x)
        # N x 21 (num_classes)

        return x
    
class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x