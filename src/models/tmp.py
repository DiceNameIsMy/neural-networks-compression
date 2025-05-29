import torch.nn.functional as F
from torch import Tensor, nn

from src.models.compression import binary, binary_ReSTE
from src.models.compression.ternarize import TernaryConv2d


class BCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.infl_ratio = 3
        self.fc1 = binary.BinarizeLinear(784, 2048 * self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(2048 * self.infl_ratio)
        self.bin2 = binary.Module_Binarize()
        self.fc2 = binary.BinarizeLinear(2048 * self.infl_ratio, 2048 * self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(2048 * self.infl_ratio)
        self.bin3 = binary.Module_Binarize()
        self.fc3 = binary.BinarizeLinear(2048 * self.infl_ratio, 2048 * self.infl_ratio)
        self.htanh3 = nn.Hardtanh()
        self.bn3 = nn.BatchNorm1d(2048 * self.infl_ratio)
        self.bin4 = binary.Module_Binarize()
        self.fc4 = nn.Linear(2048 * self.infl_ratio, 10)
        self.logsoftmax = nn.LogSoftmax()
        self.drop = nn.Dropout(0.5)

    def forward(self, x: Tensor):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.bin2(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.bin3(x)
        x = self.fc3(x)
        x = self.drop(x)
        x = self.bn3(x)
        x = self.htanh3(x)
        x = self.bin4(x)
        x = self.fc4(x)
        return self.logsoftmax(x)


class BCNN_ReSTE(nn.Module):
    def __init__(self):
        super().__init__()
        self.infl_ratio = 3
        self.fc1 = binary.BinarizeLinear(784, 2048 * self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(2048 * self.infl_ratio)
        self.bin2 = binary_ReSTE.Module_Binarize_ReSTE(threshold=1.5, o=3)
        self.fc2 = binary.BinarizeLinear(2048 * self.infl_ratio, 2048 * self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(2048 * self.infl_ratio)
        self.bin3 = binary_ReSTE.Module_Binarize_ReSTE(threshold=1.5, o=3)
        self.fc3 = binary.BinarizeLinear(2048 * self.infl_ratio, 2048 * self.infl_ratio)
        self.htanh3 = nn.Hardtanh()
        self.bn3 = nn.BatchNorm1d(2048 * self.infl_ratio)
        self.bin4 = binary_ReSTE.Module_Binarize_ReSTE(threshold=1.5, o=3)
        self.fc4 = nn.Linear(2048 * self.infl_ratio, 10)
        self.logsoftmax = nn.LogSoftmax()
        self.drop = nn.Dropout(0.5)

    def forward(self, x: Tensor):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.bin2(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.bin3(x)
        x = self.fc3(x)
        x = self.drop(x)
        x = self.bn3(x)
        x = self.htanh3(x)
        x = self.bin4(x)
        x = self.fc4(x)
        return self.logsoftmax(x)


class TCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = TernaryConv2d(1, 32, kernel_size=5)
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.conv2 = TernaryConv2d(32, 64, kernel_size=5)
        self.bn_conv2 = nn.BatchNorm2d(64)
        self.fc1 = TernaryConv2d(1024, 512, kernel_size=1)
        self.bn_fc1 = nn.BatchNorm2d(512)
        self.fc2 = nn.Conv2d(512, 10, kernel_size=1, bias=False)
        # self.bn_fc2 = nn.BatchNorm2d(10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(self.bn_conv1(x)), 2)
        x = self.conv2(x)
        x = F.max_pool2d(F.relu(self.bn_conv2(x)), 2)

        x = x.view(-1, 1024, 1, 1)
        x = F.relu(self.bn_fc1(self.fc1(x)))

        x = self.fc2(x)
        x = x.view(-1, 10)

        return x
