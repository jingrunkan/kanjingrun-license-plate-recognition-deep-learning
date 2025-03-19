import torch
import torch.nn as nn
from CMBA import CBAM
from Pconv import Partial_conv3
from Triplet import TripletAttention
import torchvision.models as models

class BasicBlockWithCBAM(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlockWithCBAM, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.cbam = CBAM(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.cbam(out)  # Apply CBAM after BN
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
class ModifiedResNet18(nn.Module):
    def __init__(self,n_div, num_classes,forward='split_cat'):
        super(ModifiedResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=True) 
        self.in_planes = 64
        self.resnet.layer1 = self._make_layer(BasicBlockWithCBAM, 64, self.resnet.layer1)
        self.resnet.layer2 = self._make_layer(BasicBlockWithCBAM, 128, self.resnet.layer2, stride=2)
        self.resnet.layer3 = self._make_layer(BasicBlockWithCBAM, 256, self.resnet.layer3, stride=2)
        self.resnet.layer4 = self._make_layer(BasicBlockWithCBAM, 512, self.resnet.layer4, stride=2)
        self.resnet.fc = nn.Linear(512 * BasicBlockWithCBAM.expansion, num_classes)
        i = 1
        for block in self.resnet.layer1:
            if i == 1:
                block.conv1 = Partial_conv3(64, n_div, forward)
                block.conv2 = Partial_conv3(64, n_div, forward)
            else:
                block.conv1 = Partial_conv3(64, n_div, forward)
                block.conv2 = Partial_conv3(64, n_div, forward)
            i+=1
        i = 1
        for block in self.resnet.layer2:
            if i == 1:
                block.conv2 = Partial_conv3(128, n_div, forward)
                i += 1
                continue
            block.conv1 = Partial_conv3(128, n_div, forward)
            block.conv2 = Partial_conv3(128, n_div, forward)
        i = 1
        for block in self.resnet.layer3:
            if i == 1:
                block.conv2 = Partial_conv3(256, n_div, forward)
                i += 1
                continue
            block.conv1 = Partial_conv3(256, n_div, forward)
            block.conv2 = Partial_conv3(256, n_div, forward)

        i = 1
        for block in self.resnet.layer4:
            if i == 1:
                block.conv2 = Partial_conv3(512, n_div, forward)
                i += 1
                continue
            block.conv1 = Partial_conv3(512, n_div, forward)
            block.conv2 = Partial_conv3(512, n_div, forward)
    def _make_layer(self, block, planes, layer, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(len(layer) - 1):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.resnet(x)
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    n_div = 2  
    model = ModifiedResNet18(n_div, 4).to(device)
    custom_module = TripletAttention()
    model.resnet.conv1 = nn.Sequential(
        model.resnet.conv1,
        custom_module
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    #optimizer = Adam(model.parameters(), lr=0.001)