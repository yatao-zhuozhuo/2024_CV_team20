import math

import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        #-----------------------------------#
        #   假设输入进来的图片是600,600,3
        #-----------------------------------#
        self.inplanes = 64
        super(ResNet, self).__init__()

        # 600,600,3 -> 300,300,64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 300,300,64 -> 150,150,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # 150,150,64 -> 150,150,256
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 150,150,256 -> 75,75,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 75,75,512 -> 38,38,1024 到这里可以获得一个38,38,1024的共享特征层
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4被用在classifier模型中
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        #-------------------------------------------------------------------#
        #   当模型需要进行高和宽的压缩的时候，就需要用到残差边的downsample
        #-------------------------------------------------------------------#
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNet_dual(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet_dual, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def sobel_process(self, x):
        # 将输入图像转换为灰度图（假设输入是RGB图像）
        # 如果输入是3通道（RGB），我们先做灰度化处理
        if x.size(1) == 3:
            x_gray = 0.2989 * x[:, 0, :, :] + 0.5870 * x[:, 1, :, :] + 0.1140 * x[:, 2, :, :]
            x_gray = x_gray.unsqueeze(1)  # 将灰度图转为单通道形式
        else:
            x_gray = x  # 如果输入已经是单通道图像

        # Sobel算子，用于提取边缘
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()  # 水平梯度
        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()  # 垂直梯度

        sobel_x = sobel_x.expand(x.size(1), 1, 3, 3)
        sobel_y = sobel_y.expand(x.size(1), 1, 3, 3)

        # 使用卷积进行边缘检测
        grad_x = F.conv2d(x_gray, sobel_x, padding=1)
        grad_y = F.conv2d(x_gray, sobel_y, padding=1)

        # 计算梯度的强度（边缘图）
        edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2)

        # 确保输出尺寸与原图一致
        edge_magnitude = torch.clamp(edge_magnitude, 0, 1)

        return edge_magnitude  # 返回边缘图

    def forward(self, x):
        # 在前传的开头使用Sobel边缘处理
        x = self.sobel_process(x)

        # 继续执行后续的ResNet层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    # 这段代码没有使用
    def fft_process(self, x):
        # 获取图像的FFT变换
        x_fft = torch.fft.fftn(x)

        # 移动零频到中心
        x_fft = torch.fft.fftshift(x_fft)
    
        # 获取图像的高度和宽度
        height, width = x.shape[1], x.shape[2]
    
        # 确定频谱中心区域的大小（可以根据需求调整）
        center_width = width // 8
        center_height = height // 8
    
        # 将频谱中心的低频区域置为 0
        #x_fft[:, height//2-center_height//2:height//2+center_height//2,
        #     width//2-center_width//2:width//2+center_width//2] = 0.0
        
        x_fft[:, 0:height//2-center_height//2, :] = 0.0   # 去除上半部分的高频信息
        x_fft[:, height//2+center_height//2:, :] = 0.0   # 去除下半部分的高频信息
        x_fft[:, :, 0:width//2-center_width//2] = 0.0    # 去除左半部分的高频信息
        x_fft[:, :, width//2+center_width//2:] = 0.0    # 去除右半部分的高频信息


        x_fft = torch.fft.ifftshift(x_fft, dim=[1, 2])
        x_fft = torch.fft.ifftn(x_fft, dim=[1, 2]).real

        return x_fft  # 返回实部作为处理后的图像

'''
def resnet50_dual(pretrained=False, fc_weight=0.5):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    model_dual = ResNet_dual(Bottleneck, [3, 4, 6, 3])

    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-19c8e357.pth", model_dir="./model_data")
        model.load_state_dict(state_dict)

        state_dict2 = torch.load("/root/rag/CV/faster-rcnn-pytorch/model_data/voc_weights_resnet.pth")  # 请检查路径的正确性
        model_dual.load_state_dict(state_dict2)

    # 获取 oral 特征提取部分
    features_oral = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    # 获取 dual 特征提取部分
    features_dual = list([model_dual.conv1, model_dual.bn1, model_dual.relu, model_dual.maxpool, model_dual.layer1, model_dual.layer2, model_dual.layer3])

    # 获取分类部分
    classifier = list([model.layer4, model.avgpool])

    class CombinedFeatures(nn.Module):
        def __init__(self, features_oral, features_dual, fc_weight):
            super(CombinedFeatures, self).__init__()
            self.features_oral = nn.ModuleList(features_oral)
            self.features_dual = nn.ModuleList(features_dual)
            self.fc_weight = fc_weight

        def forward(self, x):
            x_oral = x
            x_dual = x
            features_combined = []
            for oral_layer, dual_layer in zip(self.features_oral, self.features_dual):
                x_oral = oral_layer(x_oral)
                x_dual = dual_layer(x_dual)

                if isinstance(oral_layer, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.Sequential)):
                    # 加权融合特征图
                    combined_feature = self.fc_weight * x_oral + (1 - self.fc_weight) * x_dual
                    features_combined.append(combined_feature)
                    x_oral = combined_feature.clone()  # 使用 .clone() 避免原地操作
                    x_dual = combined_feature.clone()  # 使用 .clone() 避免原地操作
                else:
                    # 对于残差块等复杂结构，仅在该层前传时进行融合
                    combined_feature = x_oral  # 不改变，仍然传递 oral 层的输出
                    features_combined.append(combined_feature)
                    # 在此不再更新 x_oral 和 x_dual，保持残差结构的完整性

            return x_oral  # 返回融合后的特征

    combined_model = CombinedFeatures(features_oral, features_dual, fc_weight)
    classifier = nn.Sequential(*classifier)

    return combined_model, classifier
'''


def resnet50_dual(pretrained=False, fc_weight=0.5):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    model_dual = ResNet_dual(Bottleneck, [3, 4, 6, 3])
    
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-19c8e357.pth", model_dir="./model_data")
        model.load_state_dict(state_dict)
        
        # 加载模型权重
        state_dict2 = torch.load("/root/rag/CV/faster-rcnn-pytorch/model_data/voc_weights_resnet.pth")

        # 加载到你的模型
        model_dual.load_state_dict(state_dict2)
    
    # 获取 oral 特征提取部分
    features_oral = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    # 获取 dual 特征提取部分
    features_dual = list([model_dual.conv1, model_dual.bn1, model_dual.relu, model_dual.maxpool, model_dual.layer1, model_dual.layer2, model_dual.layer3])

    # 获取分类部分
    classifier = list([model.layer4, model.avgpool])
    
    # 定义一个新的特征提取层，用于合并处理
    class CombinedFeatures(nn.Module):
        def __init__(self, features_oral, features_dual, fc_weight):
            super(CombinedFeatures, self).__init__()
            self.features_oral = nn.Sequential(*features_oral)
            self.features_dual = nn.Sequential(*features_dual)
            self.fc_weight = fc_weight

        def forward(self, x):
            # 获取两个不同模型的输出
            oral_output = self.features_oral(x)
            dual_output = self.features_dual(x)

            # 加权融合两者的特征输出
            return self.fc_weight * oral_output + (1 - self.fc_weight) * dual_output

    # 实例化融合的特征提取模型
    combined_model = CombinedFeatures(features_oral, features_dual, fc_weight)

    # 使用分类部分
    classifier = nn.Sequential(*classifier)

    return combined_model, classifier
