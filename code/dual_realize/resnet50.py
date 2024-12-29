import math
import cv2
import torch

import torch.nn as nn
from torch.hub import load_state_dict_from_url

def clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(10,10))

    clahe_image = clahe.apply(image)
    
    return clahe_image

def lbp(image):
    rows, cols = image.shape
    lbp_image = np.zeros((rows, cols), dtype=np.uint8)
    
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            center = image[i, j]
            code = 0

            code |= (image[i-1, j] >= center) << 7
            code |= (image[i-1, j+1] >= center) << 6
            code |= (image[i, j+1] >= center) << 5
            code |= (image[i+1, j+1] >= center) << 4
            code |= (image[i+1, j] >= center) << 3
            code |= (image[i+1, j-1] >= center) << 2
            code |= (image[i, j-1] >= center) << 1
            code |= (image[i-1, j-1] >= center) << 0
            
            lbp_image[i, j] = code
    
    return lbp_image

def sobel_process(x):
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
    
def fft_process(x):
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
        #y = clahe(x) #使用clahe
        #y = lbp(x)
        #y = fft_process(x)
        y = sobel_process(x)
        x += y
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

def resnet50(pretrained = False):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-19c8e357.pth", model_dir="./model_data")
        model.load_state_dict(state_dict)
    #----------------------------------------------------------------------------#
    #   获取特征提取部分，从conv1到model.layer3，最终获得一个38,38,1024的特征层
    #----------------------------------------------------------------------------#
    features    = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    #----------------------------------------------------------------------------#
    #   获取分类部分，从model.layer4到model.avgpool
    #----------------------------------------------------------------------------#
    classifier  = list([model.layer4, model.avgpool])
    
    features    = nn.Sequential(*features)
    classifier  = nn.Sequential(*classifier)
    return features, classifier

def resnet50_sobel(pretrained = False):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-19c8e357.pth", model_dir="./model_data")
        model.load_state_dict(state_dict)
    #----------------------------------------------------------------------------#
    #   获取特征提取部分，从conv1到model.layer3，最终获得一个38,38,1024的特征层
    #----------------------------------------------------------------------------#
    features    = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    #----------------------------------------------------------------------------#
    #   获取分类部分，从model.layer4到model.avgpool
    #----------------------------------------------------------------------------#
    classifier  = list([model.layer4, model.avgpool])
    
    features    = nn.Sequential(*features)
    classifier  = nn.Sequential(*classifier)
    return features, classifier