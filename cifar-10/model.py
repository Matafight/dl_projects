import torch.nn as nn
import torch



## 定义残差块
## 包含两个卷积层 kernel_size=3*3, padding=1, strides=1
import torch.nn.functional as F
class Residual(nn.Module):
    def __init__(self,in_channels,out_channels,use_1x1conv=False,stride=1):
        super(Residual,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,stride=stride,padding=1,kernel_size=3)
        self.conv2 = nn.Conv2d(out_channels,out_channels,stride=1,padding=1,kernel_size=3)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride)
        else:
            self.conv3 = None
        ## 3个channel
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        self.y = F.relu(self.bn1(self.conv1(x)))
#         print(self.y.shape)
        self.y = F.relu(self.bn2(self.conv2(self.y)))
        if self.conv3:
            x=self.conv3(x)
        return x+self.y


def resnet_block(in_channels, num_channels, num_residuals, first_block=False):
    inner_net = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            inner_net.add_module('res_block' + str(num_channels) + str(i),
                                 Residual(in_channels, num_channels, use_1x1conv=True, stride=2))
        else:
            inner_net.add_module('res_block' + str(num_channels) + str(i), Residual(num_channels, num_channels))
    return inner_net


class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.net.add_module('total_res_block1', resnet_block(64, 64, 2, first_block=True))
        self.net.add_module('total_res_block2', resnet_block(64, 128, 2))
        self.net.add_module('total_res_block3', resnet_block(128, 256, 2))
        self.net.add_module('total_res_block4', resnet_block(256, 512, 2))
        self.net.add_module('global_avg_pooling', nn.AdaptiveAvgPool2d((1)))
        self.last_layer = nn.Linear(512, 10)

    def forward(self, x):
        y = self.net(x)
        y = torch.squeeze(y)
        y = self.last_layer(y)
        return y
