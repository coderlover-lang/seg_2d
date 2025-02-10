import torch 
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size=16, downsample=None):
        super().__init__()
        stride = 1 if downsample is None else kernel_size // 2 
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResnetNormalBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out 
    
class ResnetReshapeBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size):
        super().__init__()
        stride = kernel_size // 2
        padding = kernel_size // 4
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=False)
        self.down_sample = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        identity = self.down_sample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out 
    
class Resnet(nn.Module):
    def __init__(
        self, 
        kernel_sizes = [8, 4, 4, 4], 
        in_features = [1, 64, 128, 256],
        out_fetures = [64, 128, 256, 512],
        depths = [3, 4, 6, 3]
    ):
        super().__init__()
        self.stage1 = self._make_layers(in_features[0], out_fetures[0], kernel_sizes[0], depths[0])
        self.stage2 = self._make_layers(in_features[1], out_fetures[1], kernel_sizes[1], depths[1])
        self.stage3 = self._make_layers(in_features[2], out_fetures[2], kernel_sizes[2], depths[2])
        self.stage4 = self._make_layers(in_features[3], out_fetures[3], kernel_sizes[3], depths[3])
        self.total_stage = len(depths)
    def _make_layers(self, in_features, out_features, kernel_size, depth):
        layers = []
        for i in range(depth):
            if i == 0:
                layers.append(ResnetReshapeBlock(in_features, out_features, kernel_size))
            else:
                layers.append(ResnetNormalBlock(out_features, out_features))
        return nn.Sequential(*layers)
    
    def forward(self, X):
        layer_output = []
        for i in range(self.total_stage):
            X = getattr(self, f"stage{i+1}")(X)
            layer_output.append(X)
        return layer_output
        

if __name__ == "__main__":
    device = torch.device("cpu")
    X = torch.rand(16, 1, 128, 128).to(device)
    bb = Resnet()
    print([i.shape for i in bb(X)])