import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(in_channels)

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

class GeneratorModel(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(GeneratorModel, self).__init__()
        mapsize = 3
        res_units = [256, 128, 96]

        self.layers = nn.ModuleList()
        
        # Add residual blocks and upscaling layers
        in_channels = input_channels
        for i, nunits in enumerate(res_units[:-1]):
            
            if i > 0:  # 첫 번째 unit이 아닐 때 이전 레이어의 채널 수를 nunits와 맞춰주는 Conv 레이어 추가
                self.layers.append(nn.Conv2d(previous_nunits, nunits, kernel_size=1))  # 1x1 Conv 추가
                # previous_nunits를 nunits로 업데이트
            previous_nunits = nunits  # 현재 nunits를 previous_nunits로 업데이트
            
            for _ in range(2):
                self.layers.append(ResidualBlock(nunits, kernel_size=mapsize))
            
            """난 별로 이렇게 구현하는 걸 좋아하지 않는데 자연스럽게 channel 갯수를 맞출수 있는 방법을 연구할 필요가 있을 것 같다."""
            
            
            # Upscaling using transposed convolution
            self.layers.append(nn.Upsample(scale_factor=2, mode='nearest'))  # Upscaling
            self.layers.append(nn.ConvTranspose2d(nunits, nunits, kernel_size=mapsize, stride=1, padding=mapsize // 2))
            self.layers.append(nn.BatchNorm2d(nunits))
            self.layers.append(nn.ReLU(inplace=True))
                    
            

        # Finalization layers
        final_units = res_units[-1]
        self.layers.append(nn.Conv2d(res_units[-2], final_units, kernel_size=1))  # 1x1 Conv 추가
        self.layers.append(nn.Conv2d(final_units, final_units, kernel_size=mapsize, stride=1, padding=mapsize // 2))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Conv2d(final_units, final_units, kernel_size=1, stride=1))
        self.layers.append(nn.ReLU(inplace=True))

        # 1채널을 256채널로 변환하는 Convolutional Layer 추가
        self.channel_conv = nn.Conv2d(1, 256, kernel_size=1)  # 1x1 Convolution

        # Last layer with sigmoid activation
        self.final_conv = nn.Conv2d(final_units, output_channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.channel_conv(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_conv(x)
        x = self.sigmoid(x)  # Sigmoid activation for final output
        return x

"""# Example usage
features = torch.randn(1, 1, 32, 32)  # Input features (batch_size, channels, height, width)
channels = 3  # Number of output channels

generator = GeneratorModel(input_channels=features.size(1), output_channels=channels)
output = generator(features)
print(output.shape)"""


