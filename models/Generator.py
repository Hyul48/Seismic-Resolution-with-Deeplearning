import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

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

class DoubleResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DoubleResidualBlock, self).__init__()
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) 
        self.res1 = ResidualBlock(out_channels, out_channels, kernel_size=kernel_size)
        self.res2 = ResidualBlock(out_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x):
        x = self.shortcut(x)
        out = self.res1(x)
        out = self.res2(out)

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
            
            if i == 0 :
                self.layers.append(DoubleResidualBlock(1, nunits, kernel_size=mapsize))
                
            else:
                self.layers.append(DoubleResidualBlock(previous_units, nunits, kernel_size= mapsize))   
            previous_units = nunits
            
            
            # Upscaling using transposed convolution
            self.layers.append(nn.Upsample(scale_factor=2, mode='nearest'))  # Upscaling
            self.layers.append(nn.ConvTranspose2d(nunits, nunits, kernel_size=mapsize, stride=1, padding=mapsize // 2))
            self.layers.append(nn.BatchNorm2d(nunits))
            self.layers.append(nn.ReLU(inplace=True))
                    
            

        # Finalization layers
        final_units = res_units[-1]
        self.layers.append(nn.Conv2d(res_units[-2], final_units, kernel_size=mapsize, stride=1, padding=mapsize // 2))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Conv2d(final_units, final_units, kernel_size=1, stride=1))
        self.layers.append(nn.ReLU(inplace=True))


        # Last layer with sigmoid activation
        self.final_conv = nn.Conv2d(final_units, output_channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_conv(x)
        x = self.sigmoid(x)  # Sigmoid activation for final output
        return x

# Example usage
features = torch.randn(1, 1, 32, 32)  # Input features (batch_size, channels, height, width)
channels = 3  # Number of output channels

generator = GeneratorModel(input_channels=features.size(1), output_channels=channels)
output = generator(features)
print(output.shape)


