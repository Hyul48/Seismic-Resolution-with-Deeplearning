import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscriminatorModel(nn.Module):
    def __init__(self, input_channels):
        super(DiscriminatorModel, self).__init__()
        self.layers = nn.ModuleList()
        mapsize = 3
        layer_channels = [64, 128, 256, 512]

        # Fully convolutional layers
        in_channels = input_channels
        for out_channels in layer_channels:
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=mapsize, stride=2, padding=mapsize // 2),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = out_channels

        # Final "all convolutional net" layers
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=mapsize, stride=1, padding=mapsize // 2),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
        )

        self.layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
        )

        # Final layer to map to real/fake score
        self.final_conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_conv(x)
        x = x.mean(dim=[2, 3])  # Take mean over width and height (spatial dimensions)
        return x
    


# Example usage
disc_input = torch.randn(64, 3, 64, 64)  # Example input tensor (batch_size, channels, height, width)
discriminator = DiscriminatorModel(input_channels=3)
output = discriminator(disc_input)
print(output)
