import torch
import torch.nn as nn

class Convnet(nn.Module):
    def __init__(self, in_channels=3, hid_channels=64, out_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, out_channels)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.shape[0], -1)

def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class RelationNet(nn.Module):
    def __init__(self, in_channels=3200, hid_channels=640, out_channels=1):
        super().__init__()
        def conv_block(in_channels, out_channels):
            bn = nn.BatchNorm2d(out_channels)
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                bn,
                nn.ReLU(),
                #nn.MaxPool2d(2)
            )
        self.encoder = nn.Sequential(
            conv_block(in_channels, hid_channels*4),
            conv_block(hid_channels*4, hid_channels*2),
            conv_block(hid_channels*2, hid_channels),
            conv_block(hid_channels, out_channels)
        )
    def forward(self, x):
        x = self.encoder(x)
        
        return x.squeeze()

if __name__ == '__main__':
    rand_img = torch.rand(4,3,84,84)
    model = Convnet()
    ret = model(rand_img)
    print(ret.shape)