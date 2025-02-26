import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

# Define the Efficient Channel Attention (ECA) block.
class ECABlock(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECABlock, self).__init__()
        kernel_size = int(abs((torch.log2(torch.tensor(channel, dtype=torch.float32)) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)

# Modify the DeepLabv3+ model to include ECA.
def prepare_model(num_classes=2):
    model = deeplabv3_resnet50(weights='DEFAULT')

    # Add ECA to the backbone's output before ASPP.
    class BackboneWithECA(nn.Module):
        def __init__(self, backbone):
            super(BackboneWithECA, self).__init__()
            self.backbone = backbone
            self.eca = ECABlock(channel=2048)  # ResNet50 output channels.

        def forward(self, x):
            x = self.backbone(x)
            x['out'] = self.eca(x['out'])
            return x

    # Wrap the backbone with ECA.
    model.backbone = BackboneWithECA(model.backbone)

    # Update the classifier to match the number of classes.
    model.classifier[4] = nn.Conv2d(256, num_classes, 1)
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1)

    return model

# Example usage with ReduceLROnPlateau scheduler:
def get_scheduler(optimizer, T_0=5, T_mult=2, eta_min=1e-6):
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=5,  # Number of epochs before restart
        T_mult=1,  # Increases cycle length after each restart
        eta_min=1e-6  # Minimum LR
    )


"""
Uncomment the following lines to train on the DeepLabV3 ResNet101 model.
"""
# import torch.nn as nn

# from torchvision.models.segmentation import deeplabv3_resnet101
 
# def prepare_model(num_classes=2):
#     model = deeplabv3_resnet101(weights='DEFAULT')
#     model.classifier[4] = nn.Conv2d(256, num_classes, 1)
#     model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1)
#     return model