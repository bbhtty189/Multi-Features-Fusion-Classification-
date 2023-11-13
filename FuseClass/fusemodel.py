import torch
import torch.nn as nn
from models.resnet_origin import resnet50

class fusemodel(nn.Module):
    def __init__(self, num_classes, include_top):
        super(fusemodel, self).__init__()

        self.num_classes = num_classes
        self.include_top = include_top

        self.res = resnet50(self.num_classes, self.include_top)



