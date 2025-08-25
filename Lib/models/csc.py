import pandas as pd
import torch.nn as nn
import numpy as np
import torch
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from .CoRe import DictBlock
#from .sdnet import DictConv2d, cfg
from Lib.config import config as _cfg
cfg = _cfg

#pd.set_option('display.max_rows', None)#显示全部行
#pd.set_option('display.max_columns', None)#显示全部列
#np.set_printoptions(threshold=np.inf)
class DictConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(DictConv2d, self).__init__()

        self.dn = DictBlock(
            in_channels, out_channels, stride=stride, kernel_size=kernel_size, padding=padding,
            mu=cfg['MODEL']['MU'], lmbd=cfg['MODEL']['LAMBDA'][0], square_noise=cfg['MODEL']['SQUARE_NOISE'],
            n_dict=cfg['MODEL']['EXPANSION_FACTOR'], non_negative=cfg['MODEL']['NONEGATIVE'],
            n_steps=cfg['MODEL']['NUM_LAYERS'], FISTA=cfg['MODEL']['ISFISTA'], w_norm=cfg['MODEL']['WNORM']
        )
        #self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        # print(cfg)
        # print("==================")
        self.register_buffer('running_c_loss', torch.tensor(0, dtype=torch.float))

    def forward(self, x):
        out, rc = self.dn(x)
        #out = self.gap(out)

        if self.training:
            self.running_c_loss = 0.99 * self.running_c_loss + (1 - 0.99) * rc[0].item()
        #print(rc)
        return out