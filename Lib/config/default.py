# Modified based on the MDEQ repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

_C.LOG_DIR = 'home/logs'

# common params for NETWORK
_C.MODEL.NUM_LAYERS = 1
_C.MODEL.ISFISTA = False
_C.MODEL.EXPANSION_FACTOR = 1
_C.MODEL.LAMBDA = [0.1]
_C.MODEL.ADAPTIVELAMBDA = False
_C.MODEL.NONEGATIVE = True
_C.MODEL.WNORM = True
_C.MODEL.DICTLOSS = False
_C.MODEL.RCLOSS_FACTOR = 0.0
_C.MODEL.ORTHO_COEFF = 0.0
_C.MODEL.MU = 0.0
_C.MODEL.SHORTCUT = True
_C.MODEL.PAD_MODE = 'constant'
_C.MODEL.POOLING = False
_C.MODEL.SQUARE_NOISE = True
_C.MODEL.STEP_SIZE = 0.1


def update_config(cfg, args):
    cfg.defrost()
    if args.cfg:
        cfg.merge_from_file(args.cfg)

    # if args.testModel:
    #     cfg.TEST.MODEL_FILE = args.testModel

    cfg.merge_from_list(args.opts)

    cfg.freeze()


if __name__ == '__main__':
    import sys

    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
