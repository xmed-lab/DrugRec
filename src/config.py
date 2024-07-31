#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/29 13:45
# @Author  : Anonymous
# @Site    : 
# @File    : config.py
# @Software: PyCharm

# #Desc:

class DrugConfig():
    # data_info parameter
    MODEL = "DrugRec"
    TASK = 'MIII'
    RATIO = 2/3 # train-test split

    # train parameter
    SEED = 2023
    USE_CUDA = True
    GPU = '0'
    EPOCH = 50
    DIM = 64
    LR = 5e-4 # 3e-4; III 5e-4
    BATCH = 32
    WD = 0
    DDI = 0.06 # target ddi
    KP = 0.08 
    HIST = 3 

    # dir parameter
    ROOT = '/root/autodl-tmp/DrugRec/data/'
    LOG = '/root/autodl-tmp/DrugRec/log'



config = vars(DrugConfig)
config = {k:v for k,v in config.items() if not k.startswith('__')}
