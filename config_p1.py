import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HW4_ROOT = os.path.join(BASE_DIR, 'hw4_data')
DATAROOT = os.path.join(HW4_ROOT, 'mini')
TRAIN_ROOT = os.path.join(DATAROOT, 'train')
TRAIN_CSV = os.path.join(DATAROOT, 'train.csv')
VAL_ROOT = os.path.join(DATAROOT, 'val')
VAL_CSV = os.path.join(DATAROOT, 'val.csv')

if __name__ == '__main__':
    print(BASE_DIR)
    print(TRAIN_CSV)