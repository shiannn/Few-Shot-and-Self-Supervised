import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HW4_ROOT = os.path.join(BASE_DIR, 'hw4_data')
DATAROOT = os.path.join(HW4_ROOT, 'mini')
TRAIN_ROOT = os.path.join(DATAROOT, 'train')
TRAIN_CSV = os.path.join(DATAROOT, 'train.csv')
VAL_ROOT = os.path.join(DATAROOT, 'val')
VAL_CSV = os.path.join(DATAROOT, 'val.csv')

CLASSES_PER_EPISODE = 5 ### ways
N_SUPPORT = 1 ### 1-shot
SAMPLES_PER_CLASS = N_SUPPORT+15 ### 15 querys
EPISODES = 600
NUM_WORKERS = 4
EPOCH = 40
LR = 1e-3

METRIC = 'cos'

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    print(BASE_DIR)
    print(TRAIN_CSV)