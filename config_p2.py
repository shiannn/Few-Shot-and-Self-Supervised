import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HW4_ROOT = os.path.join(BASE_DIR, 'hw4_data')
DATAROOT = os.path.join(HW4_ROOT, 'mini')
TRAIN_ROOT = os.path.join(DATAROOT, 'train')
TRAIN_CSV = os.path.join(DATAROOT, 'train.csv')
VAL_ROOT = os.path.join(DATAROOT, 'val')
VAL_CSV = os.path.join(DATAROOT, 'val.csv')

FINE_TUNE_DATAROOT = os.path.join(HW4_ROOT, 'office')
FINE_TUNE_TRAIN_ROOT = os.path.join(FINE_TUNE_DATAROOT, 'train')
FINE_TUNE_TRAIN_CSV = os.path.join(FINE_TUNE_DATAROOT, 'train.csv')
FINE_TUNE_VAL_ROOT = os.path.join(FINE_TUNE_DATAROOT, 'val')
FINE_TUNE_VAL_CSV = os.path.join(FINE_TUNE_DATAROOT, 'val.csv')
LABEL2CIDX = {'Alarm_Clock': 0, 'Backpack': 1, 'Batteries': 2, 'Bed': 3, 'Bike': 4, 'Bottle': 5, 'Bucket': 6, 'Calculator': 7, 'Calendar': 8, 'Candles': 9, 'Chair': 10, 'Clipboards': 11, 'Computer': 12, 'Couch': 13, 'Curtains': 14, 'Desk_Lamp': 15, 'Drill': 16, 'Eraser': 17, 'Exit_Sign': 18, 'Fan': 19, 'File_Cabinet': 20, 'Flipflops': 21, 'Flowers': 22, 'Folder': 23, 'Fork': 24, 'Glasses': 25, 'Hammer': 26, 'Helmet': 27, 'Kettle': 28, 'Keyboard': 29, 'Knives': 30, 'Lamp_Shade': 31, 'Laptop': 32, 'Marker': 33, 'Monitor': 34, 'Mop': 35, 'Mouse': 36, 'Mug': 37, 'Notebook': 38, 'Oven': 39, 'Pan': 40, 'Paper_Clip': 41, 'Pen': 42, 'Pencil': 43, 'Postit_Notes': 44, 'Printer': 45, 'Push_Pin': 46, 'Radio': 47, 'Refrigerator': 48, 'Ruler': 49, 'Scissors': 50, 'Screwdriver': 51, 'Shelf': 52, 'Sink': 53, 'Sneakers': 54, 'Soda': 55, 'Speaker': 56, 'Spoon': 57, 'TV': 58, 'Table': 59, 'Telephone': 60, 'ToothBrush': 61, 'Toys': 62, 'Trash_Can': 63, 'Webcam': 64}

NUM_WORKERS = 4
EPOCH = 160
LR=3e-4
BATCH_SIZE = 32
FINE_TUNE_LR = 1e-3
NUM_CLASSES = 65

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")