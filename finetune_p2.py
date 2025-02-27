import os
import torch
import argparse
import torch.nn as nn
from config_p2 import (
    FINE_TUNE_TRAIN_CSV, FINE_TUNE_TRAIN_ROOT, FINE_TUNE_VAL_CSV, FINE_TUNE_VAL_ROOT,
    NUM_WORKERS, FINE_TUNE_EPOCH, BATCH_SIZE, FINE_TUNE_LR, DEVICE, NUM_CLASSES
)
from torchvision import models
from torch.utils.data import DataLoader
from dataset_p2 import OfficeHomeDataset

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str)
    parser.add_argument("--freeze_backbone", action="store_true")
    args = parser.parse_args()

    return args

def training(args):
    resnet = models.resnet50()
    if args.load is not None:
        print('loading...', args.load)
        resnet.load_state_dict(torch.load(args.load))
    else:
        print('without pretraining...')
    
    resnet.fc = nn.Linear(2048, NUM_CLASSES)
    if args.freeze_backbone:
        for param in resnet.parameters():
            param.requires_grad = False
        resnet.fc.weight.requires_grad = True
        resnet.fc.bias.requires_grad = True
        
    resnet = resnet.to(DEVICE)
    
    opt = torch.optim.Adam(resnet.parameters(), lr=FINE_TUNE_LR)
    criterion = nn.CrossEntropyLoss()

    finetune_trainset = OfficeHomeDataset(FINE_TUNE_TRAIN_CSV, FINE_TUNE_TRAIN_ROOT)
    finetune_trainloader = DataLoader(finetune_trainset, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=True)
    finetune_valset = OfficeHomeDataset(FINE_TUNE_VAL_CSV, FINE_TUNE_VAL_ROOT, is_valid=True)
    finetune_valloader = DataLoader(finetune_valset, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=False)

    losses = []
    BEST_ACC = 0
    for epoch in range(FINE_TUNE_EPOCH):
        resnet.train()
        for idx, batch_data in enumerate(finetune_trainloader):
            opt.zero_grad()
            imgs, labels = batch_data
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            output = resnet(imgs)
            loss = criterion(output, labels)
            
            losses.append(loss.item())
            if len(losses) > len(finetune_trainloader):
                losses.pop(0)
            if idx % (len(finetune_trainloader)//10) == 0:
                avg_loss = sum(losses)/len(losses)
                print('Epoch: {} Idx: {}/{} Loss: {}'.format(epoch, idx, len(finetune_trainloader), avg_loss))

            loss.backward()
            opt.step()
        resnet.eval()
        total_acc = 0
        with torch.no_grad():
            for idx, batch_data in enumerate(finetune_valloader):
                imgs, labels = batch_data
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                output = resnet(imgs)
                _, pred = output.max(dim=1)
                acc = (pred == labels).sum()
                total_acc += acc
            print(total_acc, len(finetune_valloader.dataset))
            if total_acc > BEST_ACC:
                BEST_ACC = total_acc
                print('best {} save...'.format(BEST_ACC))
                torch.save(resnet.state_dict(), 'resnet_finetune.pkl')

if __name__ == '__main__':
    args = arg_parse()
    training(args)