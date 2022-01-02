import math
import torch
import argparse
from byol_pytorch import BYOL
from torchvision import models
from dataset_p2 import MiniDataset
from torch.utils.data import DataLoader
from config_p2 import (
    TRAIN_CSV, TRAIN_ROOT, VAL_CSV, VAL_ROOT,
    LR, EPOCH, BATCH_SIZE, NUM_WORKERS, DEVICE
)


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str)
    args = parser.parse_args()

    return args

def training(args):
    resnet = models.resnet50().to(DEVICE)
    if args.load is not None:
        print('loading...', args.load)
        resnet.load_state_dict(torch.load(args.load))
    learner = BYOL(
        resnet,
        image_size = 128,
        hidden_layer = 'avgpool'
    )
    opt = torch.optim.Adam(learner.parameters(), lr=LR)

    trainset = MiniDataset(TRAIN_CSV, TRAIN_ROOT)
    trainloader = DataLoader(trainset, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=True)

    losses = []
    BEST_LOSS = 20
    for epoch in range(EPOCH):
        for idx, batch_data in enumerate(trainloader):
            opt.zero_grad()
            imgs, labels = batch_data
            if idx == 0 and epoch == 0:
                print(imgs.shape)
            imgs = imgs.to(DEVICE)
            loss = learner(imgs)
            losses.append(loss.item())
            if len(losses) > len(trainloader):
                losses.pop(0)
            loss.backward()
            opt.step()
            learner.update_moving_average() # update moving average of target encoder
            if idx % (len(trainloader)//10) == 0:
                avg_loss = sum(losses)/len(losses)
                print('Epoch: {} Idx: {}/{} Loss: {}'.format(epoch, idx, len(trainloader), avg_loss))
                if avg_loss < BEST_LOSS:
                    BEST_LOSS = avg_loss
                    print('best {} save...'.format(BEST_LOSS))
                    if args.load is not None:
                        torch.save(resnet.state_dict(), './byol_resnet50_keep.pt')
                    else:
                        torch.save(resnet.state_dict(), './byol_resnet50.pt')

if __name__ == '__main__':
    args = arg_parse()
    training(args)