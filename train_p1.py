import torch
import numpy as np
from dataset_p1 import MiniDataset
from sampler_p1 import ProtoTypicalBatchSampler
from torch.utils.data import DataLoader, Dataset
from config_p1 import (TRAIN_CSV, TRAIN_ROOT, 
    CLASSES_PER_EPISODE, SAMPLES_PER_CLASS, N_SUPPORT,
    EPOCH, EPISODES, NUM_WORKERS, DEVICE,
    LR, VAL_ROOT, VAL_CSV, METRIC
)
from model_p1 import Convnet, RelationNet
from prototypical_loss import prototypical_loss

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def train():
    trainset = MiniDataset(TRAIN_CSV, TRAIN_ROOT)
    protoTypicalBatchSampler = ProtoTypicalBatchSampler(
        data_csv_root=TRAIN_CSV,
        classes_per_episode=CLASSES_PER_EPISODE,
        samples_per_class=SAMPLES_PER_CLASS,
        episodes=EPISODES
    )
    trainloader = DataLoader(
        trainset,
        num_workers=NUM_WORKERS, pin_memory=False, worker_init_fn=worker_init_fn,
        batch_sampler=protoTypicalBatchSampler
    )
    validset = MiniDataset(VAL_CSV, VAL_ROOT)
    validBatchSampler = ProtoTypicalBatchSampler(
        data_csv_root=VAL_CSV,
        classes_per_episode=CLASSES_PER_EPISODE,
        samples_per_class=SAMPLES_PER_CLASS,
        episodes=EPISODES
    )
    validloader = DataLoader(
        validset,
        num_workers=NUM_WORKERS, pin_memory=False, worker_init_fn=worker_init_fn,
        batch_sampler=validBatchSampler
    )
    model = Convnet().to(DEVICE)
    relationNet = RelationNet().to(DEVICE)
    if METRIC == 'parametric':
        optimizer = torch.optim.Adam(
            params=list(model.parameters())+list(relationNet.parameters()),
            lr=LR
        )
    else:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
    statis_loss = []
    BEST_ACC = 0
    for epoch in range(EPOCH):
        model.train()
        relationNet.train()
        for idx, batch_data in enumerate(trainloader):
            optimizer.zero_grad()
            imgs, labels = batch_data
            imgs = imgs.to(DEVICE)
            #labels = labels.to(DEVICE)
            output = model(imgs)
            #print(output.shape)
            #print(labels)
            loss_val, _ = prototypical_loss(output, labels, metric=METRIC, relationNet=relationNet)
            loss_val.backward()
            optimizer.step()

            statis_loss.append(loss_val.item())
            if len(statis_loss) > 20:
                statis_loss.pop(0)
        #print(sum(statis_loss)/len(statis_loss), sum(statis_acc)/len(statis_acc))
        model.eval()
        relationNet.eval()
        episode_accs = []
        with torch.no_grad():
            for idx, batch_data in enumerate(validloader):
                imgs, labels = batch_data
                imgs = imgs.to(DEVICE)
                output = model(imgs)
                _, correct_num = prototypical_loss(output, labels)
                num_query = CLASSES_PER_EPISODE*(SAMPLES_PER_CLASS-N_SUPPORT)
                episode_acc = correct_num.item()/num_query
                episode_accs.append(episode_acc)
            overall_acc = sum(episode_accs)/len(episode_accs)
            print('EPOCH {}'.format(epoch), overall_acc)
            if overall_acc > BEST_ACC:
                BEST_ACC = overall_acc
                print('Best', BEST_ACC)
                #torch.save(model.state_dict(), 'fewshot1.pkl')

if __name__ == '__main__':
    train()