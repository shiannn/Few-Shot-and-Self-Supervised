import numpy as np
from dataset_p1 import MiniDataset
from sampler_p1 import ProtoTypicalBatchSampler
from torch.utils.data import DataLoader, Dataset
from config_p1 import (TRAIN_CSV, TRAIN_ROOT, 
    CLASSES_PER_EPISODE, SAMPLES_PER_CLASS, 
    EPOCH, EPISODES, NUM_WORKERS, DEVICE
)
from model_p1 import Convnet

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
    model = Convnet().to(DEVICE)
    for epoch in range(EPOCH):
        for idx, batch_data in enumerate(trainloader):
            imgs, labels = batch_data
            imgs = imgs.to(DEVICE)
            #labels = labels.to(DEVICE)
            output = model(imgs)
            print(output.shape)
            print(labels)
        break

if __name__ == '__main__':
    train()