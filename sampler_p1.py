import torch
import numpy as np
import pandas as pd

class ProtoTypicalBatchSampler(object):
    def __init__(self, data_csv_root, classes_per_episode, samples_per_class, episodes):
        super(ProtoTypicalBatchSampler, self).__init__()
        self.data_csv_root = data_csv_root
        self.classes_per_episode = classes_per_episode
        self.samples_per_class = samples_per_class
        self.episodes = episodes

        train_df = pd.read_csv(self.data_csv_root).set_index("id")
        #print(train_df.values)
        self.classes, self.counts = np.unique(train_df.values[:,1], return_counts=True)
        #print(self.classes, self.counts)
        self.indexes = np.ones((self.classes.shape[0], self.counts.max()), dtype=int) * (-1)
        #print(self.indexes.shape)
        
        for idx, label in enumerate(train_df.values[:,1]):
            cidx, = np.nonzero(self.classes==label)
            cidx = cidx.item()
            row, = np.nonzero(self.indexes[cidx]==(-1))
            idx_to_insert = row.min()
            self.indexes[cidx, idx_to_insert] = idx
        #self.counts = torch.tensor(self.counts)
        #self.indexes = torch.tensor(self.indexes)
        #print(self.indexes)
    
    def __iter__(self):
        #print(self.indexes)
        for episode in range(self.episodes):
            #batch_size = self.samples_per_class* self.classes_per_episode
            #batch_data = torch.zeros(batch_size)
            #print(batch_data)
            ### random sample self.classes_per_episode classes
            #cidxs = torch.randperm(self.classes.shape[0])[:self.classes_per_episode]
            cidxs = np.random.choice(self.classes.shape[0], self.classes_per_episode)
            #print(self.indexes[cidxs])
            #print(self.counts[cidxs])
            sidxs = np.floor(
                np.expand_dims(self.counts[cidxs], axis=1)*
                np.random.rand(self.classes_per_episode, self.samples_per_class)
            ).astype(int)
            """
            sidxs = torch.floor(
                self.counts[cidxs].unsqueeze(1)* 
                torch.rand(self.classes_per_episode, self.samples_per_class)
            ).type(torch.int64)
            """
            #print(sidxs.flatten())
            #samples2D = torch.gather(self.indexes, dim=1, index=sidxs)
            #print(sidxs)
            #print(self.indexes)
            samples2D = np.take_along_axis(self.indexes[cidxs], indices=sidxs, axis=1)
            #print(samples2D)
            #print(self.indexes[cidxs, sidxs])
            batch_data = samples2D.flatten()
            yield batch_data
    
    def __len__(self):
        return self.episodes

if __name__ == '__main__':
    from config_p1 import TRAIN_CSV
    protoTypicalBatchSampler = ProtoTypicalBatchSampler(
        data_csv_root=TRAIN_CSV,
        classes_per_episode=5, 
        samples_per_class=6,
        episodes=10
    )
    #protoTypicalBatchSampler_iter = iter(protoTypicalBatchSampler)
    for epoch in range(3):
        for idx, a in enumerate(protoTypicalBatchSampler):
            print(idx)
    #s = next(protoTypicalBatchSampler_iter)
    #print(s)
    #ProtoTypicalBatchSampler()