import os
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sampler_p1 import ProtoTypicalBatchSampler

# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        filenameToPILImage = lambda x: Image.open(x)
        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.data_df)

if __name__=='__main__':
    from config_p1 import TRAIN_CSV, TRAIN_ROOT
    protoTypicalBatchSampler = ProtoTypicalBatchSampler(
        data_csv_root=TRAIN_CSV,
        classes_per_episode=5, 
        samples_per_class=6,
        episodes=10
    )
    dataset = MiniDataset(TRAIN_CSV, TRAIN_ROOT)
    #print(dataset[0])
    loader = DataLoader(
        dataset,
        #num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        batch_sampler=protoTypicalBatchSampler
    )
    for idx, b in enumerate(loader):
        img, lb = b
        print(idx, img.shape)