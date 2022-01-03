import os
import torch
import argparse
import pandas as pd
from torchvision import models
from config_p2 import (
    NUM_CLASSES, SAVE_DIR, NUM_WORKERS, BATCH_SIZE, DEVICE, CIDX2LABEL
)
from dataset_p2 import OfficeHomeDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, help="input path for predicting")
    parser.add_argument("--input_dir", type=str, help="input path for predicting")
    parser.add_argument("--output_csv", type=str, help="output path for predicting")
    parser.parse_args()

    args = parser.parse_args()
    return args

def predicting(args):
    testset = OfficeHomeDataset(csv_path=args.input_csv, data_dir=args.input_dir, is_test=True)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )

    resnet = models.resnet50()
    resnet.fc = torch.nn.Linear(2048, NUM_CLASSES)
    resnet.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'resnet_finetune.pkl')))
    resnet = resnet.to(DEVICE)
    resnet.eval()
    preds = None
    with torch.no_grad():
        for idx, batch_data in enumerate(test_loader):
            imgs = batch_data
            imgs = imgs.to(DEVICE)
            output = resnet(imgs)
            _, pred = output.max(dim=1)
            preds = pred.cpu() if preds is None else torch.cat((preds, pred.cpu()), axis=0)
        preds = preds.numpy().tolist()
        write_csv = {
            "id": [i for i in range(len(preds))],
            "filename": [os.path.basename(testset.data_df.loc[i, "filename"]) for i in range(len(preds))],
            "label": [CIDX2LABEL[pred] for pred in preds]
        }
        write_csv = pd.DataFrame(write_csv)
        print(write_csv)
        write_csv.to_csv(args.output_csv, index=False)

if __name__ == '__main__':
    args = parse_args()
    predicting(args)