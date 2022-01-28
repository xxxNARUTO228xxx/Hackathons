import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
from PIL import Image

import config
from model import ViTLightningModule




# collate_fn for test
def collate_fn(examples):
    pixel_values = torch.stack([example[0] for example in examples])
    labels = [example[1] for example in examples]
    return {"pixel_values": pixel_values, "labels": labels}


class InferenceDataset(Dataset):
    def __init__(self, root_dir="./data/test", img_size=config.IMG_SIZE): # im_size = real img size, but im_size for model is 256
        self.img_size = img_size
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        self.trans = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = Image.open(img_path)

        image = self.trans(image)
        print(img_path)

        return image, img_path.split('/')[-1]


if __name__ == "__main__":
    test_ds = InferenceDataset()
    test_dataloader = DataLoader(test_ds, collate_fn=collate_fn, batch_size=1)

    model = ViTLightningModule()
    model.load_state_dict(torch.load("./data/weight/model.pt"))
    model.eval()

    sub_df = pd.DataFrame(columns=['image_name', 'class_id'])

    preds = []
    names = []
    with torch.no_grad():
        for x in test_dataloader:
            out = model(x['pixel_values'])
            preds.append(np.argmax(np.array(out), axis=1))
            names.append(x['labels'])

    sub_df['class_id'] = np.array(preds).flatten()
    sub_df['image_name'] = np.array(names).flatten()
    sub_df.to_csv("./data/out/submission.csv", index=False, sep='\t')
