import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from model import ViTLightningModule
from utils import make_weights_for_balanced_classes, collate_fn
import config




if __name__ == "__main__":
    # make slight augmentation and normalization on ImageNet statistics
    trans = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_ds = ImageFolder(config.TRAIN_DATASET, transform=trans)
    val_ds = ImageFolder(config.VAL_DATASET, transform=trans)
    test_ds = ImageFolder(config.TEST_DATASET, transform=trans)

    # deal with class disbalance
    weights = make_weights_for_balanced_classes(train_ds.imgs, len(train_ds.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_dataloader = DataLoader(train_ds, collate_fn=collate_fn, batch_size=config.BATCH_SIZE, sampler = sampler)
    val_dataloader = DataLoader(val_ds, collate_fn=collate_fn, batch_size=config.BATCH_SIZE)
    test_dataloader = DataLoader(test_ds, collate_fn=collate_fn, batch_size=config.BATCH_SIZE)


    # for early stopping, see https://pytorch-lightning.readthedocs.io/en/1.0.0/early_stopping.html?highlight=early%20stopping
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        strict=False,
        verbose=False,
        mode='min',
    )

    model = ViTLightningModule()
    model.train()
    trainer = Trainer(gpus=1, callbacks=[EarlyStopping(monitor='val_loss')])
    trainer.fit(model)
