import torch
import torch.nn as nn
import os
import argparse
import numpy as np
import shutil

from datasets import get_images, get_dataset, get_data_loaders
from engine import train, validate
from model import prepare_model, get_scheduler
from config import ALL_CLASSES, LABEL_COLORS_LIST
from utils import save_model, SaveBestModel, save_plots

parser = argparse.ArgumentParser()
parser.add_argument(
    '--epochs',
    default=10,
    help='number of epochs to train for',
    type=int
)
parser.add_argument(
    '--lr',
    default=0.0001,
    help='learning rate for optimizer',
    type=float
)
parser.add_argument(
    '--batch',
    default=4,
    help='batch size for data loader',
    type=int
)
args = parser.parse_args()
print(args)

if __name__ == '__main__':
    out_dir = os.path.join('..', 'outputs')
    out_dir_valid_preds = os.path.join('..', 'outputs', 'valid_preds')

    # Delete model files if they exist.
    for model_file in ['best_model.pth', 'model.pth']:
        model_path = os.path.join(out_dir, model_file)
        if os.path.exists(model_path):
            os.remove(model_path)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_valid_preds, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = prepare_model(num_classes=len(ALL_CLASSES)).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    get_scheduler(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    train_images, train_masks, valid_images, valid_masks = get_images(
        root_path='../input/Water_Bodies_Dataset_Split'
    )

    classes_to_train = ALL_CLASSES

    train_dataset, valid_dataset = get_dataset(
        train_images,
        train_masks,
        valid_images,
        valid_masks,
        ALL_CLASSES,
        classes_to_train,
        LABEL_COLORS_LIST,
        img_size=256
    )

    train_dataloader, valid_dataloader = get_data_loaders(
        train_dataset, valid_dataset, batch_size=args.batch
    )

    save_best_model = SaveBestModel()

    EPOCHS = args.epochs
    train_loss, train_pix_acc, train_iou = [], [], []
    valid_loss, valid_pix_acc, valid_iou = [], [], []
    for epoch in range(EPOCHS):
        print(f"EPOCH: {epoch + 1}")
        train_epoch_loss, train_epoch_pixacc, train_epoch_iou = train(
            model,
            train_dataset,
            train_dataloader,
            device,
            optimizer,
            criterion,
            classes_to_train
        )
        valid_epoch_loss, valid_epoch_pixacc, valid_epoch_iou = validate(
            model,
            valid_dataset,
            valid_dataloader,
            device,
            criterion,
            classes_to_train,
            LABEL_COLORS_LIST,
            epoch,
            ALL_CLASSES,
            save_dir=out_dir_valid_preds
        )
        train_loss.append(train_epoch_loss)
        train_pix_acc.append(train_epoch_pixacc.cpu())
        train_iou.append(train_epoch_iou)
        valid_loss.append(valid_epoch_loss)
        valid_pix_acc.append(valid_epoch_pixacc.cpu())
        valid_iou.append(valid_epoch_iou)

        save_best_model(
            valid_epoch_iou, epoch, model
        )
        print('-' * 50)
        print("Epoch Summary:")
        print(f"Train Epoch Loss: {train_epoch_loss:.4f}, Train Epoch PixAcc: {train_epoch_pixacc:.4f}, Train Epoch IoU: {np.nanmean(train_epoch_iou):.4f}")
        print(f"Valid Epoch Loss: {valid_epoch_loss:.4f}, Valid Epoch PixAcc: {valid_epoch_pixacc:.4f}, Valid Epoch IoU: {valid_epoch_iou:.4f}")
        print('-' * 50)

    save_model(EPOCHS, model, optimizer, criterion, out_dir)
    save_plots(
        train_pix_acc, valid_pix_acc, train_loss, valid_loss, train_iou, valid_iou, out_dir
    )
    print('TRAINING COMPLETE')