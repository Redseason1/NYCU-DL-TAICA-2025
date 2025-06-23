import argparse
import numpy as np
import os
import torch
from torch import nn
from utils import dice_score
from models.unet import UNet
from models.resnet34_unet import ResNet34UNet
from oxford_pet import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(args, model):
    if args.load_model_epoch != 0:
        model.load_state_dict(torch.load(f"saved_models/{args.model}/{args.model}_epoch_{args.load_model_epoch}.pth"))

    train_loader = DataLoader(load_dataset(args.data_path, "train"), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(load_dataset(args.data_path, "valid"), batch_size=args.batch_size, shuffle=False)

    criterion = nn.BCELoss()  # make sure loss correct
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

    losses = []
    dice_scores = []
    for i in range(args.epochs):
        model.train()
        train_loss = 0
        for sample in tqdm(train_loader):
            image = sample["image"].to(device).float()
            mask = sample["mask"].to(device).float()

            pred_mask = model(image)  # model predict
            loss = criterion(pred_mask, mask)  # calculate traning loss
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        model.eval()
        with torch.no_grad():
            dice_sum = 0
            for sample in tqdm(valid_loader):
                image = sample["image"].to(device).float()
                mask = sample["mask"].to(device).float()

                pred_mask = model(image)
                dice_sum += dice_score(pred_mask, mask)  # calculate Dice Score

            dice = dice_sum / len(valid_loader)
            print(f"Epoch {args.load_model_epoch + i + 1}, Traning Loss: {train_loss:.4f}, Dice Score: {dice:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        if args.lr_scheduler:
            scheduler.step()

        save_path = f"saved_models/{args.model}"
        losses.append(train_loss)
        dice_scores.append(dice)
        os.makedirs(save_path, exist_ok=True)

        if i % 50 == 0:
            torch.save(model.state_dict(), f"{save_path}/{args.model}_epoch_{i}.pth")

    torch.save(model.state_dict(), f"{save_path}/{args.model}_epoch_final.pth")
    np.save(f"{save_path}/{args.model}_losses.npy", losses)
    np.save(f"{save_path}/{args.model}_dice_scores.npy", dice_scores)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, default='dataset\oxford-iiit-pet' ,help='path of the input data')
    parser.add_argument("--model", type=str, default="U", help="U: unet / R: resnet34_unet")
    parser.add_argument('--epochs', '-e', type=int, default=100, help='number of epochs')    #  100  200
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='learning rate')   #   R: 1e-2  U: 1e-3
    parser.add_argument('--lr_scheduler', '-lrs', type=bool, default=True, help='learning rate scheduler')
    parser.add_argument('--load_model_epoch', '-lme', type=int, default=0, help='load model epoch')
    
    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    if args.model == "U":
        model = UNet(3, 1).to(device)
    elif args.model == "R":
        model = ResNet34UNet(3, 1).to(device)
    train(args, model)