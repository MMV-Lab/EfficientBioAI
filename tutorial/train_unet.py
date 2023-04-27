import torch
import argparse
from monai.losses import DiceLoss
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
from data import generate_data_dict, train_transform
from custom import train
from model.unet import Unet


def main():
    seed_value = 2023
    torch.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    set_determinism(seed=seed_value)

    parser = argparse.ArgumentParser(description="Train the UNet")
    parser.add_argument(
        "--data_path",
        type=str,
        help="path to the data",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        help="path to the ground truth",
    )
    parser.add_argument(
        "--num_epoch",
        type=int,
        default=20,
        help="number of epochs",
    )
    args = parser.parse_args()
    net = Unet(in_channels=1, classes=2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = Dataset(
        data=generate_data_dict(args.data_path, args.gt_path), transform=train_transform
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    model = train(net, dataloader, args.num_epoch, device)

    torch.save(model.state_dict(), "./unet.pth")


if __name__ == "__main__":
    main()
