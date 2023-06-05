import torch
from tqdm.contrib import tenumerate
from monai.losses import DiceLoss
from typing import Iterable, Any


def infer(model: Any, data: Iterable, calib_num: int, device: Any) -> Any:
    """used for calibrating the model during the quantization process"""
    model.eval()
    with torch.no_grad():
        for i, x in tenumerate(data):
            model(x["img"].as_tensor())
            if i >= calib_num:
                break
    return model


def train(model, dataloader, device=torch.device("cpu"), num_epoch=20):
    model.to(device)
    criterion = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    model.train()
    for i in range(num_epoch):
        epoch_loss = 0
        for j, batch_data in tenumerate(dataloader):
            data, label = batch_data["img"].as_tensor(), batch_data["seg"].as_tensor()
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(
            f"fine tune epoch {i+1}/{num_epoch}, avg loss: {epoch_loss / len(dataloader)}"
        )
        scheduler.step()
    return model
