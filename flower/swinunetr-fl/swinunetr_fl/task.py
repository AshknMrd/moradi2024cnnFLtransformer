"""First-example: A Flower / PyTorch app."""
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=FutureWarning)

import math
import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
from monai.losses import FocalLoss
from monai.metrics import DiceMetric
from monai.data import Dataset, load_decathlon_datalist, decollate_batch
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    RandFlipd,
    SpatialPadd,
    CenterSpatialCropd,
    RandShiftIntensityd,
    Spacingd,
    RandRotate90d,
    ScaleIntensityRangePercentilesd,
)

current_date = datetime.datetime.now().strftime("%Y-%m-%d")

# -------------------------------------------------------------------------
# TRANSFORMS
# -------------------------------------------------------------------------
train_transforms_simple = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(0.5, 0.5, 3.0), mode=("bilinear", "nearest")),
    SpatialPadd(keys=["image", "label"], spatial_size=(256, 256, 32)),
    ScaleIntensityRangePercentilesd(
        keys=["image"], lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True
    ),
    RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
    RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
    RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
    RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
    RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(0.5, 0.5, 3.0), mode=("bilinear", "nearest")),
    SpatialPadd(keys=["image", "label"], spatial_size=(256, 256, 32)),
    CenterSpatialCropd(keys=["image", "label"], roi_size=(256, 256, 32)),
    ScaleIntensityRangePercentilesd(
        keys=["image"], lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True
    ),
])

# -------------------------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------------------------
def load_data(
    partition_id: int,
    num_partitions: int,
    dataset_json="./workdir/dataset_unetr_picai.json",
    num_train=600,
    num_val=100,
):
    """Load and partition local dataset."""
    datalist = load_decathlon_datalist(dataset_json, True, "training")[:num_train]
    val_files = load_decathlon_datalist(dataset_json, True, "validation")[:num_val]

    total = len(datalist)
    images_per_partition = math.ceil(total / num_partitions)
    start = partition_id * images_per_partition
    end = min(start + images_per_partition, total)

    train_subset = datalist[start:end]

    train_ds = Dataset(data=train_subset, transform=train_transforms_simple)
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=8, pin_memory=True
    )

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader

# -------------------------------------------------------------------------
# TRAINING
# -------------------------------------------------------------------------
def train(model, trainloader, epochs, device):
    """Train the model on the training set."""
    model.to(device)
    model.train()

    loss_function = FocalLoss(to_onehot_y=True, gamma=2.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    total_loss = 0.0

    for _ in range(epochs):
        epoch_loss = 0.0
        step = 0

        for step, batch in enumerate(trainloader, start=1):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        total_loss += epoch_loss / step

    return total_loss / epochs

# -------------------------------------------------------------------------
# VALIDATION
# -------------------------------------------------------------------------
def test(model, valloader, device, post_label, post_pred):
    """Validate the model on the validation set."""
    model.to(device)
    model.eval()

    loss_function = FocalLoss(to_onehot_y=True, gamma=2.0).to(device)
    dice_metric = DiceMetric(
        include_background=False, reduction="mean", get_not_nans=False
    )

    total_loss = 0.0

    with torch.no_grad():
        for batch in valloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            with torch.amp.autocast("cuda", enabled=True):
                outputs = model(images)

            loss = loss_function(outputs, labels).item()
            total_loss += loss

            # For batch size 1
            label_post = post_label(labels.squeeze(0))
            pred_post = post_pred(outputs.squeeze(0))

            dice_metric(y_pred=pred_post, y=label_post)

    avg_loss = total_loss / len(valloader)
    avg_dice = dice_metric.aggregate().item()
    dice_metric.reset()

    print(f"Loss: {avg_loss}, Dice: {avg_dice}")
    return avg_loss, avg_dice

# -------------------------------------------------------------------------
# FL CLIENT HELPERS
# -------------------------------------------------------------------------
def get_weights(net):
    """Return model weights as a list of NumPy arrays."""
    return [p.cpu().numpy() for p in net.state_dict().values()]

def set_weights(net, parameters):
    """Set model weights from list of NumPy arrays."""
    state_dict = OrderedDict({
        k: torch.tensor(v) for k, v in zip(net.state_dict().keys(), parameters)
    })
    net.load_state_dict(state_dict, strict=True)