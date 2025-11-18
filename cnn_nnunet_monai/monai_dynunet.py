import os
import json
import datetime
import warnings
import argparse

import torch
from torch.utils.data import DataLoader

from monai.losses import FocalLoss
from monai.metrics import DiceMetric
from monai.networks.nets import DynUNet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, SpatialPadd, CenterSpatialCropd, RandFlipd,
    RandRotate90d, RandShiftIntensityd, ScaleIntensityRangePercentilesd,
    AsDiscrete
)
from monai.data import Dataset, decollate_batch

from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)


# ----------------------------
# CLI Arguments
# ----------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Train DynUNet with MONAI and nnU-Net dataset")
    parser.add_argument("--root_dir", type=str, default="./raw_data", help="Root directory for outputs")
    parser.add_argument("--datasets_json", type=str, required=True, help="Path to dataset JSON file")
    parser.add_argument("--num_train", type=int, default=10, help="Number of training samples")
    parser.add_argument("--num_val", type=int, default=2, help="Number of validation samples")
    parser.add_argument("--max_iterations", type=int, default=5, help="Number of global training iterations")
    parser.add_argument("--eval_num", type=int, default=1, help="Frequency of evaluation")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training and validation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: 'cuda' or 'cpu'")
    return parser.parse_args()


# ----------------------------
# Main function
# ----------------------------
def main():
    args = get_args()

    # Set up directories
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    root_dir = os.path.join(args.root_dir, current_date)
    os.makedirs(root_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # ----------------------------
    # Transforms
    # ----------------------------
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(0.5, 0.5, 3.0), mode=("bilinear", "nearest")),
        SpatialPadd(keys=["image", "label"], spatial_size=(256, 256, 32)),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True),
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
        ScaleIntensityRangePercentilesd(keys=["image"], lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True),
    ])

    # ----------------------------
    # Load datasets
    # ----------------------------
    train_files = load_decathlon_datalist(args.datasets_json, True, "training")[:args.num_train]
    val_files = load_decathlon_datalist(args.datasets_json, True, "validation")[:args.num_val]

    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ----------------------------
    # Model, Loss, Optimizer
    # ----------------------------
    model = DynUNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=2,
        filters=[32, 64, 128, 256, 320],
        kernel_size=[3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2],
        upsample_kernel_size=[2, 2, 2, 2],
        norm_name="INSTANCE",
        res_block=True
    ).to(device)

    loss_function = FocalLoss(to_onehot_y=True, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    torch.backends.cudnn.benchmark = True

    post_label = AsDiscrete(to_onehot=2)
    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    # ----------------------------
    # Training Loop
    # ----------------------------
    epoch_loss_values = []
    metric_values = []
    dice_val_best = 0.0
    global_step_best = 0

    for global_step in range(args.max_iterations):
        model.train()
        epoch_loss = 0
        print(f"\nGlobal training step: {global_step+1}/{args.max_iterations}")

        for step, batch in enumerate(tqdm(train_loader, desc="Training", disable=True), 1):
            x, y = batch["image"].to(device), batch["label"].to(device)
            logits = model(x)
            loss = loss_function(logits, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

            print(f"Train loss at step {step}/{len(train_loader)}: {loss:.4f}")

        epoch_loss_avg = epoch_loss / step
        epoch_loss_values.append(epoch_loss_avg)
        print(f"Epoch {global_step} average loss: {epoch_loss_avg:.4f}")

        # Validation
        if ((global_step+1) % args.eval_num == 0 and global_step != 0) or global_step == args.max_iterations-1:
            model.eval()
            print(f"Validation run at global iteration {global_step+1}")
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation", disable=True):
                    val_inputs, val_labels = batch["image"].to(device), batch["label"].to(device)
                    with torch.amp.autocast("cuda", enabled=True):
                        val_outputs = model(val_inputs)

                    val_labels_convert = [post_label(i) for i in decollate_batch(val_labels)]
                    val_outputs_convert = [post_pred(i) for i in decollate_batch(val_outputs)]

                    dice_metric(y_pred=val_outputs_convert, y=val_labels_convert)
                    _ = torch.sigmoid(val_outputs)

                mean_dice_val = dice_metric.aggregate().item()
                dice_metric.reset()
                print(f"Validation dice at global step {global_step+1}/{args.max_iterations}: {mean_dice_val:.4f}")

                metric_values.append(mean_dice_val)
                if mean_dice_val > dice_val_best:
                    dice_val_best = mean_dice_val
                    global_step_best = global_step
                    torch.save(model.state_dict(), os.path.join(root_dir, f"best_metric_model_{current_date}.pth"))

    # Save final model
    torch.save(model.state_dict(), os.path.join(root_dir, f"final_model_FocalLoss_{current_date}.pth"))

    # Save metrics
    json_data = {
        "global_step": list(range(args.max_iterations)),
        "loss": epoch_loss_values,
        "dice_metric": metric_values
    }
    with open(os.path.join(root_dir, f"data_metrics_{current_date}.json"), "w") as f:
        json.dump(json_data, f, indent=4)

    print(f"\nTraining completed. Best metric: {dice_val_best:.4f} at iteration {global_step_best}")


if __name__ == "__main__":
    main()