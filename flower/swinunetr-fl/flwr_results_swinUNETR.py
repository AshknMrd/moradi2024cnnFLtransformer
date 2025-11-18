import os
import warnings
import json
import glob
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import monai
from monai.transforms import (
    AsDiscrete, EnsureChannelFirstd, Compose, LoadImaged, Orientationd,
    RandFlipd, RandRotate90d, RandShiftIntensityd, SpatialPadd,
    CenterSpatialCropd, Spacingd, ScaleIntensityRangePercentilesd)
from monai.data import (
    DataLoader, Dataset, load_decathlon_datalist, decollate_batch)
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import SaveImaged

from picai_eval import evaluate
from report_guided_annotation import extract_lesion_candidates

warnings.simplefilter(action="ignore", category=FutureWarning)

# =========================
# Paths & Constants
# =========================
root_dir = "./flwr_output/"
datasets_json = "dataset_unetr_picai.json"
results_json = os.path.join(root_dir, "results.json")

test_dataset_json = "dataset_test.json"
global_model_dir = "./flower/global_model_round10.pt"


prediction_dir = os.path.join(root_dir, "prediction")
gt_dir = "./picai_test/gt"
os.makedirs(prediction_dir, exist_ok=True)
os.makedirs(gt_dir, exist_ok=True)

# =========================
# Training / Validation Transforms
# =========================
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(0.5, 0.5, 3.0),  mode=("bilinear", "nearest")),
    SpatialPadd(keys=["image", "label"], spatial_size=(256, 256, 32)),
    ScaleIntensityRangePercentilesd(keys=["image"], lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True),
    RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
    RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
    RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
    RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
    RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(0.5, 0.5, 3.0), mode=("bilinear", "nearest")),
    SpatialPadd(keys=["image", "label"], spatial_size=(256, 256, 32)),
    CenterSpatialCropd(keys=["image", "label"], roi_size=(256, 256, 32)),
    ScaleIntensityRangePercentilesd(keys=["image"], lower=1, upper=99,  b_min=0.0, b_max=1.0, clip=True)])

# =========================
# Model & Training Setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SwinUNETR(
    in_channels=3,
    out_channels=2,
    img_size=(256, 256, 32),
    feature_size=48,
    drop_rate=0.02,
    dropout_path_rate=0.1,
    use_checkpoint=True,
    use_v2=True
).to(device)

post_label = AsDiscrete(to_onehot=2)
post_pred = AsDiscrete(argmax=True, to_onehot=2)
dice_metric = DiceMetric(include_background=False, reduction="mean")

# =========================
# Plotting Learning Curves
# =========================
with open(results_json, "r") as f:
    results = json.load(f)

loss_vals = results["loss"]
dice_vals = results["dice"]

plt.figure("training", (12, 6))

plt.subplot(1, 2, 1)
plt.title("Iteration Average Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim(0, 0.005)
plt.plot(loss_vals, label="Training Loss", color="blue", marker="o")
plt.legend()

plt.subplot(1, 2, 2)
plt.title("Validation Mean Dice")
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.plot(dice_vals, label="Validation Dice", color="green", marker="x")
plt.legend()

plt.show()

# =========================
# Test Set Setup
# =========================

test_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(0.5, 0.5, 3.0), mode=("bilinear", "nearest")),
    SpatialPadd(keys=["image", "label"], spatial_size=(256, 256, 32)),
    CenterSpatialCropd(keys=["image", "label"], roi_size=(256, 256, 32)),
    ScaleIntensityRangePercentilesd(keys=["image"], lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True)])
test_files = load_decathlon_datalist(test_dataset_json, True, "test")
test_loader = DataLoader(test_files, batch_size=1, shuffle=False, transform=test_transforms, num_workers=4, pin_memory=True)

# =========================
# Load Model for Inference
# =========================
model.load_state_dict(torch.load(global_model_dir))
model.eval()

# =========================
# Prediction Saving Setup
# =========================
save_pred = SaveImaged(
    keys="pred", output_dir=prediction_dir, output_postfix="pred",
    resample=False, separate_folder=False, print_log=False,
    meta_key_postfix="meta_dict", output_ext=".nii.gz")
save_gt = SaveImaged(
    keys="gt", output_dir=gt_dir, output_postfix="gt",
    resample=False, separate_folder=False, print_log=False,
    meta_key_postfix="meta_dict", output_ext=".nii.gz")

# =========================
# Inference Loop
# =========================
with torch.no_grad():
    for batch in tqdm(test_loader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)

        labels_post = [post_label(i) for i in decollate_batch(labels)]
        preds_post = [post_pred(i) for i in decollate_batch(outputs)]

        dice_metric(y_pred=preds_post, y=labels_post)

        pred_soft = torch.softmax(outputs, dim=1)[:, 1]

        meta = {
            "filename_or_obj": batch["label_meta_dict"]["filename_or_obj"][0],
            "affine": batch["label_meta_dict"]["affine"][0],}
        save_pred({"pred": pred_soft[0], "meta_dict": meta})
        save_gt({"gt": labels_post[0], "meta_dict": meta})

    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()

print(mean_dice)

# =========================
# Evaluation (PICAI)
# =========================
pred_cases = sorted(glob.glob(f"{prediction_dir}/**/*.nii.gz", recursive=True))
gt_cases = sorted(glob.glob(f"{gt_dir}/**/*.nii.gz", recursive=True))

metrics = evaluate(
    y_det=pred_cases,
    y_true=gt_cases,
    y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred, threshold="dynamic")[0])

print()
print("AUROC:", round(metrics.auroc, 4))
print("AP:", round(metrics.AP, 4))
print("PICAI score:", round(0.5 * (metrics.auroc + metrics.AP), 4))