import os
import json
import glob
import datetime
import warnings
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.data import Dataset, DataLoader, decollate_batch, load_decathlon_datalist
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
                              SpatialPadd, CenterSpatialCropd, ScaleIntensityRangePercentilesd, AsDiscrete, SaveImaged)
from picai_eval import evaluate
from report_guided_annotation import extract_lesion_candidates
warnings.simplefilter(action='ignore', category=FutureWarning)

# ----------------------------
# Configuration
# ----------------------------
root_dir = "./results_directory"
datasets_json = "./workdir/dataset.json"
json_file_path = "./results_data.json"

test_dir = "./workdir/prostate_testset"
test_dataset_json = "./dataset_prostate_test.json"

prediction_dir = './prediction'
os.makedirs(prediction_dir, exist_ok=True)

gt_dir = "./workdir/prostate_testset/gt"
os.makedirs(gt_dir, exist_ok=True)

torch.backends.cudnn.benchmark = True

post_label = AsDiscrete(to_onehot=2)
post_pred = AsDiscrete(argmax=True, to_onehot=2)
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

current_date = datetime.datetime.now().strftime("%Y-%m-%d")

# ----------------------------
# Model, Loss, Optimizer
# ----------------------------
model = SwinUNETR(
    in_channels=3,
    out_channels=2,
    img_size=(256, 256, 32),
    feature_size=48,
    drop_rate=0.02,
    attn_drop_rate=0.0,
    dropout_path_rate=0.1,
    use_checkpoint=True,
    use_v2=True
).to(device)

# ----------------------------
# Plot training metrics
# ----------------------------
with open(json_file_path, "r") as f:
    data_scores = json.load(f)

plt.figure("Training Metrics", (12, 6))

# Loss
plt.subplot(1, 2, 1)
plt.plot(data_scores["loss"], label="Training Loss", color="blue", marker="o")
plt.title("Iteration Average Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim(0, 0.005)
plt.legend()

# Dice
plt.subplot(1, 2, 2)
plt.plot(data_scores["dice_metric"], label="Validation Dice", color="green", marker="x")
plt.title("Validation Mean Dice")
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.legend()
plt.show()

# ----------------------------
# Test set evaluation
# ----------------------------
test_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(0.5, 0.5, 3.0), mode=("bilinear", "nearest")),
    SpatialPadd(keys=["image", "label"], spatial_size=(256, 256, 32)),
    CenterSpatialCropd(keys=["image", "label"], roi_size=(256, 256, 32)),
    ScaleIntensityRangePercentilesd(keys=["image"], lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True),
])

test_files = load_decathlon_datalist(test_dataset_json, True, "test")
test_loader = DataLoader(Dataset(data=test_files, transform=test_transforms),
                         batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

# Load trained model
model.load_state_dict(torch.load(os.path.join(root_dir, "final_model_FocalLoss.pth")))
model.eval()

save_pred = SaveImaged(keys="pred", output_dir=prediction_dir, output_postfix="pred", resample=False, separate_folder=False, print_log=False, meta_key_postfix="meta_dict", output_ext=".nii.gz")
save_gt = SaveImaged(keys="gt", output_dir=gt_dir, output_postfix="gt", resample=False, separate_folder=False, print_log=False, meta_key_postfix="meta_dict", output_ext=".nii.gz")

with torch.no_grad():
    for batch in tqdm(test_loader):
        test_inputs, test_label = batch["image"].to(device), batch["label"].to(device)
        test_output = model(test_inputs)

        test_labels_convert = [post_label(i) for i in decollate_batch(test_label)]
        test_output_convert = [post_pred(i) for i in decollate_batch(test_output)]

        dice_metric(y_pred=test_output_convert, y=test_labels_convert)

        meta_dict = {
            "filename_or_obj": batch["label_meta_dict"]["filename_or_obj"][0],
            "affine": batch["label_meta_dict"]["affine"][0]
        }
        save_pred({"pred": torch.softmax(test_output, dim=1)[:,1,:,:,:][0], "meta_dict": meta_dict})
        save_gt({"gt": test_labels_convert[0], "meta_dict": meta_dict})

mean_dice = dice_metric.aggregate().item()
dice_metric.reset()
print(f"Mean Dice on Test Set: {mean_dice:.4f}")

# ----------------------------
# PICAI evaluation
# ----------------------------
pred_cases = sorted(glob.glob(f"{prediction_dir}/**/*.nii.gz", recursive=True))
gt_cases = sorted(glob.glob(f"{gt_dir}/**/*.nii.gz", recursive=True))

metrics = evaluate(
    y_det=pred_cases,
    y_true=gt_cases,
    y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred, threshold="dynamic")[0]
)

print("\nEvaluation Results:")
print(f"AUROC: {metrics.auroc:.4f}")
print(f"AP: {metrics.AP:.4f}")
print(f"PICAI score: {0.5*(metrics.auroc+metrics.AP):.4f}")