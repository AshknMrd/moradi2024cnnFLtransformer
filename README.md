# moradi2024cnnFLtransformer
In this project, the impact of the local model architecture on the performance of federated learning–based clinically significant prostate cancer detection is investigated.


Before running preprocessing, make sure that the raw data folders are structured correctly:

```
workdir/  
├── nnUNet_raw/  
│   └── Dataset104_picai/  
│       ├── imagesTr/  
│       ├── labelsTr/  
│       └── dataset.json
├── nnUNet_preprocessed/  
└── nnUNet_results/  
```

where the `dataset.json` is 
```json
{
    "channel_names": {
        "0": "T2W",
        "1": "ADC",
        "2": "HBV"
    },
    "labels": {
        "background": 0,
        "lesion": 1
    },
    "numTraining": 1500,
    "file_ending": ".nii.gz",
    "name": "picai_nnunetv2",
    "reference": "none",
    "release": "1.0",
    "description": "bpMRI scans from PI-CAI dataset to train by nnUNetv2",
    "overwrite_image_reader_writer": "SimpleITKIO"
}
```

## CNN-based model training - nnUNet V2

**Preprocessing:**

```bash
nnUNetv2_plan_and_preprocess -d 108 -c 3d_fullres --verify_dataset_integrity
```

**Training (folds 0–4):**

```bash
for FOLD in 0 1 2 3 4; do
    nnUNetv2_train Dataset108_picai 3d_fullres "$FOLD" \
        -tr nnUNetTrainerCELoss_1000epochs --npz
done
```

## CNN-based model training - MONAI DynUnet

```bash
python ./cnn_nnunet_monai/monai_dynunet.py \
    --datasets_json ./nnUNet_raw/Dataset108_picai/dataset_unetr_picai.json \
    --root_dir ./workdir \
    --num_train 1000 \
    --num_val 300 \
    --max_iterations 300 \
    --batch_size 2 \
    --eval_num 5
```