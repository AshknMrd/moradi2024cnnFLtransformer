# moradi2024cnnFLtransformer
This project investigates the impact of the local model architecture on the performance of federated learning–based clinically significant prostate cancer detection.


Before running preprocessing, make sure that the raw data folders are structured correctly:

```
workdir/  
├── nnUNet_raw/  
│   └── Dataset104_picai/  
│       ├── imagesTr/  
│       ├── labelsTr/  
│       ├── dataset.json
│       └── dataset_unetr.json
├── nnUNet_preprocessed/  
└── nnUNet_results/  
```

where the `dataset.json` is used for the nnUNet experiments and is 
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

while the `dataset_unetr.json` is used for the tranformer-based and DynUNet experiments as: 

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
    "numTraining": 1200,
    "file_ending": ".nii.gz",
    "name": "picai_nnunetv2",
    "reference": "none",
    "release": "1.0",
    "description": "bpMRI scans from PI-CAI dataset to train by nnUNetv2",
    "overwrite_image_reader_writer": "SimpleITKIO",
    "training": [
        {
            "image": [
                "./imagesTr/10000_1000000_0000.nii.gz",
                "./imagesTr/10000_1000000_0001.nii.gz",
                "./imagesTr/10000_1000000_0002.nii.gz"
            ],
            "label": "./labelsTr/10000_1000000.nii.gz"
        }
    ],
    "validation": [
        {
            "image": [
                "./imagesTr/10015_1000015_0000.nii.gz",
                "./imagesTr/10015_1000015_0001.nii.gz",
                "./imagesTr/10015_1000015_0002.nii.gz"
            ],
            "label": "./labelsTr/10015_1000015.nii.gz"
        }
    ]
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


## Transformer-based model training - SwinUNetr

```bash
python swinunetr_train.py \
    --datasets_json ./workdir/dataset.json \
    --root_dir ./workdir \
    --num_train 100 \
    --num_val 20 \
    --max_iterations 5 \
    --batch_size 1 \
    --eval_num 1
```