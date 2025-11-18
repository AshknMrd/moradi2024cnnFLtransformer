"""DynUNet FL: Flower / PyTorch server application."""

import os
import json
import warnings
from datetime import date
from typing import List, Tuple

import torch
from monai.networks.nets import DynUNet
from monai.transforms import AsDiscrete

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from first_example.task import get_weights, set_weights, test, load_data
from first_example.new_strategy import CustomFedAvg

warnings.simplefilter("ignore", DeprecationWarning)


# -------------------------------------------------------------------------
# Evaluation callback
# -------------------------------------------------------------------------
def get_evaluate_fn(save_dir: str, valloader, device: torch.device):
    """Return a callback that evaluates the global model."""

    def evaluate(server_round, parameters_ndarrays, config):
        model = DynUNet(
            spatial_dims=3,
            in_channels=3,
            out_channels=2,
            filters=[32, 64, 128, 256, 320],
            kernel_size=[3, 3, 3, 3, 3],
            strides=[1, 2, 2, 2, 2],
            upsample_kernel_size=[2, 2, 2, 2],
            norm_name="INSTANCE",
            res_block=True,
        )

        post_label = AsDiscrete(to_onehot=2)
        post_pred = AsDiscrete(argmax=True, to_onehot=2)

        set_weights(model, parameters_ndarrays)
        model.to(device)

        loss, dice = test(model, valloader, device, post_label, post_pred)

        results = {
            "Loss": round(loss, 3),
            "Dice": round(dice, 2),
        }

        with open(os.path.join(save_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)

        return loss, {"cen_dice": dice}

    return evaluate


# -------------------------------------------------------------------------
# Metrics aggregation
# -------------------------------------------------------------------------
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Compute a weighted average of Dice scores."""
    weighted_dice = [n * m["dice"] for n, m in metrics]
    total_examples = sum(n for n, _ in metrics)
    return {"dice": sum(weighted_dice) / total_examples}


# -------------------------------------------------------------------------
# Per-round training configuration
# -------------------------------------------------------------------------
def on_fit_config(server_round: int) -> Metrics:
    """Update learning rate based on round."""
    lr = 0.005 if server_round > 2 else 0.01
    return {"lr": lr}


# -------------------------------------------------------------------------
# Server application
# -------------------------------------------------------------------------
def server_fn(context: Context):
    """Construct and return the Flower server components."""

    today = date.today().isoformat()
    base_dir = context.run_config["save-dir"]
    save_dir = os.path.join(base_dir, today)
    os.makedirs(save_dir, exist_ok=True)

    dataset_json = context.run_config["dataset-json"]
    _, valloader = load_data(0, 2, dataset_json, num_train=80, num_val=20)

    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model and parameters
    model = DynUNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=2,
        filters=[32, 64, 128, 256, 320],
        kernel_size=[3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2],
        upsample_kernel_size=[2, 2, 2, 2],
        norm_name="INSTANCE",
        res_block=True,
    )
    initial_parameters = ndarrays_to_parameters(get_weights(model))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    strategy = CustomFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=on_fit_config,
        evaluate_fn=get_evaluate_fn(save_dir, valloader, device),
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)