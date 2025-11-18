# Creating customized strategy
import os
from datetime import date

import torch
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Parameters, parameters_to_ndarrays
from monai.networks.nets import SwinUNETR

from first_example.task import set_weights


# Output directory
today = date.today().isoformat()
save_dir = f"./flwr_output_swinunetr/{today}"
os.makedirs(save_dir, exist_ok=True)


class CustomFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results_to_save = {}
        self.train_loss_log = {}

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:

        # Perform FedAvg aggregation
        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Convert Flower parameters → numpy arrays
        ndarrays = parameters_to_ndarrays(aggregated_params)

        # Recreate and load global model
        model = SwinUNETR(
            in_channels=3,
            out_channels=2,
            img_size=(256, 256, 32),
            feature_size=48,
            drop_rate=0.02,
            attn_drop_rate=0.0,
            dropout_path_rate=0.1,
            use_checkpoint=True,
            use_v2=True,
        )
        set_weights(model, ndarrays)

        # Save global model snapshot
        model_path = os.path.join(save_dir, f"global_model_round{server_round}.pt")
        torch.save(model.state_dict(), model_path)

        return aggregated_params, aggregated_metrics